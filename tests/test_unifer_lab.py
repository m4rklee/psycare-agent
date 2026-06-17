from io import BytesIO

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from experiments.unifer_lab import server


client = TestClient(server.app)


def _image_bytes(format_name: str = "JPEG") -> bytes:
    image = Image.new("RGB", (16, 16), color=(180, 160, 140))
    output = BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_unifer_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_unifer_lab_models_contract_ollama(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.BACKEND_ENV, "ollama")
    monkeypatch.setenv(server.HF_HOME_ENV, str(tmp_path / "hf-cache"))
    monkeypatch.setattr(server, "ollama_model_registered", lambda model_name=None: False)
    monkeypatch.setattr(server, "gguf_files_present", lambda: False)
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "UniFER-7B"
    assert data["modelId"] == server.MODEL_ID
    assert data["labels"] == server.UNIFER_LABELS
    assert data["backend"] == "ollama"
    assert data["ollamaModel"] == server.DEFAULT_OLLAMA_MODEL
    assert data["runtimeReady"] is False
    assert data["runtimeStatus"] == "model missing, download Karl28/UniFER-7B first"
    assert "create-unifer-ollama-model.sh" in data["installCommand"]


def test_unifer_lab_models_contract_transformers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.BACKEND_ENV, "transformers")
    monkeypatch.setenv(server.HF_HOME_ENV, str(tmp_path / "hf-cache"))
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "transformers"
    assert data["runtimeReady"] is False
    assert data["runtimeStatus"] == "model missing"
    assert data["localFilesOnly"] is True
    assert "torchvision" in data["installCommand"]


def test_analyze_frame_rejects_empty_file() -> None:
    response = client.post(
        "/analyze-frame",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "图片帧不能为空"


def test_analyze_frame_rejects_non_image_file() -> None:
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", b"not-image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "无法解析图片帧"


def test_analyze_frame_rejects_oversized_file() -> None:
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", b"x" * (server.MAX_IMAGE_BYTES + 1), "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "图片帧不能超过 8MB"


def test_analyze_frame_returns_503_when_transformers_model_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.BACKEND_ENV, "transformers")
    monkeypatch.setenv(server.HF_HOME_ENV, str(tmp_path / "hf-cache"))
    monkeypatch.setenv(server.LOCAL_ONLY_ENV, "true")
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "model missing" in response.json()["detail"]


def test_analyze_frame_returns_503_when_ollama_missing(monkeypatch) -> None:
    monkeypatch.setenv(server.BACKEND_ENV, "ollama")
    monkeypatch.setattr(server, "_ollama_runtime_status", lambda: (False, "ollama model missing"))
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "Ollama runtime not ready" in response.json()["detail"]


def test_analyze_frame_normalizes_happiness_answer(monkeypatch) -> None:
    def fake_predict(image: Image.Image) -> dict[str, object]:
        assert image.size == (16, 16)
        return {
            "modelResponse": (
                "<think>The mouth corners are raised and the face appears relaxed.</think>"
                "<answer>happiness</answer>"
            )
        }

    monkeypatch.setattr(server, "predict_with_unifer", fake_predict)
    response = client.post(
        "/analyze-frame",
        data={
            "preprocessMode": "mediapipe_affine_align",
            "cropBox": '{"x":10,"y":12,"width":80,"height":90}',
            "sourceWidth": "640",
            "sourceHeight": "480",
            "outputSize": "224",
            "fallback": "false",
        },
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "NORMAL"
    assert data["visualEmotion"] == "NORMAL"
    assert data["riskLevel"] == "LOW"
    assert data["features"]["rawEmotion"] == "happiness"
    assert data["features"]["answer"] == "happiness"
    assert data["features"]["parseStatus"] == "answer_tag"
    assert data["features"]["preprocess"]["mode"] == "mediapipe_affine_align"
    assert data["features"]["preprocess"]["outputSize"] == 224
    assert data["features"]["preprocess"]["fallback"] is False


def test_predict_with_ollama_uses_chat_api(monkeypatch) -> None:
    monkeypatch.setenv(server.BACKEND_ENV, "ollama")
    monkeypatch.setattr(server, "_ollama_runtime_status", lambda: (True, "ready"))

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "message": {
                    "content": (
                        "<think>Relaxed face.</think>"
                        "<answer>neutral</answer>"
                    )
                }
            }

    captured: dict[str, object] = {}

    class FakeClient:
        def post(self, url: str, json: dict[str, object], timeout: float) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(httpx, "post", FakeClient().post)
    image = Image.new("RGB", (224, 224), color=(180, 160, 140))
    result = server._predict_with_ollama(image)
    assert result["modelResponse"].endswith("<answer>neutral</answer>")
    assert captured["url"] == f"{server.DEFAULT_OLLAMA_BASE_URL}/api/chat"
    payload = captured["json"]
    assert payload["model"] == server.DEFAULT_OLLAMA_MODEL
    assert payload["messages"][0]["content"] == server.PROMPT
    assert payload["messages"][0]["images"]
    assert payload["options"]["num_predict"] == server.get_max_new_tokens()


def test_unifer_label_mapping() -> None:
    sad = server.normalize_prediction({"modelResponse": "<answer>sadness</answer>"})
    assert sad["visualEmotion"] == "DEPRESSED"
    assert sad["riskLevel"] == "MEDIUM"

    fear = server.normalize_prediction({"modelResponse": "<answer>fear</answer>"})
    assert fear["visualEmotion"] == "HIGH_RISK"
    assert fear["riskLevel"] == "HIGH"

    text_match = server.normalize_prediction({"modelResponse": "The person looks happy."})
    assert text_match["features"]["rawEmotion"] == "happiness"
    assert text_match["features"]["parseStatus"] == "text_match"
