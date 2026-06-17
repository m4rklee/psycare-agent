from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from experiments.emotiefflib_lab import server


client = TestClient(server.app)


def _image_bytes(format_name: str = "JPEG") -> bytes:
    image = Image.new("RGB", (16, 16), color=(180, 160, 140))
    output = BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_emotieff_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_emotieff_lab_models_contract() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "onnx" in data["engines"]
    assert "enet_b0_8_best_vgaf" in data["models"]
    assert "enet_b2_8" in data["models"]
    assert data["defaultEngine"] == "onnx"
    assert data["defaultModel"] == "enet_b2_8"


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


def test_analyze_frame_normalizes_mocked_prediction(monkeypatch) -> None:
    def fake_predict(frame, *, engine: str, model: str, device: str) -> dict[str, object]:
        assert frame.shape == (16, 16, 3)
        assert engine == "onnx"
        assert model == "enet_b2_8"
        assert device == "cpu"
        return {
            "rawEmotion": "Sadness",
            "confidence": 0.77,
            "probabilities": {"Sadness": 0.77, "Neutral": 0.12},
        }

    monkeypatch.setattr(server, "predict_with_emotieff", fake_predict)
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "DEPRESSED"
    assert data["visualEmotion"] == "DEPRESSED"
    assert data["score"] >= 3.0
    assert data["riskLevel"] == "MEDIUM"
    assert data["confidence"] == 0.77
    assert data["features"]["rawEmotion"] == "Sadness"
    assert data["features"]["probabilities"]["Sadness"] == 0.77


def test_emotion_mapping_for_high_and_normal_labels() -> None:
    high = server.normalize_prediction({"rawEmotion": "Fear", "confidence": 0.8})
    assert high["visualEmotion"] == "HIGH_RISK"
    assert high["visualScore"] >= 3.0

    normal = server.normalize_prediction({"rawEmotion": "Neutral", "confidence": 0.8})
    assert normal["visualEmotion"] == "NORMAL"
    assert normal["riskLevel"] == "LOW"
