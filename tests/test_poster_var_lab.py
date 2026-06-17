from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from experiments.poster_var_lab import server


client = TestClient(server.app)


def _image_bytes(format_name: str = "JPEG") -> bytes:
    image = Image.new("RGB", (16, 16), color=(180, 160, 140))
    output = BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_poster_var_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_poster_var_lab_models_contract(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "rafdb_best.pth"
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "POSTER-Var"
    assert data["dataset"] == "RAF-DB"
    assert data["labels"] == server.RAFDB_LABELS
    assert data["checkpointPath"] == str(checkpoint)
    assert data["weightsPresent"] is False
    assert data["weightsStatus"] == "weights missing"


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


def test_analyze_frame_returns_503_when_weights_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(tmp_path / "missing.pth"))
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "weights missing" in response.json()["detail"]


def test_analyze_frame_normalizes_mocked_prediction(monkeypatch) -> None:
    def fake_predict(image: Image.Image) -> dict[str, object]:
        assert image.size == (16, 16)
        return {
            "rawEmotion": "sad",
            "confidence": 0.9,
            "probabilities": {
                "surprise": 0.01,
                "fear": 0.02,
                "disgust": 0.01,
                "happy": 0.01,
                "sad": 0.9,
                "angry": 0.02,
                "neutral": 0.03,
            },
        }

    monkeypatch.setattr(server, "predict_with_poster_var", fake_predict)
    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "DEPRESSED"
    assert data["visualEmotion"] == "DEPRESSED"
    assert data["score"] == 3.72
    assert data["riskLevel"] == "MEDIUM"
    assert data["confidence"] == 0.9
    assert data["features"]["rawEmotion"] == "sad"
    assert data["features"]["probabilities"]["sad"] == 0.9
    assert "不是心理诊断" in data["evidence"]


def test_emotion_mapping_for_high_and_normal_labels() -> None:
    high = server.normalize_prediction({"rawEmotion": "fear", "confidence": 0.8})
    assert high["visualEmotion"] == "HIGH_RISK"
    assert high["visualScore"] >= 4.0
    assert high["riskLevel"] == "HIGH"

    normal = server.normalize_prediction({"rawEmotion": "neutral", "confidence": 0.8})
    assert normal["visualEmotion"] == "NORMAL"
    assert normal["riskLevel"] == "LOW"
