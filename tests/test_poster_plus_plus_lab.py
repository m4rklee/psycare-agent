from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from experiments.poster_plus_plus_lab import server


client = TestClient(server.app)


def _image_bytes(format_name: str = "JPEG") -> bytes:
    image = Image.new("RGB", (16, 16), color=(180, 160, 140))
    output = BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_poster_plus_plus_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_poster_plus_plus_lab_models_contract(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "rafdb_best.pth"
    pretrain_dir = tmp_path / "pretrain"
    upstream = tmp_path / "POSTER_V2"
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(server.PRETRAIN_DIR_ENV, str(pretrain_dir))
    monkeypatch.setenv(server.UPSTREAM_ENV, str(upstream))

    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "POSTER++"
    assert data["venue"] == "Pattern Recognition"
    assert data["defaultDataset"] == "RAF-DB"
    assert data["labels"] == server.RAFDB_LABELS
    assert data["checkpointPath"] == str(checkpoint)
    assert data["weightsPresent"] is False
    assert data["pretrainPresent"] is False
    assert data["upstreamPresent"] is False
    assert data["runtimeStatus"] == "weights missing"
    assert "AffectNet-8" in data["datasets"]


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


def test_analyze_frame_returns_503_when_pretrain_missing(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "rafdb_best.pth"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(server.PRETRAIN_DIR_ENV, str(tmp_path / "pretrain"))

    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "pretrain missing" in response.json()["detail"]


def test_analyze_frame_returns_503_when_upstream_missing(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "rafdb_best.pth"
    checkpoint.write_bytes(b"fake")
    pretrain_dir = tmp_path / "pretrain"
    pretrain_dir.mkdir()
    (pretrain_dir / "ir50.pth").write_bytes(b"fake")
    (pretrain_dir / "mobilefacenet_model_best.pth.tar").write_bytes(b"fake")
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(server.PRETRAIN_DIR_ENV, str(pretrain_dir))
    monkeypatch.setenv(server.UPSTREAM_ENV, str(tmp_path / "missing-upstream"))

    response = client.post(
        "/analyze-frame",
        files={"file": ("frame.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "upstream code missing" in response.json()["detail"]


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

    monkeypatch.setattr(server, "predict_with_poster_pp", fake_predict)
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
    assert data["features"]["source"] == "POSTER++"
    assert "不是心理诊断" in data["evidence"]


def test_analyze_frame_includes_preprocess_metadata(monkeypatch) -> None:
    def fake_predict(image: Image.Image) -> dict[str, object]:
        assert image.size == (16, 16)
        return {"rawEmotion": "neutral", "confidence": 0.75}

    monkeypatch.setattr(server, "predict_with_poster_pp", fake_predict)
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
    preprocess = response.json()["features"]["preprocess"]
    assert preprocess["mode"] == "mediapipe_affine_align"
    assert preprocess["cropBox"] == '{"x":10,"y":12,"width":80,"height":90}'
    assert preprocess["sourceWidth"] == 640
    assert preprocess["sourceHeight"] == 480
    assert preprocess["outputSize"] == 224
    assert preprocess["fallback"] is False


def test_emotion_mapping_for_high_and_normal_labels() -> None:
    high = server.normalize_prediction({"rawEmotion": "fear", "confidence": 0.8})
    assert high["visualEmotion"] == "HIGH_RISK"
    assert high["visualScore"] >= 4.0
    assert high["riskLevel"] == "HIGH"

    normal = server.normalize_prediction({"rawEmotion": "neutral", "confidence": 0.8})
    assert normal["visualEmotion"] == "NORMAL"
    assert normal["riskLevel"] == "LOW"
