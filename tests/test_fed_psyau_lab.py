from fastapi.testclient import TestClient

from experiments.fed_psyau_lab import server


client = TestClient(server.app)


def test_fed_psyau_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_fed_psyau_lab_models_contract(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "dfme_best.pth"
    upstream = tmp_path / "FED-PsyAU"
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(server.UPSTREAM_ENV, str(upstream))
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "FED-PsyAU"
    assert data["venue"] == "ICCV 2025"
    assert data["task"] == "micro_expression_recognition"
    assert data["inputType"] == "short_video_clip"
    assert data["dataset"] == "DFME"
    assert data["labels"] == server.DFME_LABELS
    assert data["checkpointPath"] == str(checkpoint)
    assert data["weightsPresent"] is False
    assert data["upstreamPresent"] is False
    assert data["runtimeStatus"] == "weights missing"


def test_models_supports_casme3_labels(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.DATASET_ENV, "CASME3")
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(tmp_path / "casme3_best.pth"))
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "CASME3"
    assert data["labels"] == server.CASME3_LABELS


def test_analyze_clip_rejects_empty_video() -> None:
    response = client.post(
        "/analyze-clip",
        files={"file": ("empty.webm", b"", "video/webm")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "视频片段不能为空"


def test_analyze_clip_rejects_non_video_file() -> None:
    response = client.post(
        "/analyze-clip",
        files={"file": ("frame.jpg", b"not-video", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "仅支持 video/webm 或 video/mp4 短视频片段"


def test_analyze_clip_rejects_oversized_video() -> None:
    response = client.post(
        "/analyze-clip",
        files={"file": ("clip.webm", b"x" * (server.MAX_CLIP_BYTES + 1), "video/webm")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "视频片段不能超过 20MB"


def test_analyze_clip_returns_503_when_weights_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(tmp_path / "missing.pth"))
    response = client.post(
        "/analyze-clip",
        files={"file": ("clip.webm", b"video-bytes", "video/webm")},
    )
    assert response.status_code == 503
    assert "weights missing" in response.json()["detail"]


def test_analyze_clip_returns_503_when_upstream_missing(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "dfme_best.pth"
    checkpoint.write_bytes(b"fake-checkpoint")
    monkeypatch.setenv(server.CHECKPOINT_ENV, str(checkpoint))
    monkeypatch.setenv(server.UPSTREAM_ENV, str(tmp_path / "missing-upstream"))
    response = client.post(
        "/analyze-clip",
        files={"file": ("clip.webm", b"video-bytes", "video/webm")},
    )
    assert response.status_code == 503
    assert "upstream code missing" in response.json()["detail"]


def test_analyze_clip_normalizes_mocked_dfme_prediction(monkeypatch) -> None:
    def fake_predict(clip_bytes: bytes, *, content_type: str) -> dict[str, object]:
        assert clip_bytes == b"video-bytes"
        assert content_type == "video/webm"
        return {
            "dataset": "DFME",
            "microExpression": "Fear",
            "confidence": 0.82,
            "probabilities": {
                "Happiness": 0.01,
                "Surprise": 0.04,
                "Disgust": 0.03,
                "Sadness": 0.05,
                "Anger": 0.02,
                "Fear": 0.82,
                "Contempt": 0.03,
            },
            "auPredictions": {"AU4": 0.71, "AU6": 0.23},
        }

    monkeypatch.setattr(server, "predict_with_fed_psyau", fake_predict)
    response = client.post(
        "/analyze-clip",
        files={"file": ("clip.webm", b"video-bytes", "video/webm")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["microExpression"] == "Fear"
    assert data["emotion"] == "HIGH_RISK"
    assert data["visualEmotion"] == "HIGH_RISK"
    assert data["riskLevel"] == "HIGH"
    assert data["confidence"] == 0.82
    assert data["features"]["probabilities"]["Fear"] == 0.82
    assert data["features"]["auPredictions"]["AU4"] == 0.71
    assert "不是心理诊断" in data["evidence"]


def test_normalize_casme3_prediction() -> None:
    data = server.normalize_prediction(
        {
            "dataset": "CASME3",
            "microExpression": "negative",
            "confidence": 0.7,
            "probabilities": {"positive": 0.1, "negative": 0.7, "surprise": 0.2},
        }
    )
    assert data["microExpression"] == "negative"
    assert data["visualEmotion"] == "ANXIETY"
    assert data["riskLevel"] == "LOW"
