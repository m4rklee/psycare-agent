from fastapi.testclient import TestClient

from experiments.realtime_camera_lab.server import app


client = TestClient(app)


def test_realtime_lab_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_analyze_frame_normalizes_facemesh_payload() -> None:
    response = client.post(
        "/analyze-frame",
        json={
            "faceDetected": True,
            "landmarkCount": 478,
            "score": 2.35,
            "confidence": 0.81,
            "evidence": "MediaPipe FaceLandmarker 浏览器端分析。",
            "features": {
                "browTension": 0.18,
                "eyeTension": 0.22,
                "mouthDown": 0.14,
                "muscleTension": 0.31,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "ANXIETY"
    assert data["visualEmotion"] == "ANXIETY"
    assert data["score"] == 2.35
    assert data["visualScore"] == 2.35
    assert data["riskLevel"] == "LOW"
    assert data["confidence"] == 0.81
    assert data["features"]["faceDetected"] is True
    assert data["features"]["landmarkCount"] == 478
    assert data["timestamp"]


def test_analyze_frame_accepts_no_face_payload() -> None:
    response = client.post(
        "/analyze-frame",
        json={
            "faceDetected": False,
            "score": 0.0,
            "confidence": 0.32,
            "features": {"faceDetected": False, "landmarkCount": 0},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["emotion"] == "NORMAL"
    assert data["riskLevel"] == "LOW"
    assert "未检测到人脸" in data["evidence"]


def test_analyze_frame_rejects_invalid_features() -> None:
    response = client.post("/analyze-frame", json={"features": "not-an-object"})

    assert response.status_code == 400
    assert response.json()["detail"] == "features must be an object"


def test_analyze_frame_rejects_face_without_landmarks() -> None:
    response = client.post("/analyze-frame", json={"faceDetected": True, "landmarkCount": 0})

    assert response.status_code == 400
    assert response.json()["detail"] == "landmarkCount is required when faceDetected=true"


def test_analyze_frame_rejects_bad_emotion() -> None:
    response = client.post(
        "/analyze-frame",
        json={
            "faceDetected": True,
            "landmarkCount": 478,
            "emotion": "SURPRISED",
            "score": 0.5,
        },
    )

    assert response.status_code == 400
    assert "emotion must be" in response.json()["detail"]
