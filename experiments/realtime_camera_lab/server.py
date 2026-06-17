from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
VALID_EMOTIONS = {"NORMAL", "ANXIETY", "DEPRESSED", "HIGH_RISK"}
VALID_RISKS = {"LOW", "MEDIUM", "HIGH"}

app = FastAPI(
    title="Realtime Camera Face Mesh Lab",
    description="Isolated browser-side MediaPipe FaceLandmarker experiment for multimodalAgent.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "UP"}


@app.post("/analyze-frame")
async def analyze_frame(payload: dict[str, Any] = Body(...)) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    return normalize_facemesh_payload(payload)


def normalize_facemesh_payload(payload: dict[str, Any]) -> dict[str, object]:
    features = payload.get("features", {})
    if features is None:
        features = {}
    if not isinstance(features, dict):
        raise HTTPException(status_code=400, detail="features must be an object")

    face_detected = bool(payload.get("faceDetected", features.get("faceDetected", False)))
    landmark_count = _optional_int(payload.get("landmarkCount", features.get("landmarkCount", 0)))
    if face_detected and landmark_count <= 0:
        raise HTTPException(status_code=400, detail="landmarkCount is required when faceDetected=true")

    score = _number(payload.get("score", payload.get("visualScore", features.get("visualScore", 0.0))), "score")
    score = _clamp(score, 0.0, 4.5)
    emotion = _normalize_emotion(payload.get("emotion") or payload.get("visualEmotion")) or _emotion_from_score(score)
    risk_level = _normalize_risk(payload.get("riskLevel")) or _risk_from_score(score)
    confidence = _number(payload.get("confidence", 0.72 if face_detected else 0.35), "confidence")
    confidence = _clamp(confidence, 0.0, 1.0)
    normalized_features = dict(features)
    normalized_features["faceDetected"] = face_detected
    normalized_features["landmarkCount"] = landmark_count
    normalized_features["visualScore"] = round(score, 2)
    evidence = str(payload.get("evidence") or _default_evidence(face_detected, score, normalized_features))

    return {
        "emotion": emotion,
        "visualEmotion": emotion,
        "score": round(score, 2),
        "visualScore": round(score, 2),
        "riskLevel": risk_level,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "features": normalized_features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _number(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be numeric") from exc


def _optional_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_emotion(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"HIGH", "高风险"}:
        normalized = "HIGH_RISK"
    if normalized in {"SAD", "LOW_MOOD", "低落"}:
        normalized = "DEPRESSED"
    if normalized in {"ANXIOUS", "焦虑"}:
        normalized = "ANXIETY"
    if normalized not in VALID_EMOTIONS:
        raise HTTPException(status_code=400, detail="emotion must be NORMAL, ANXIETY, DEPRESSED, or HIGH_RISK")
    return normalized


def _normalize_risk(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized not in VALID_RISKS:
        raise HTTPException(status_code=400, detail="riskLevel must be LOW, MEDIUM, or HIGH")
    return normalized


def _emotion_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH_RISK"
    if score >= 3.0:
        return "DEPRESSED"
    if score >= 2.0:
        return "ANXIETY"
    return "NORMAL"


def _risk_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def _default_evidence(face_detected: bool, score: float, features: dict[str, Any]) -> str:
    if not face_detected:
        return "MediaPipe FaceLandmarker 未检测到人脸。"
    return (
        "MediaPipe FaceLandmarker 浏览器端分析："
        f"landmarkCount={features.get('landmarkCount', 0)}, visualScore={score:.2f}。"
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
