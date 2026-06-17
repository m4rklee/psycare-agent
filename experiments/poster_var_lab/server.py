from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = BASE_DIR.parents[1]

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "poster-var" / "checkpoints" / "rafdb_best.pth"
CHECKPOINT_ENV = "POSTER_VAR_CHECKPOINT"
UPSTREAM_ENV = "POSTER_VAR_UPSTREAM"
RAFDB_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(
    title="POSTER-Var SOTA Lab",
    description="Isolated POSTER-Var facial-expression-recognition experiment for multimodalAgent.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/models")
async def models() -> dict[str, object]:
    return get_model_metadata()


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)) -> dict[str, object]:
    content_type = (file.content_type or "").lower()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPEG 或 PNG 图片帧")

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片帧不能为空")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片帧不能超过 8MB")

    image = decode_image(image_bytes)
    try:
        prediction = predict_with_poster_var(image)
    except PosterVarUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - protects the experiment UI from raw traces.
        raise HTTPException(status_code=503, detail=f"POSTER-Var 分析失败：{exc}") from exc

    return normalize_prediction(prediction)


class PosterVarUnavailable(RuntimeError):
    pass


def get_checkpoint_path() -> Path:
    return Path(os.getenv(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT))).expanduser()


def get_upstream_path() -> Path | None:
    value = os.getenv(UPSTREAM_ENV)
    if not value:
        return None
    return Path(value).expanduser()


def get_model_metadata() -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    upstream_path = get_upstream_path()
    weights_present = checkpoint_path.exists()
    upstream_present = bool(upstream_path and upstream_path.exists())
    return {
        "model": "POSTER-Var",
        "dataset": "RAF-DB",
        "labels": RAFDB_LABELS,
        "checkpointPath": str(checkpoint_path),
        "weightsPresent": weights_present,
        "weightsStatus": "ready" if weights_present else "weights missing",
        "upstreamPath": str(upstream_path) if upstream_path else None,
        "upstreamPresent": upstream_present,
        "source": "https://github.com/lg2578/poster-var",
        "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12923884/",
        "installCommand": (
            f"{CHECKPOINT_ENV}=models/poster-var/checkpoints/rafdb_best.pth "
            "uv run --with torch --with torchvision --with timm --with einops "
            "--with opencv-python-headless --with pyyaml --with numpy --with pillow "
            "uvicorn experiments.poster_var_lab.server:app --host 127.0.0.1 --port 8094"
        ),
    }


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=400, detail="仅支持 JPEG 或 PNG 图片帧")
            return image.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="无法解析图片帧") from exc


def predict_with_poster_var(image: Image.Image) -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    if not checkpoint_path.exists():
        raise PosterVarUnavailable(
            f"POSTER-Var weights missing: place RAF-DB checkpoint at {checkpoint_path}"
        )

    upstream_path = get_upstream_path()
    if not upstream_path or not upstream_path.exists():
        raise PosterVarUnavailable(
            f"POSTER-Var upstream code missing: set {UPSTREAM_ENV} to a local clone of lg2578/poster-var"
        )

    raise PosterVarUnavailable(
        "POSTER-Var runtime adapter is not configured yet. "
        "Place upstream code and wire its model constructor before real inference."
    )


def normalize_prediction(prediction: dict[str, object]) -> dict[str, object]:
    raw_emotion = str(prediction.get("rawEmotion") or prediction.get("label") or "neutral").strip().lower()
    probabilities = prediction.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        probabilities = {}
    probabilities = _normalized_probabilities(probabilities)
    if raw_emotion not in RAFDB_LABELS:
        raw_emotion = _dominant_from_probabilities(probabilities)
    confidence = _clamp(float(prediction.get("confidence") or probabilities.get(raw_emotion, 0.0)), 0.0, 1.0)
    visual_emotion, visual_score = _map_emotion(raw_emotion, confidence)
    risk_level = _risk_from_score(visual_score)
    features = {
        "rawEmotion": raw_emotion,
        "probabilities": probabilities,
        "dataset": "RAF-DB",
        "labels": RAFDB_LABELS,
        "source": "POSTER-Var",
        "checkpointPath": str(get_checkpoint_path()),
    }
    evidence = (
        "POSTER-Var RAF-DB 表情分类："
        f"rawEmotion={raw_emotion}, confidence={confidence:.2f}。"
        "该视觉结果是表情弱信号，不是心理诊断。"
    )
    return {
        "emotion": visual_emotion,
        "visualEmotion": visual_emotion,
        "score": round(visual_score, 2),
        "visualScore": round(visual_score, 2),
        "riskLevel": risk_level,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _normalized_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label in RAFDB_LABELS:
        value = probabilities.get(label, probabilities.get(label.capitalize(), 0.0))
        try:
            normalized[label] = round(_clamp(float(value), 0.0, 1.0), 4)
        except (TypeError, ValueError):
            normalized[label] = 0.0
    return normalized


def _dominant_from_probabilities(probabilities: dict[str, float]) -> str:
    if not probabilities:
        return "neutral"
    return max(probabilities.items(), key=lambda item: item[1])[0]


def _map_emotion(raw_emotion: str, confidence: float) -> tuple[str, float]:
    if raw_emotion == "sad":
        return "DEPRESSED", 3.0 + confidence * 0.8
    if raw_emotion in {"fear", "angry", "disgust"}:
        return "HIGH_RISK", 3.2 + confidence * 1.2
    return "NORMAL", confidence * 0.8


def _risk_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
