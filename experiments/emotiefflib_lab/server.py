from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_ENGINE = "onnx"
DEFAULT_MODEL = "enet_b2_8"
DEFAULT_DEVICE = "cpu"
DEFAULT_ENGINES = ["onnx", "torch"]
DEFAULT_MODELS = [
    "enet_b0_8_best_vgaf",
    "enet_b0_8_best_afew",
    "enet_b2_8",
    "enet_b0_8_va_mtl",
    "enet_b2_7",
    "mbf_va_mtl",
    "mobilevit_va_mtl",
]
DEFAULT_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(
    title="EmotiEffLib Emotion Lab",
    description="Isolated EmotiEffLib ONNX experiment for multimodalAgent.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_recognizer_cache: dict[tuple[str, str, str], Any] = {}


class DecodedImage:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        width, height = image.size
        self.shape = (height, width, 3)


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
async def analyze_frame(
    file: UploadFile = File(...),
    engine: str = Form(DEFAULT_ENGINE),
    model: str = Form(DEFAULT_MODEL),
    device: str = Form(DEFAULT_DEVICE),
) -> dict[str, object]:
    engine = engine.strip() or DEFAULT_ENGINE
    model = model.strip() or DEFAULT_MODEL
    device = device.strip() or DEFAULT_DEVICE
    _validate_engine_model(engine, model)

    content_type = (file.content_type or "").lower()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPEG 或 PNG 图片帧")

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片帧不能为空")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片帧不能超过 8MB")

    frame = decode_image(image_bytes)
    try:
        prediction = predict_with_emotieff(frame, engine=engine, model=model, device=device)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - protects the experiment UI from raw stack traces.
        raise HTTPException(status_code=503, detail=f"EmotiEffLib 分析失败：{exc}") from exc

    return normalize_prediction(prediction, engine=engine, model=model, device=device)


def get_model_metadata() -> dict[str, object]:
    available = True
    import_error = None
    engines = DEFAULT_ENGINES
    models = DEFAULT_MODELS
    try:
        from emotiefflib.facial_analysis import get_model_list, get_supported_engines

        engines = list(get_supported_engines())
        models = list(get_model_list())
    except Exception as exc:  # pragma: no cover - depends on optional local experiment deps.
        available = False
        import_error = str(exc)
    return {
        "available": available,
        "engines": engines,
        "models": models,
        "defaultEngine": DEFAULT_ENGINE,
        "defaultModel": DEFAULT_MODEL,
        "defaultDevice": DEFAULT_DEVICE,
        "importError": import_error,
        "installCommand": (
            "uv run --with emotiefflib==1.1.1 --with opencv-python-headless --with numpy "
            "uvicorn experiments.emotiefflib_lab.server:app --host 127.0.0.1 --port 8092"
        ),
    }


def _validate_engine_model(engine: str, model: str) -> None:
    metadata = get_model_metadata()
    engines = set(metadata["engines"])
    models = set(metadata["models"])
    if engine not in engines:
        raise HTTPException(status_code=400, detail=f"engine must be one of: {', '.join(sorted(engines))}")
    if model not in models:
        raise HTTPException(status_code=400, detail=f"model must be one of: {', '.join(sorted(models))}")


def decode_image(image_bytes: bytes) -> Any:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=400, detail="仅支持 JPEG 或 PNG 图片帧")
            rgb_image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="无法解析图片帧") from exc
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is supplied by experiment command.
        return DecodedImage(rgb_image)
    return np.asarray(rgb_image)


def predict_with_emotieff(frame: Any, *, engine: str, model: str, device: str) -> dict[str, object]:
    try:
        import numpy as np
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    except Exception as exc:  # pragma: no cover - optional experiment dependency.
        raise RuntimeError(
            "缺少 EmotiEffLib 依赖，请按 README 使用 uv --with emotiefflib==1.1.1 启动实验服务"
        ) from exc

    recognizer = _get_recognizer(engine=engine, model=model, device=device, factory=EmotiEffLibRecognizer)
    if isinstance(frame, DecodedImage):
        frame = np.asarray(frame.image)
    raw_emotion, raw_scores = recognizer.predict_emotions(frame, logits=False)
    emotion = _first_emotion(raw_emotion)
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim == 2:
        scores = scores[0]
    labels = _labels_from_recognizer(recognizer, len(scores))
    probabilities = {label: round(float(score), 4) for label, score in zip(labels, scores, strict=False)}
    confidence = float(max(probabilities.values())) if probabilities else 0.0
    return {
        "rawEmotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _get_recognizer(*, engine: str, model: str, device: str, factory: Any) -> Any:
    key = (engine, model, device)
    recognizer = _recognizer_cache.get(key)
    if recognizer is None:
        recognizer = factory(engine=engine, model_name=model, device=device)
        _recognizer_cache[key] = recognizer
    return recognizer


def _first_emotion(raw_emotion: Any) -> str:
    if isinstance(raw_emotion, (list, tuple)):
        if not raw_emotion:
            return "Unknown"
        return str(raw_emotion[0])
    return str(raw_emotion)


def _labels_from_recognizer(recognizer: Any, count: int) -> list[str]:
    mapping = getattr(recognizer, "idx_to_emotion_class", None)
    if isinstance(mapping, dict):
        return [str(mapping.get(index, DEFAULT_LABELS[index] if index < len(DEFAULT_LABELS) else index)) for index in range(count)]
    return [DEFAULT_LABELS[index] if index < len(DEFAULT_LABELS) else f"class_{index}" for index in range(count)]


def normalize_prediction(
    prediction: dict[str, object], *, engine: str = DEFAULT_ENGINE, model: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE
) -> dict[str, object]:
    raw_emotion = str(prediction.get("rawEmotion") or "Unknown")
    confidence = _clamp(float(prediction.get("confidence") or 0.0), 0.0, 1.0)
    probabilities = prediction.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        probabilities = {}

    visual_emotion, visual_score = _map_emotion(raw_emotion, confidence)
    risk_level = _risk_from_score(visual_score)
    features = {
        "rawEmotion": raw_emotion,
        "probabilities": probabilities,
        "engine": engine,
        "model": model,
        "device": device,
        "source": "EmotiEffLib",
    }
    evidence = (
        "EmotiEffLib 面部表情分类："
        f"rawEmotion={raw_emotion}, confidence={confidence:.2f}, "
        f"model={model}, engine={engine}。"
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


def _map_emotion(raw_emotion: str, confidence: float) -> tuple[str, float]:
    normalized = raw_emotion.strip().lower()
    if normalized in {"fear", "anger", "disgust", "contempt"}:
        return "HIGH_RISK", 3.4 + confidence * 1.1
    if normalized == "sadness":
        return "DEPRESSED", 3.0 + confidence * 0.8
    if normalized in {"neutral", "happiness", "surprise"}:
        return "NORMAL", confidence * 0.8
    return "ANXIETY", 2.0 + confidence * 0.8


def _risk_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
