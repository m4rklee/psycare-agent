from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = BASE_DIR.parents[1]

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "poster-plus-plus" / "checkpoints" / "rafdb_best.pth"
DEFAULT_PRETRAIN_DIR = REPO_ROOT / "models" / "poster-plus-plus" / "pretrain"
CHECKPOINT_ENV = "POSTER_PP_CHECKPOINT"
PRETRAIN_DIR_ENV = "POSTER_PP_PRETRAIN_DIR"
UPSTREAM_ENV = "POSTER_PP_UPSTREAM"
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
RAFDB_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
AFFECTNET_8_LABELS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry", "contempt"]

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, Any] = {}

app = FastAPI(
    title="POSTER++ Lab",
    description="Isolated POSTER++ / POSTER V2 facial-expression-recognition experiment.",
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
async def analyze_frame(
    file: UploadFile = File(...),
    preprocess_mode: str | None = Form(default=None, alias="preprocessMode"),
    crop_box: str | None = Form(default=None, alias="cropBox"),
    source_width: int | None = Form(default=None, alias="sourceWidth"),
    source_height: int | None = Form(default=None, alias="sourceHeight"),
    output_size: int | None = Form(default=None, alias="outputSize"),
    fallback: str | None = Form(default=None, alias="fallback"),
) -> dict[str, object]:
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
        prediction = predict_with_poster_pp(image)
    except PosterPPUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - keeps the lab UI from exposing raw traces.
        raise HTTPException(status_code=503, detail=f"POSTER++ 分析失败：{exc}") from exc

    preprocess = build_preprocess_info(
        preprocess_mode,
        crop_box,
        source_width,
        source_height,
        output_size,
        fallback,
    )
    return normalize_prediction(prediction, preprocess=preprocess)


class PosterPPUnavailable(RuntimeError):
    pass


def get_checkpoint_path() -> Path:
    return Path(os.getenv(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT))).expanduser()


def get_pretrain_dir() -> Path:
    return Path(os.getenv(PRETRAIN_DIR_ENV, str(DEFAULT_PRETRAIN_DIR))).expanduser()


def get_upstream_path() -> Path | None:
    value = os.getenv(UPSTREAM_ENV)
    if not value:
        return None
    return Path(value).expanduser()


def get_pretrain_paths() -> dict[str, Path]:
    pretrain_dir = get_pretrain_dir()
    return {
        "ir50": pretrain_dir / "ir50.pth",
        "mobilefacenet": pretrain_dir / "mobilefacenet_model_best.pth.tar",
    }


def get_model_metadata() -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    upstream_path = get_upstream_path()
    pretrain_paths = get_pretrain_paths()
    missing_pretrain = [name for name, path in pretrain_paths.items() if not path.exists()]
    weights_present = checkpoint_path.exists()
    upstream_present = bool(upstream_path and upstream_path.exists())
    pretrain_present = not missing_pretrain
    runtime_status = "ready"
    if not weights_present:
        runtime_status = "weights missing"
    elif not pretrain_present:
        runtime_status = "pretrain missing"
    elif not upstream_present:
        runtime_status = "upstream missing"
    return {
        "model": "POSTER++",
        "aliases": ["POSTER V2"],
        "venue": "Pattern Recognition",
        "task": "facial_expression_recognition",
        "defaultDataset": "RAF-DB",
        "datasets": {
            "RAF-DB": RAFDB_LABELS,
            "AffectNet-7": RAFDB_LABELS,
            "AffectNet-8": AFFECTNET_8_LABELS,
            "CAER-S": RAFDB_LABELS,
        },
        "labels": RAFDB_LABELS,
        "checkpointPath": str(checkpoint_path),
        "weightsPresent": weights_present,
        "pretrainDir": str(get_pretrain_dir()),
        "pretrainPaths": {name: str(path) for name, path in pretrain_paths.items()},
        "pretrainPresent": pretrain_present,
        "missingPretrain": missing_pretrain,
        "upstreamPath": str(upstream_path) if upstream_path else None,
        "upstreamPresent": upstream_present,
        "runtimeReady": bool(weights_present and pretrain_present and upstream_present),
        "runtimeStatus": runtime_status,
        "source": "https://github.com/Talented-Q/POSTER_V2",
        "paper": "https://www.sciencedirect.com/science/article/pii/S0031320324007027",
        "installCommand": (
            f"{CHECKPOINT_ENV}=models/poster-plus-plus/checkpoints/rafdb_best.pth "
            f"{UPSTREAM_ENV}=/absolute/path/to/POSTER_V2 "
            f"{PRETRAIN_DIR_ENV}=models/poster-plus-plus/pretrain "
            "uv run --with torch --with timm --with thop --with numpy --with pillow "
            "uvicorn experiments.poster_plus_plus_lab.server:app --host 127.0.0.1 --port 8096"
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


def predict_with_poster_pp(image: Image.Image) -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    if not checkpoint_path.exists():
        raise PosterPPUnavailable(
            f"POSTER++ weights missing: place RAF-DB checkpoint at {checkpoint_path}"
        )

    pretrain_paths = get_pretrain_paths()
    missing = [str(path) for path in pretrain_paths.values() if not path.exists()]
    if missing:
        raise PosterPPUnavailable("POSTER++ pretrain missing: place " + ", ".join(missing))

    upstream_path = get_upstream_path()
    if not upstream_path or not upstream_path.exists():
        raise PosterPPUnavailable(
            f"POSTER++ upstream code missing: set {UPSTREAM_ENV} to a local clone of Talented-Q/POSTER_V2"
        )

    predictor = _load_predictor(checkpoint_path, upstream_path, pretrain_paths)
    return predictor(image)


def _load_predictor(
    checkpoint_path: Path, upstream_path: Path, pretrain_paths: dict[str, Path]
):
    cache_key = "|".join(
        [
            str(checkpoint_path.resolve()),
            str(upstream_path.resolve()),
            str(pretrain_paths["ir50"].resolve()),
            str(pretrain_paths["mobilefacenet"].resolve()),
        ]
    )
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached:
            return cached

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only in real runtime.
            raise PosterPPUnavailable(
                "POSTER++ runtime dependency missing: start with uv --with torch --with timm --with thop --with numpy --with pillow"
            ) from exc

        if str(upstream_path) not in sys.path:
            sys.path.insert(0, str(upstream_path))

        module = importlib.import_module("models.PosterV2_7cls")
        original_torch_load = torch.load

        def redirected_torch_load(path: object, *args: object, **kwargs: object) -> object:
            path_text = str(path)
            if "mobilefacenet_model_best.pth.tar" in path_text:
                path = pretrain_paths["mobilefacenet"]
            elif "ir50.pth" in path_text:
                path = pretrain_paths["ir50"]
            return _trusted_torch_load(original_torch_load, path, *args, **kwargs)

        try:
            torch.load = redirected_torch_load
            model = module.pyramid_trans_expr2(img_size=224, num_classes=7)
        finally:
            torch.load = original_torch_load

        _ensure_checkpoint_pickle_globals()
        checkpoint = _trusted_torch_load(torch.load, checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        def predictor(image: Image.Image) -> dict[str, object]:
            tensor = _image_to_tensor(image, torch)
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
            pairs = dict(zip(RAFDB_LABELS, probabilities, strict=True))
            raw_emotion = max(pairs.items(), key=lambda item: item[1])[0]
            return {
                "rawEmotion": raw_emotion,
                "confidence": float(pairs[raw_emotion]),
                "probabilities": pairs,
            }

        _MODEL_CACHE[cache_key] = predictor
        return predictor


def _strip_module_prefix(state_dict: Any) -> Any:
    if not hasattr(state_dict, "items"):
        return state_dict
    return {
        (key[7:] if isinstance(key, str) and key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _trusted_torch_load(torch_load: Any, path: object, *args: object, **kwargs: object) -> object:
    try:
        return torch_load(path, *args, weights_only=False, **kwargs)
    except TypeError:
        return torch_load(path, *args, **kwargs)


def _ensure_checkpoint_pickle_globals() -> None:
    main_module = sys.modules.get("__main__")
    if not main_module:
        return

    class RecorderMeter:
        pass

    class RecorderMeter1:
        pass

    for name, cls in {"RecorderMeter": RecorderMeter, "RecorderMeter1": RecorderMeter1}.items():
        if not hasattr(main_module, name):
            setattr(main_module, name, cls)


def _image_to_tensor(image: Image.Image, torch: Any) -> Any:
    import numpy as np

    resized = image.resize((224, 224))
    array = np.asarray(resized).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def build_preprocess_info(
    preprocess_mode: str | None,
    crop_box: str | None,
    source_width: int | None,
    source_height: int | None,
    output_size: int | None = None,
    fallback: str | None = None,
) -> dict[str, object] | None:
    if not any([preprocess_mode, crop_box, source_width, source_height, output_size, fallback]):
        return None
    return {
        "mode": str(preprocess_mode or "none"),
        "cropBox": crop_box or "",
        "sourceWidth": source_width,
        "sourceHeight": source_height,
        "outputSize": output_size,
        "fallback": _parse_bool(fallback),
    }


def normalize_prediction(
    prediction: dict[str, object],
    preprocess: dict[str, object] | None = None,
) -> dict[str, object]:
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
        "source": "POSTER++",
        "checkpointPath": str(get_checkpoint_path()),
        "pretrainDir": str(get_pretrain_dir()),
    }
    if preprocess:
        features["preprocess"] = preprocess
    evidence = (
        "POSTER++ RAF-DB 表情分类："
        f"rawEmotion={raw_emotion}, confidence={confidence:.2f}。"
        "该视觉结果是表情分类弱信号，不是心理诊断。"
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


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}
