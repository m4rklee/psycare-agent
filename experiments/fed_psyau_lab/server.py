from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = BASE_DIR.parents[1]

MAX_CLIP_BYTES = 20 * 1024 * 1024
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "fed-psyau" / "checkpoints" / "dfme_best.pth"
CHECKPOINT_ENV = "FED_PSYAU_CHECKPOINT"
UPSTREAM_ENV = "FED_PSYAU_UPSTREAM"
DATASET_ENV = "FED_PSYAU_DATASET"
SUPPORTED_CONTENT_TYPES = {"video/webm", "video/mp4"}
CASME3_LABELS = ["positive", "negative", "surprise"]
DFME_LABELS = ["Happiness", "Surprise", "Disgust", "Sadness", "Anger", "Fear", "Contempt"]

app = FastAPI(
    title="FED-PsyAU Micro-Expression Lab",
    description="Isolated ICCV 2025 FED-PsyAU short-video experiment for multimodalAgent.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class FedPsyauUnavailable(RuntimeError):
    pass


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/models")
async def models() -> dict[str, object]:
    return get_model_metadata()


@app.post("/analyze-clip")
async def analyze_clip(file: UploadFile = File(...)) -> dict[str, object]:
    content_type = (file.content_type or "").lower().split(";")[0]
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 video/webm 或 video/mp4 短视频片段")

    clip_bytes = await file.read(MAX_CLIP_BYTES + 1)
    if not clip_bytes:
        raise HTTPException(status_code=400, detail="视频片段不能为空")
    if len(clip_bytes) > MAX_CLIP_BYTES:
        raise HTTPException(status_code=400, detail="视频片段不能超过 20MB")

    try:
        prediction = predict_with_fed_psyau(clip_bytes, content_type=content_type)
    except FedPsyauUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - protects the experiment UI from raw traces.
        raise HTTPException(status_code=503, detail=f"FED-PsyAU 分析失败：{exc}") from exc

    return normalize_prediction(prediction)


def get_checkpoint_path() -> Path:
    return Path(os.getenv(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT))).expanduser()


def get_upstream_path() -> Path | None:
    value = os.getenv(UPSTREAM_ENV)
    if not value:
        return None
    return Path(value).expanduser()


def get_dataset_name() -> str:
    value = os.getenv(DATASET_ENV, "DFME").strip().upper()
    if value in {"CASME3", "CAS(ME)3", "CAS(ME)³"}:
        return "CASME3"
    return "DFME"


def get_labels(dataset: str | None = None) -> list[str]:
    dataset_name = dataset or get_dataset_name()
    return CASME3_LABELS if dataset_name == "CASME3" else DFME_LABELS


def get_model_metadata() -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    upstream_path = get_upstream_path()
    weights_present = checkpoint_path.exists()
    upstream_present = bool(upstream_path and upstream_path.exists())
    dataset = get_dataset_name()
    labels = get_labels(dataset)
    runtime_ready = weights_present and upstream_present
    if not weights_present:
        status = "weights missing"
    elif not upstream_present:
        status = "upstream missing"
    else:
        status = "adapter pending"
    return {
        "model": "FED-PsyAU",
        "venue": "ICCV 2025",
        "task": "micro_expression_recognition",
        "inputType": "short_video_clip",
        "dataset": dataset,
        "labels": labels,
        "checkpointPath": str(checkpoint_path),
        "weightsPresent": weights_present,
        "upstreamPath": str(upstream_path) if upstream_path else None,
        "upstreamPresent": upstream_present,
        "runtimeReady": runtime_ready,
        "runtimeStatus": status,
        "source": "https://github.com/MELABIPCAS/FED-PsyAU",
        "paper": (
            "https://openaccess.thecvf.com/content/ICCV2025/html/"
            "Li_FED-PsyAU_Privacy-Preserving_Micro-Expression_Recognition_via_Psychological_AU_Coordination_and_Dynamic_ICCV_2025_paper.html"
        ),
        "installCommand": (
            f"{CHECKPOINT_ENV}=models/fed-psyau/checkpoints/dfme_best.pth "
            f"{UPSTREAM_ENV}=/absolute/path/to/FED-PsyAU "
            "uv run --with torch --with torchvision --with opencv-python-headless --with numpy --with pillow "
            "uvicorn experiments.fed_psyau_lab.server:app --host 127.0.0.1 --port 8095"
        ),
    }


def predict_with_fed_psyau(clip_bytes: bytes, *, content_type: str) -> dict[str, object]:
    checkpoint_path = get_checkpoint_path()
    if not checkpoint_path.exists():
        raise FedPsyauUnavailable(
            f"FED-PsyAU weights missing: place compatible checkpoint at {checkpoint_path}"
        )

    upstream_path = get_upstream_path()
    if not upstream_path or not upstream_path.exists():
        raise FedPsyauUnavailable(
            f"FED-PsyAU upstream code missing: set {UPSTREAM_ENV} to a local clone of MELABIPCAS/FED-PsyAU"
        )

    raise FedPsyauUnavailable(
        "FED-PsyAU runtime adapter is not configured yet. "
        "Real inference requires upstream preprocessing: TV-L1 optical flow, onset/apex handling, ROI/AU features."
    )


def normalize_prediction(prediction: dict[str, object]) -> dict[str, object]:
    dataset = str(prediction.get("dataset") or get_dataset_name()).upper()
    if dataset in {"CAS(ME)3", "CAS(ME)³"}:
        dataset = "CASME3"
    if dataset not in {"DFME", "CASME3"}:
        dataset = get_dataset_name()
    labels = get_labels(dataset)
    raw_label = str(
        prediction.get("microExpression")
        or prediction.get("rawEmotion")
        or prediction.get("label")
        or labels[0]
    ).strip()
    probabilities = _normalized_probabilities(prediction.get("probabilities") or {}, labels)
    if raw_label not in labels:
        raw_label = _dominant_from_probabilities(probabilities, labels)
    confidence = _clamp(float(prediction.get("confidence") or probabilities.get(raw_label, 0.0)), 0.0, 1.0)
    visual_emotion, visual_score = _map_micro_expression(raw_label, confidence)
    risk_level = _risk_from_score(visual_score)
    au_predictions = prediction.get("auPredictions") or {}
    if not isinstance(au_predictions, dict):
        au_predictions = {}
    features = {
        "task": "micro_expression_recognition",
        "dataset": dataset,
        "microExpression": raw_label,
        "probabilities": probabilities,
        "auPredictions": _normalized_au_predictions(au_predictions),
        "labels": labels,
        "source": "FED-PsyAU",
        "checkpointPath": str(get_checkpoint_path()),
    }
    evidence = (
        "FED-PsyAU 微表情短视频分析："
        f"microExpression={raw_label}, confidence={confidence:.2f}。"
        "该视觉结果是微表情弱信号，不是心理诊断。"
    )
    return {
        "emotion": visual_emotion,
        "visualEmotion": visual_emotion,
        "microExpression": raw_label,
        "score": round(visual_score, 2),
        "visualScore": round(visual_score, 2),
        "riskLevel": risk_level,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _normalized_probabilities(probabilities: dict[str, Any], labels: list[str]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    lower_lookup = {str(key).lower(): value for key, value in probabilities.items()}
    for label in labels:
        value = probabilities.get(label, lower_lookup.get(label.lower(), 0.0))
        try:
            normalized[label] = round(_clamp(float(value), 0.0, 1.0), 4)
        except (TypeError, ValueError):
            normalized[label] = 0.0
    return normalized


def _normalized_au_predictions(values: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = round(_clamp(float(value), 0.0, 1.0), 4)
        except (TypeError, ValueError):
            continue
    return normalized


def _dominant_from_probabilities(probabilities: dict[str, float], labels: list[str]) -> str:
    if not probabilities:
        return labels[0]
    return max(probabilities.items(), key=lambda item: item[1])[0]


def _map_micro_expression(raw_label: str, confidence: float) -> tuple[str, float]:
    normalized = raw_label.strip().lower()
    if normalized in {"happiness", "positive", "surprise"}:
        return "NORMAL", confidence * 0.8
    if normalized == "sadness":
        return "DEPRESSED", 3.0 + confidence * 0.8
    if normalized in {"fear", "anger", "disgust"}:
        return "HIGH_RISK", 3.2 + confidence * 1.2
    if normalized in {"contempt", "negative"}:
        return "ANXIETY", 2.0 + confidence * 0.8
    return "ANXIETY", 2.0 + confidence * 0.5


def _risk_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
