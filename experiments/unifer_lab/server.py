from __future__ import annotations

import base64
import os
import re
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
UNIFER_DIR = REPO_ROOT / "models" / "unifer-7b"

MODEL_ID = "Karl28/UniFER-7B"
MODEL_ENV = "UNIFER_MODEL_ID"
HF_HOME_ENV = "HF_HOME"
LOCAL_ONLY_ENV = "UNIFER_LOCAL_FILES_ONLY"
DEVICE_MAP_ENV = "UNIFER_DEVICE_MAP"
TORCH_DTYPE_ENV = "UNIFER_TORCH_DTYPE"
BACKEND_ENV = "UNIFER_BACKEND"
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV = "OLLAMA_UNIFER_MODEL"
MAX_NEW_TOKENS_ENV = "UNIFER_MAX_NEW_TOKENS"

DEFAULT_HF_HOME = REPO_ROOT / "models" / "unifer-7b" / "hf-cache"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "unifer-7b:q4_k_m"
GGUF_LLM_NAME = "unifer-7b-q4_k_m.gguf"
GGUF_MMPROJ_NAME = "unifer-7b-mmproj-f16.gguf"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
UNIFER_LABELS = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
PROMPT = (
    "As an expert in facial expression recognition, which expression is most prominent "
    "in this image? Please select your answer from the following candidate labels: "
    "surprise, fear, disgust, happiness, sadness, anger, neutral. Provide your detailed "
    "reasoning between the <think></think> tags, and then give your final answer between "
    "the <answer></answer> tags."
)

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, Any] = {}

app = FastAPI(
    title="UniFER-7B Lab",
    description="Isolated UniFER-7B multimodal facial-expression reasoning experiment.",
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
        prediction = predict_with_unifer(image)
    except UniFERUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - shields the lab UI from raw traces.
        raise HTTPException(status_code=503, detail=f"UniFER-7B 分析失败：{exc}") from exc

    preprocess = build_preprocess_info(
        preprocess_mode,
        crop_box,
        source_width,
        source_height,
        output_size,
        fallback,
    )
    return normalize_prediction(prediction, preprocess=preprocess)


class UniFERUnavailable(RuntimeError):
    pass


def get_model_id() -> str:
    return os.getenv(MODEL_ENV, MODEL_ID)


def get_hf_home() -> Path:
    return Path(os.getenv(HF_HOME_ENV, str(DEFAULT_HF_HOME))).expanduser()


def local_files_only() -> bool:
    return os.getenv(LOCAL_ONLY_ENV, "true").strip().lower() not in {"0", "false", "no", "off"}


def get_model_cache_dir() -> Path:
    model_id = get_model_id()
    return get_hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"


def get_snapshot_dirs() -> list[Path]:
    snapshots_dir = get_model_cache_dir() / "snapshots"
    if not snapshots_dir.exists():
        return []
    return sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)


def get_backend() -> str:
    backend = os.getenv(BACKEND_ENV, "ollama").strip().lower()
    if backend in {"ollama", "transformers"}:
        return backend
    return "ollama"


def get_ollama_base_url() -> str:
    return os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def get_ollama_model() -> str:
    return os.getenv(OLLAMA_MODEL_ENV, DEFAULT_OLLAMA_MODEL)


def get_max_new_tokens() -> int:
    raw = os.getenv(MAX_NEW_TOKENS_ENV, "256").strip()
    try:
        return max(32, int(raw))
    except ValueError:
        return 256


def get_gguf_paths() -> tuple[Path, Path]:
    return UNIFER_DIR / GGUF_LLM_NAME, UNIFER_DIR / GGUF_MMPROJ_NAME


def gguf_files_present() -> bool:
    llm_path, mmproj_path = get_gguf_paths()
    return llm_path.is_file() and mmproj_path.is_file()


def safetensors_files_present() -> bool:
    snapshot_dirs = get_snapshot_dirs()
    if not snapshot_dirs:
        return False
    snapshot = snapshot_dirs[0]
    return (snapshot / "config.json").exists() and (
        (snapshot / "model.safetensors.index.json").exists()
        or any(snapshot.glob("model-*.safetensors"))
    )


def _normalize_ollama_model_name(name: str) -> str:
    return name.removesuffix(":latest")


def ollama_model_registered(model_name: str | None = None) -> bool:
    target = _normalize_ollama_model_name(model_name or get_ollama_model())
    try:
        import httpx

        response = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=2.0)
        response.raise_for_status()
        for entry in response.json().get("models", []):
            candidate = _normalize_ollama_model_name(str(entry.get("name", "")))
            if candidate == target or candidate.startswith(f"{target}:"):
                return True
    except Exception:
        return False
    return False


def _ollama_runtime_status() -> tuple[bool, str]:
    if ollama_model_registered():
        return True, "ready"
    if gguf_files_present():
        return False, "ollama model missing, run ./scripts/create-unifer-ollama-model.sh"
    if safetensors_files_present():
        return False, "gguf missing, run ./scripts/convert-unifer-to-gguf.sh"
    return False, "model missing, download Karl28/UniFER-7B first"


def _transformers_runtime_status() -> tuple[bool, str]:
    if safetensors_files_present():
        return True, "ready"
    return False, "model missing"


def get_install_command() -> str:
    if get_backend() == "ollama":
        return (
            "./scripts/create-unifer-ollama-model.sh && "
            "UNIFER_BACKEND=ollama uv run --with httpx --with pillow "
            "uvicorn experiments.unifer_lab.server:app --host 127.0.0.1 --port 8097"
        )
    return (
        "UNIFER_BACKEND=transformers HF_HOME=models/unifer-7b/hf-cache "
        "uv run --with torch --with torchvision --with transformers --with accelerate "
        "--with qwen-vl-utils --with pillow --with huggingface-hub "
        "uvicorn experiments.unifer_lab.server:app --host 127.0.0.1 --port 8097"
    )


def get_model_metadata() -> dict[str, object]:
    backend = get_backend()
    snapshot_dirs = get_snapshot_dirs()
    files_present = safetensors_files_present()
    gguf_present = gguf_files_present()
    ollama_model = get_ollama_model()
    ollama_ready = ollama_model_registered()
    if backend == "ollama":
        runtime_ready, runtime_status = _ollama_runtime_status()
        model_size_hint = "about 6GB GGUF (Q4_K_M + mmproj)"
        device_map = "ollama"
        torch_dtype = "metal"
    else:
        runtime_ready, runtime_status = _transformers_runtime_status()
        model_size_hint = "about 16.6GB"
        device_map = os.getenv(DEVICE_MAP_ENV, "auto")
        torch_dtype = os.getenv(TORCH_DTYPE_ENV, "auto")
    return {
        "model": "UniFER-7B",
        "modelId": get_model_id(),
        "source": "https://github.com/zfkarl/UniFER",
        "modelPage": "https://huggingface.co/Karl28/UniFER-7B",
        "task": "multimodal_facial_expression_reasoning",
        "labels": UNIFER_LABELS,
        "modelSizeHint": model_size_hint,
        "backend": backend,
        "ollamaModel": ollama_model,
        "ollamaBaseUrl": get_ollama_base_url(),
        "ollamaReady": ollama_ready,
        "ggufPresent": gguf_present,
        "hfHome": str(get_hf_home()),
        "cacheDir": str(get_model_cache_dir()),
        "snapshotPresent": bool(snapshot_dirs),
        "filesPresent": files_present,
        "localFilesOnly": local_files_only(),
        "deviceMap": device_map,
        "torchDtype": torch_dtype,
        "maxNewTokens": get_max_new_tokens(),
        "runtimeReady": runtime_ready,
        "runtimeStatus": runtime_status,
        "installCommand": get_install_command(),
    }


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=400, detail="仅支持 JPEG 或 PNG 图片帧")
            return image.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="无法解析图片帧") from exc


def predict_with_unifer(image: Image.Image) -> dict[str, object]:
    if get_backend() == "ollama":
        return _predict_with_ollama(image)
    predictor = _load_predictor()
    return predictor(image)


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _predict_with_ollama(image: Image.Image) -> dict[str, object]:
    runtime_ready, runtime_status = _ollama_runtime_status()
    if not runtime_ready:
        raise UniFERUnavailable(f"UniFER-7B Ollama runtime not ready: {runtime_status}")

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - real runtime only.
        missing = getattr(exc, "name", None) or str(exc)
        raise UniFERUnavailable(
            f"UniFER-7B Ollama dependency missing: ({missing}). "
            "Start with httpx and pillow via uv --with httpx --with pillow"
        ) from exc

    payload = {
        "model": get_ollama_model(),
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [_image_to_base64(image)],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": get_max_new_tokens(),
        },
    }
    try:
        response = httpx.post(
            f"{get_ollama_base_url()}/api/chat",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - depends on local Ollama runtime.
        raise UniFERUnavailable(f"UniFER-7B Ollama inference failed: {exc}") from exc

    content = str(data.get("message", {}).get("content", ""))
    if not content:
        raise UniFERUnavailable("UniFER-7B Ollama returned an empty response")
    return {"modelResponse": content}


def _load_predictor():
    cache_key = "|".join(
        [
            get_model_id(),
            str(get_hf_home()),
            str(local_files_only()),
            os.getenv(DEVICE_MAP_ENV, "auto"),
            os.getenv(TORCH_DTYPE_ENV, "auto"),
        ]
    )
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached:
            return cached

        if local_files_only() and not safetensors_files_present():
            raise UniFERUnavailable(
                f"UniFER-7B model missing: download {get_model_id()} into {get_hf_home()}"
            )

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - real runtime only.
            missing = getattr(exc, "name", None) or str(exc)
            raise UniFERUnavailable(
                "UniFER-7B runtime dependency missing: "
                f"({missing}). Start with torch, torchvision, transformers, "
                "accelerate, qwen-vl-utils, pillow, huggingface-hub"
            ) from exc

        torch_dtype = _resolve_torch_dtype(torch)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                get_model_id(),
                cache_dir=str(get_hf_home() / "hub"),
                local_files_only=local_files_only(),
                torch_dtype=torch_dtype,
                device_map=os.getenv(DEVICE_MAP_ENV, "auto"),
            )
            processor = AutoProcessor.from_pretrained(
                get_model_id(),
                cache_dir=str(get_hf_home() / "hub"),
                local_files_only=local_files_only(),
            )
        except Exception as exc:  # pragma: no cover - depends on local model/runtime.
            raise UniFERUnavailable(f"UniFER-7B model load failed: {exc}") from exc

        def predictor(input_image: Image.Image) -> dict[str, object]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": input_image,
                            "resized_height": 224,
                            "resized_width": 224,
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=get_max_new_tokens(),
                    use_cache=True,
                )
            generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1] :]
            response = processor.decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return {"modelResponse": response}

        _MODEL_CACHE[cache_key] = predictor
        return predictor


def _resolve_torch_dtype(torch: Any) -> Any:
    requested = os.getenv(TORCH_DTYPE_ENV, "auto").strip().lower()
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


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
    response = str(prediction.get("modelResponse") or prediction.get("response") or "")
    raw_label, parse_status = extract_label(response)
    visual_emotion, visual_score = _map_emotion(raw_label)
    risk_level = _risk_from_score(visual_score)
    think = extract_tag(response, "think")
    answer = extract_tag(response, "answer")
    features: dict[str, object] = {
        "rawEmotion": raw_label,
        "answer": answer,
        "think": think,
        "parseStatus": parse_status,
        "modelResponse": response,
        "labels": UNIFER_LABELS,
        "source": "UniFER-7B",
        "modelId": get_model_id(),
    }
    if preprocess:
        features["preprocess"] = preprocess
    evidence = (
        "UniFER-7B 表情推理："
        f"rawEmotion={raw_label}, parseStatus={parse_status}。"
        "该视觉结果是表情识别弱信号，不是心理诊断。"
    )
    return {
        "emotion": visual_emotion,
        "visualEmotion": visual_emotion,
        "score": round(visual_score, 2),
        "visualScore": round(visual_score, 2),
        "riskLevel": risk_level,
        "confidence": 1.0 if parse_status == "answer_tag" else 0.6,
        "evidence": evidence,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_label(response: str) -> tuple[str, str]:
    answer = extract_tag(response, "answer").lower()
    if answer:
        label = _label_from_text(answer)
        if label != "unknown":
            return label, "answer_tag"
    label = _label_from_text(response.lower())
    if label != "unknown":
        return label, "text_match"
    return "neutral", "fallback_neutral"


def _label_from_text(text: str) -> str:
    normalized = text.lower()
    direct_labels = [
        "happiness",
        "sadness",
        "surprise",
        "fear",
        "disgust",
        "anger",
        "neutral",
    ]
    for label in direct_labels:
        if re.search(rf"\b{label}\b", normalized):
            return label
    aliases = {
        "happiness": ["happy", "joy", "smile", "smiling"],
        "sadness": ["sad", "sorrow", "unhappy"],
        "anger": ["angry", "mad"],
        "disgust": ["revulsion", "repulsion"],
        "fear": ["terror", "fright", "afraid"],
        "neutral": ["normal", "calm", "blank"],
        "surprise": ["shock", "astonishment", "surprised"],
    }
    for label, alternatives in aliases.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in alternatives):
            return label
    return "unknown"


def _map_emotion(raw_label: str) -> tuple[str, float]:
    if raw_label == "sadness":
        return "DEPRESSED", 3.7
    if raw_label in {"fear", "anger", "disgust"}:
        return "HIGH_RISK", 4.2
    return "NORMAL", 0.8


def _risk_from_score(score: float) -> str:
    if score >= 4.0:
        return "HIGH"
    if score >= 3.0:
        return "MEDIUM"
    return "LOW"


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}
