#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/unifer-7b}"
HF_HOME="${HF_HOME:-$ROOT_DIR/models/unifer-7b/hf-cache}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/.tools/llama.cpp}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

LLM_F16="$OUT_DIR/unifer-7b-f16.gguf"
MMPROJ_F16="$OUT_DIR/unifer-7b-mmproj-f16.gguf"
LLM_QUANT="$OUT_DIR/unifer-7b-q4_k_m.gguf"

resolve_snapshot_dir() {
  local cache_dir="$HF_HOME/hub/models--Karl28--UniFER-7B/snapshots"
  if [ ! -d "$cache_dir" ]; then
    echo "UniFER snapshot missing under $cache_dir" >&2
    echo "Download first:" >&2
    echo "  HF_HOME=$HF_HOME uv run --with huggingface-hub hf download Karl28/UniFER-7B" >&2
    exit 1
  fi
  local snapshot
  snapshot="$(find "$cache_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$snapshot" ] || [ ! -f "$snapshot/config.json" ]; then
    echo "No valid UniFER snapshot found in $cache_dir" >&2
    exit 1
  fi
  printf '%s' "$snapshot"
}

gguf_looks_complete() {
  local path="$1"
  local min_bytes="$2"
  [ -f "$path" ] && [ "$(wc -c <"$path")" -ge "$min_bytes" ]
}

ensure_llama_cpp() {
  if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
    echo "Cloning llama.cpp into $LLAMA_CPP_DIR ..."
    mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
  fi
}

find_llama_quantize() {
  if [ -n "${LLAMA_QUANTIZE_BIN:-}" ] && [ -x "$LLAMA_QUANTIZE_BIN" ]; then
    printf '%s' "$LLAMA_QUANTIZE_BIN"
    return 0
  fi
  if [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    printf '%s' "$LLAMA_CPP_DIR/build/bin/llama-quantize"
    return 0
  fi
  if command -v llama-quantize >/dev/null 2>&1; then
    command -v llama-quantize
    return 0
  fi
  return 1
}

ensure_llama_quantize() {
  if find_llama_quantize >/dev/null; then
    find_llama_quantize
    return
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build llama-quantize." >&2
    echo "Install with: brew install cmake" >&2
    echo "Or set LLAMA_QUANTIZE_BIN to an existing llama-quantize executable." >&2
    exit 1
  fi
  echo "Building llama-quantize ..." >&2
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release >&2
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize llama-mtmd-cli -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >&2
  find_llama_quantize
}

find_llama_mtmd_cli() {
  if [ -n "${LLAMA_MTMD_CLI_BIN:-}" ] && [ -x "$LLAMA_MTMD_CLI_BIN" ]; then
    printf '%s' "$LLAMA_MTMD_CLI_BIN"
    return 0
  fi
  if [ -x "$LLAMA_CPP_DIR/build/bin/llama-mtmd-cli" ]; then
    printf '%s' "$LLAMA_CPP_DIR/build/bin/llama-mtmd-cli"
    return 0
  fi
  if command -v llama-mtmd-cli >/dev/null 2>&1; then
    command -v llama-mtmd-cli
    return 0
  fi
  return 1
}

ensure_llama_mtmd_cli() {
  if find_llama_mtmd_cli >/dev/null; then
    find_llama_mtmd_cli
    return
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build llama-mtmd-cli." >&2
    exit 1
  fi
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-mtmd-cli -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >&2
  find_llama_mtmd_cli
}

mkdir -p "$OUT_DIR"
SNAPSHOT_DIR="$(resolve_snapshot_dir)"
ensure_llama_cpp

CONVERT_PY="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT_PY" ]; then
  echo "convert_hf_to_gguf.py not found in $LLAMA_CPP_DIR" >&2
  exit 1
fi

echo "Using snapshot: $SNAPSHOT_DIR"
echo "Output dir: $OUT_DIR"

if ! gguf_looks_complete "$LLM_F16" 1000000000; then
  rm -f "$LLM_F16"
  echo "Converting UniFER LLM weights to F16 GGUF ..."
  uv run --with torch --with transformers --with sentencepiece --with protobuf \
    python "$CONVERT_PY" "$SNAPSHOT_DIR" --outfile "$LLM_F16" --outtype f16
else
  echo "Reusing existing $LLM_F16"
fi

if ! gguf_looks_complete "$MMPROJ_F16" 500000000; then
  rm -f "$MMPROJ_F16"
  echo "Converting UniFER vision projector (mmproj) ..."
  uv run --with torch --with transformers --with sentencepiece --with protobuf \
    python "$CONVERT_PY" "$SNAPSHOT_DIR" --mmproj --outfile "$MMPROJ_F16" --outtype f16
else
  echo "Reusing existing $MMPROJ_F16"
fi

QUANTIZE_BIN="$(ensure_llama_quantize)"
if [ ! -f "$LLM_QUANT" ]; then
  echo "Quantizing LLM to $QUANT_TYPE ..."
  "$QUANTIZE_BIN" "$LLM_F16" "$LLM_QUANT" "$QUANT_TYPE"
else
  echo "Reusing existing $LLM_QUANT"
fi

if [ "$SKIP_SMOKE" != "1" ]; then
  MTMD_CLI="$(ensure_llama_mtmd_cli)"
  SMOKE_IMAGE="${SMOKE_IMAGE:-}"
  if [ -z "$SMOKE_IMAGE" ]; then
    SMOKE_IMAGE="$(mktemp /tmp/unifer-smoke-XXXXXX.jpg)"
  uv run python - <<'PY' "$SMOKE_IMAGE"
from PIL import Image
import sys
Image.new("RGB", (224, 224), (180, 160, 140)).save(sys.argv[1], format="JPEG")
PY
  fi
  PROMPT='As an expert in facial expression recognition, which expression is most prominent in this image? Please select your answer from the following candidate labels: surprise, fear, disgust, happiness, sadness, anger, neutral. Provide your detailed reasoning between the <think></think> tags, and then give your final answer between the <answer></answer> tags.'
  echo "Running smoke test with llama-mtmd-cli ..."
  "$MTMD_CLI" -m "$LLM_QUANT" --mmproj "$MMPROJ_F16" -n 128 --temp 0 \
    -p "$PROMPT" --image "$SMOKE_IMAGE" | tee "$OUT_DIR/smoke-test.txt"
  if ! grep -Eiq '<(answer>|happiness|sadness|neutral|anger|fear|disgust|surprise)' "$OUT_DIR/smoke-test.txt"; then
    echo "Smoke test did not produce a recognizable FER label. Review $OUT_DIR/smoke-test.txt" >&2
    exit 1
  fi
  echo "Smoke test passed."
fi

echo
echo "GGUF conversion complete:"
echo "  $LLM_QUANT"
echo "  $MMPROJ_F16"
echo
echo "Next: register with Ollama"
echo "  ./scripts/create-unifer-ollama-model.sh"
