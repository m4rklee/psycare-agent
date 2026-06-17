#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/unifer-7b"
DEFAULT_OLLAMA_BIN="$(command -v ollama || true)"
if [ -z "$DEFAULT_OLLAMA_BIN" ] && [ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
  DEFAULT_OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$DEFAULT_OLLAMA_BIN}"
MODEL_TAG="${OLLAMA_UNIFER_MODEL:-unifer-7b:q4_k_m}"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Cannot find Ollama. Install Ollama or set OLLAMA_BIN." >&2
  exit 1
fi

if [ ! -f "$MODEL_DIR/unifer-7b-q4_k_m.gguf" ] || [ ! -f "$MODEL_DIR/unifer-7b-mmproj-f16.gguf" ]; then
  echo "GGUF files missing. Run ./scripts/convert-unifer-to-gguf.sh first." >&2
  exit 1
fi

"$OLLAMA_BIN" create "$MODEL_TAG" -f "$MODEL_DIR/Modelfile"

echo "Created $MODEL_TAG"
echo "Verify: $OLLAMA_BIN list | grep unifer"
echo "Start lab: UNIFER_BACKEND=ollama uv run --with httpx --with pillow uvicorn experiments.unifer_lab.server:app --host 127.0.0.1 --port 8097"
