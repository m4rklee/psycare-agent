#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_OLLAMA_BIN="$(command -v ollama || true)"
if [ -z "$DEFAULT_OLLAMA_BIN" ] && [ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
  DEFAULT_OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$DEFAULT_OLLAMA_BIN}"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Cannot find Ollama."
  echo "Install Ollama or set OLLAMA_BIN to the ollama executable path."
  exit 1
fi

"$OLLAMA_BIN" create multimodalAgent-qwen2.5-7b-ft:latest -f "$ROOT_DIR/models/multimodalAgent-qwen2.5-7b-ft/Modelfile"

echo "Created multimodalAgent-qwen2.5-7b-ft:latest"
echo "Run PsyCare Agent with: ./scripts/run-python-dev.sh"
