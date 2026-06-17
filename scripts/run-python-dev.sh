#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Cannot find uv. Install uv or add it to PATH."
  exit 1
fi

mkdir -p data
uv run uvicorn app.main:app --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8080}"
