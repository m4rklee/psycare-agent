# UniFER-7B Local Assets

This directory is reserved for the isolated UniFER-7B experiment. Do not commit downloaded model files or GGUF weights.

## Model

- Model: `Karl28/UniFER-7B`
- Source: https://huggingface.co/Karl28/UniFER-7B
- Upstream code/paper repo: https://github.com/zfkarl/UniFER
- Size: about 16.6GB (BF16 Safetensors) or about 6GB (Q4_K_M GGUF + mmproj)
- Architecture: Qwen2.5-VL post-trained for facial expression reasoning

## Recommended path: GGUF + Ollama (fast on Mac)

The lab defaults to `UNIFER_BACKEND=ollama`. Convert the downloaded Safetensors checkpoint once, register it in Ollama, then start a lightweight FastAPI server.

### 1. Download Safetensors (one-time)

```bash
HF_HOME=models/unifer-7b/hf-cache \
uv run --with huggingface-hub \
  hf download Karl28/UniFER-7B
```

### 2. Convert to GGUF (one-time, memory-heavy)

Requires enough RAM/disk and `cmake` (`brew install cmake`). Clones `llama.cpp` into `.tools/llama.cpp` if missing.

```bash
./scripts/convert-unifer-to-gguf.sh
```

Outputs (gitignored):

```text
models/unifer-7b/unifer-7b-q4_k_m.gguf
models/unifer-7b/unifer-7b-mmproj-f16.gguf
```

Skip the smoke test if needed:

```bash
SKIP_SMOKE=1 ./scripts/convert-unifer-to-gguf.sh
```

### 3. Register in Ollama

Requires Ollama >= 0.7.0 and the Ollama app/daemon running.

```bash
./scripts/create-unifer-ollama-model.sh
```

This creates `unifer-7b:q4_k_m` from [`Modelfile`](Modelfile).

### 4. Start the lab

```bash
UNIFER_BACKEND=ollama \
uv run --with httpx --with pillow \
  uvicorn experiments.unifer_lab.server:app --host 127.0.0.1 --port 8097
```

Open:

```text
http://127.0.0.1:8097/
```

## Fallback path: transformers (slow on 24GB Mac)

If GGUF conversion fails or labels look wrong, switch back to the original Hugging Face runtime:

```bash
UNIFER_BACKEND=transformers \
HF_HOME=models/unifer-7b/hf-cache \
uv run --with torch --with torchvision --with transformers --with accelerate --with qwen-vl-utils --with pillow --with huggingface-hub \
  uvicorn experiments.unifer_lab.server:app --host 127.0.0.1 --port 8097
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFER_BACKEND` | `ollama` | `ollama` or `transformers` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP API |
| `OLLAMA_UNIFER_MODEL` | `unifer-7b:q4_k_m` | Registered Ollama model tag |
| `UNIFER_MAX_NEW_TOKENS` | `256` | Generation cap for both backends |
| `HF_HOME` | `models/unifer-7b/hf-cache` | Safetensors cache (transformers path) |
| `UNIFER_LOCAL_FILES_ONLY` | `true` | Avoid surprise HF downloads |
| `UNIFER_DEVICE_MAP` | `auto` | transformers device map |
| `UNIFER_TORCH_DTYPE` | `auto` | transformers dtype |

## Runtime notes

- Ollama keeps the model loaded and uses Metal on Apple Silicon; FastAPI stays responsive during inference.
- The transformers path may offload weights to disk on 24GB Macs and block the server on first load.
- Automatic analysis in the UI is set to 10 seconds.
- Model output is facial-expression reasoning, not a psychological or medical diagnosis.
- Q4_K_M may reduce FER accuracy; try `QUANT_TYPE=Q6_K` in the convert script if labels drift.

## Cleanup

```bash
rm -rf models/unifer-7b/hf-cache
rm -f models/unifer-7b/unifer-7b-*.gguf models/unifer-7b/smoke-test.txt
```
