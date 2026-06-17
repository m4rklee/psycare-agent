# POSTER++ Local Assets

This directory is reserved for local POSTER++ / POSTER V2 assets. Do not commit downloaded model files.

Expected files:

- `checkpoints/rafdb_best.pth`
- `pretrain/ir50.pth`
- `pretrain/mobilefacenet_model_best.pth.tar`

Official upstream:

- Code: https://github.com/Talented-Q/POSTER_V2
- Paper: https://arxiv.org/abs/2301.12149
- RAF-DB checkpoint: https://drive.google.com/file/d/1aVm_hmJyZ5E_0p25XTbm3X9ophsKqCxv/view
- IR50 pretrain: https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view
- MobileFaceNet pretrain: https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view

Example startup:

```bash
POSTER_PP_CHECKPOINT=models/poster-plus-plus/checkpoints/rafdb_best.pth \
POSTER_PP_UPSTREAM=/absolute/path/to/POSTER_V2 \
POSTER_PP_PRETRAIN_DIR=models/poster-plus-plus/pretrain \
uv run --with torch --with timm --with thop --with numpy --with pillow \
  uvicorn experiments.poster_plus_plus_lab.server:app --host 127.0.0.1 --port 8096
```
