# POSTER-Var SOTA Lab

这是一个独立的 POSTER-Var 面部表情识别实验模块，用来验证近年 SOTA 方法在 MindBridge 实时摄像头场景中的可行性。

## 边界

- 不接入正式 `/api/chat/multimodal/stream`。
- 不修改 `app/`。
- 不写数据库、Excel、Redis、Chroma。
- 不保存原始图片帧。
- 不触发心理报告或预警。
- 不回退到 DeepFace 或 EmotiEffLib；缺少权重时明确返回 `weights missing`。
- POSTER-Var 输出仍是表情分类弱信号，不是医学诊断。

## 准备权重

默认 checkpoint 路径：

```text
models/poster-var/checkpoints/rafdb_best.pth
```

上游 POSTER-Var README 指向 best models 下载入口：

```text
https://github.com/lg2578/poster-var
```

如果下载到的 checkpoint 文件名不同，可以用环境变量指定：

```bash
POSTER_VAR_CHECKPOINT=/absolute/path/to/checkpoint.pth
```

如果要接入真实上游模型构造，还需要把上游代码 clone 到本地，并通过：

```bash
POSTER_VAR_UPSTREAM=/absolute/path/to/poster-var
```

指定位置。

## 启动

```bash
POSTER_VAR_CHECKPOINT=models/poster-var/checkpoints/rafdb_best.pth \
uv run --with torch --with torchvision --with timm --with einops --with opencv-python-headless --with pyyaml --with numpy --with pillow \
  uvicorn experiments.poster_var_lab.server:app --host 127.0.0.1 --port 8094
```

打开：

```text
http://127.0.0.1:8094/
```

## 接口

### `GET /health`

```json
{"status":"UP"}
```

### `GET /models`

返回 POSTER-Var 元信息、RAF-DB 标签、checkpoint 路径和权重状态。

### `POST /analyze-frame`

`multipart/form-data`：

- `file`：JPEG 或 PNG 图片帧。

返回字段兼容正式 Python 视觉服务语义：

```json
{
  "emotion": "DEPRESSED",
  "visualEmotion": "DEPRESSED",
  "score": 3.72,
  "visualScore": 3.72,
  "riskLevel": "MEDIUM",
  "confidence": 0.9,
  "evidence": "POSTER-Var RAF-DB 表情分类...",
  "features": {
    "rawEmotion": "sad",
    "probabilities": {}
  },
  "timestamp": "2026-06-09T00:00:00+00:00"
}
```
