# EmotiEffLib Lab

这是一个独立的 EmotiEffLib 本地实验模块，用来验证“摄像头抽帧 -> 本地面部表情模型 -> 兼容正式视觉服务字段”的链路。

## 边界

- 不接入正式 `/api/chat/multimodal/stream`。
- 不修改 `app/`。
- 不写数据库、Excel、Redis、Chroma。
- 不保存原始图片帧。
- 不触发心理报告或预警。
- 这是面部表情分类实验，不是医学诊断，也不是 Qwen 对视频语义的判断。

## 启动

本实验不把 EmotiEffLib 加入正式后端依赖，使用 `uv --with` 临时注入：

```bash
uv run --with emotiefflib==1.1.1 --with opencv-python-headless --with numpy \
  uvicorn experiments.emotiefflib_lab.server:app --host 127.0.0.1 --port 8092
```

打开：

```text
http://127.0.0.1:8092/
```

首次推理时，EmotiEffLib 可能会下载 ONNX 模型文件。

## 默认配置

- Engine：`onnx`
- Device：`cpu`
- Model：`enet_b2_8`
- 图片大小上限：8MB
- 支持输入：JPEG / PNG

## 接口

### `GET /health`

```json
{"status":"UP"}
```

### `GET /models`

返回当前可用 engine/model 列表、默认配置和启动命令。

### `POST /analyze-frame`

`multipart/form-data`：

- `file`：JPEG 或 PNG 图片帧。
- `engine`：默认 `onnx`。
- `model`：默认 `enet_b2_8`。
- `device`：默认 `cpu`。

返回字段兼容正式 Python 视觉服务语义：

```json
{
  "emotion": "NORMAL",
  "visualEmotion": "NORMAL",
  "score": 0.42,
  "visualScore": 0.42,
  "riskLevel": "LOW",
  "confidence": 0.53,
  "evidence": "EmotiEffLib 面部表情分类...",
  "features": {
    "rawEmotion": "Neutral",
    "probabilities": {}
  },
  "timestamp": "2026-06-09T00:00:00+00:00"
}
```
