# Realtime Face Mesh Lab

这是一个独立的浏览器端 MediaPipe FaceLandmarker 实验模块，用来验证“摄像头视频 -> Face Mesh landmarks/blendshapes -> 视觉情绪标签展示”的链路。

## 边界

- 不接入正式聊天链路。
- 不修改 `app/`。
- 不写数据库、Excel、Redis、Chroma。
- 不保存原始视频、图片帧、landmarks 或 blendshapes。
- Face Mesh 在浏览器本地运行；后端只提供页面和 JSON 标准化调试接口。
- 情绪标签仍是启发式映射，不是医学诊断，也不是 Qwen 对视频的判断。

## 启动

```bash
uv run uvicorn experiments.realtime_camera_lab.server:app --host 127.0.0.1 --port 8091
```

然后打开：

```text
http://127.0.0.1:8091/
```

首次打开摄像头时，页面会从 CDN 加载 MediaPipe Tasks Vision WASM 和官方 `face_landmarker.task` 模型。

## 接口

### `GET /health`

返回：

```json
{"status":"UP"}
```

### `POST /analyze-frame`

调试接口，接收前端 FaceLandmarker 计算出的 JSON，不接收原始图片帧：

```json
{
  "faceDetected": true,
  "landmarkCount": 478,
  "score": 2.35,
  "confidence": 0.81,
  "features": {
    "browTension": 0.18,
    "eyeTension": 0.22,
    "mouthDown": 0.14,
    "muscleTension": 0.31
  }
}
```

返回字段兼容正式 Python `MEDIAPIPE_MODE=http` 语义：

```json
{
  "emotion": "ANXIETY",
  "visualEmotion": "ANXIETY",
  "score": 2.35,
  "visualScore": 2.35,
  "riskLevel": "LOW",
  "confidence": 0.81,
  "evidence": "MediaPipe FaceLandmarker 浏览器端分析...",
  "features": {},
  "timestamp": "2026-06-09T00:00:00+00:00"
}
```

## 使用

1. 点击“开启摄像头”，允许浏览器摄像头权限。
2. 等待页面显示 FaceLandmarker 已就绪。
3. 点击“单帧检测”进行一次检测。
4. 勾选“自动分析”后，页面会持续检测并刷新情绪标签。
5. 点击“停止摄像头”释放摄像头。
