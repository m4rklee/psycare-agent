# FED-PsyAU Lab

这是一个独立的 FED-PsyAU 微表情视频实验模块，用来验证 ICCV 2025 方法在摄像头短视频片段上的可行性。

## 边界

- 不接入正式 `/api/chat/multimodal/stream`。
- 不修改 `app/`。
- 不写数据库、Excel、Redis、Chroma。
- 不保存原始视频片段。
- 不触发心理报告或预警。
- 不回退到 DeepFace、EmotiEffLib 或 POSTER-Var；缺少权重或上游代码时明确返回 503。
- FED-PsyAU 输出是微表情识别弱信号，不是医学诊断。

## 准备资源

默认 checkpoint 路径：

```text
models/fed-psyau/checkpoints/dfme_best.pth
```

上游代码：

```text
https://github.com/MELABIPCAS/FED-PsyAU
```

通过环境变量指定：

```bash
FED_PSYAU_CHECKPOINT=models/fed-psyau/checkpoints/dfme_best.pth
FED_PSYAU_UPSTREAM=/absolute/path/to/FED-PsyAU
```

## 启动

```bash
FED_PSYAU_CHECKPOINT=models/fed-psyau/checkpoints/dfme_best.pth \
FED_PSYAU_UPSTREAM=/absolute/path/to/FED-PsyAU \
uv run --with torch --with torchvision --with opencv-python-headless --with numpy --with pillow \
  uvicorn experiments.fed_psyau_lab.server:app --host 127.0.0.1 --port 8095
```

打开：

```text
http://127.0.0.1:8095/
```

## 接口

### `GET /health`

```json
{"status":"UP"}
```

### `GET /models`

返回 FED-PsyAU、ICCV 2025、标签集、checkpoint/upstream 路径和运行状态。

### `POST /analyze-clip`

`multipart/form-data`：

- `file`：`video/webm` 或 `video/mp4` 短视频片段。

返回字段兼容正式 Python 视觉服务语义：

```json
{
  "emotion": "HIGH_RISK",
  "visualEmotion": "HIGH_RISK",
  "microExpression": "Fear",
  "score": 4.18,
  "visualScore": 4.18,
  "riskLevel": "HIGH",
  "confidence": 0.82,
  "evidence": "FED-PsyAU 微表情短视频分析...",
  "features": {
    "task": "micro_expression_recognition",
    "auPredictions": {
      "AU4": 0.71
    }
  },
  "timestamp": "2026-06-09T00:00:00+00:00"
}
```
