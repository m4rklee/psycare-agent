const TASKS_VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#canvas");
const faceCanvas = document.querySelector("#faceCanvas");
const overlayCanvas = document.querySelector("#overlayCanvas");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const captureBtn = document.querySelector("#captureBtn");
const autoToggle = document.querySelector("#autoToggle");
const serviceStatus = document.querySelector("#serviceStatus");
const fields = {
  emotion: document.querySelector("#emotion"),
  rawEmotion: document.querySelector("#rawEmotion"),
  score: document.querySelector("#score"),
  riskLevel: document.querySelector("#riskLevel"),
  confidence: document.querySelector("#confidence"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  preprocessStatus: document.querySelector("#preprocessStatus"),
  parseStatus: document.querySelector("#parseStatus"),
  modelState: document.querySelector("#modelState"),
  cacheState: document.querySelector("#cacheState"),
  deviceState: document.querySelector("#deviceState"),
  thinkText: document.querySelector("#thinkText"),
  rawResponse: document.querySelector("#rawResponse"),
  evidence: document.querySelector("#evidence"),
};

let stream = null;
let timer = null;
let busy = false;
let faceLandmarker = null;
let visionModule = null;
let previewFrame = null;
let lastPreviewAt = 0;
let latestFaceCrop = null;

const PREVIEW_INTERVAL_MS = 125;
const AUTO_ANALYZE_MS = 10000;
const ALIGN_OUTPUT_SIZE = 224;
const TARGET_EYE_Y = ALIGN_OUTPUT_SIZE * 0.38;
const TARGET_MOUTH_Y = ALIGN_OUTPUT_SIZE * 0.70;
const TARGET_EYE_DISTANCE = ALIGN_OUTPUT_SIZE * 0.36;
const TARGET_EYE_MOUTH_DISTANCE = TARGET_MOUTH_Y - TARGET_EYE_Y;
const MIN_EYE_DISTANCE = 18;
const LEFT_EYE_LANDMARKS = [33, 133, 159, 145];
const RIGHT_EYE_LANDMARKS = [362, 263, 386, 374];
const MOUTH_LANDMARKS = [61, 291, 13, 14];

async function checkService() {
  try {
    const [health, models] = await Promise.all([fetch("/health"), fetch("/models")]);
    if (!health.ok || !models.ok) throw new Error("service unavailable");
    const modelData = await models.json();
    serviceStatus.textContent = "服务就绪";
    renderModelState(modelData);
  } catch (error) {
    serviceStatus.textContent = "服务不可用";
    fields.evidence.textContent = `无法连接实验服务：${readableError(error)}`;
  }
}

function renderModelState(modelData) {
  const backend = modelData.backend || "ollama";
  fields.runtimeStatus.textContent = modelData.runtimeStatus || "-";
  fields.modelState.textContent = modelData.runtimeReady
    ? `${modelData.modelId} ready (${backend})`
    : `${modelData.modelId} missing (${backend})`;
  if (backend === "ollama") {
    fields.cacheState.textContent = modelData.ggufPresent
      ? "gguf ready"
      : `gguf missing / ${modelData.ollamaModel || "-"}`;
    fields.deviceState.textContent = `ollama / ${modelData.ollamaModel || "metal"}`;
  } else {
    fields.cacheState.textContent = modelData.hfHome || "-";
    fields.deviceState.textContent = `${modelData.deviceMap || "auto"} / ${modelData.torchDtype || "auto"}`;
  }

  if (!modelData.runtimeReady) {
    const hint =
      backend === "ollama"
        ? "请按 models/unifer-7b/README.md 转换 GGUF 并执行 ./scripts/create-unifer-ollama-model.sh。"
        : "请按 models/unifer-7b/README.md 下载 Safetensors 后重启 8097。";
    fields.evidence.textContent = `UniFER-7B 模型未就绪：${modelData.runtimeStatus}。${hint}`;
    if (modelData.installCommand) {
      fields.evidence.textContent += ` 启动命令：${modelData.installCommand}`;
    }
  }
}

async function ensureFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;
  fields.preprocessStatus.textContent = "loading";
  fields.evidence.textContent = "正在加载 MediaPipe FaceLandmarker...";
  if (!visionModule) {
    visionModule = await import(TASKS_VISION_URL);
  }
  const { FaceLandmarker, FilesetResolver } = visionModule;
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  fields.preprocessStatus.textContent = "ready";
  return faceLandmarker;
}

async function startCamera() {
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    fields.evidence.textContent = "摄像头已开启，正在准备人脸预处理。";
    await ensureFaceLandmarker();
    fields.evidence.textContent = "摄像头已开启，FaceLandmarker 已就绪。";
    startPreviewLoop();
  } catch (error) {
    fields.preprocessStatus.textContent = "failed";
    fields.evidence.textContent = `无法开启摄像头或加载 FaceLandmarker：${readableError(error)}`;
  }
}

function stopCamera() {
  stopPreviewLoop();
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  autoToggle.checked = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
  latestFaceCrop = null;
  clearCanvas(faceCanvas);
  clearCanvas(overlayCanvas);
}

async function analyzeFrame() {
  if (!stream || busy || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  busy = true;
  fields.evidence.textContent = "UniFER-7B 正在分析单帧，首次加载可能较慢...";
  try {
    const faceCrop = latestFaceCrop || captureFaceInput();
    if (!faceCrop.ok) {
      renderNoFace(faceCrop.message);
      return;
    }

    const form = new FormData();
    drawFaceInput(faceCrop);
    drawCropPreview();
    drawOverlay(faceCrop.cropBox, faceCrop.sourceWidth, faceCrop.sourceHeight);
    form.append("file", await cropToBlob(), "face-crop.jpg");
    form.append("preprocessMode", faceCrop.preprocessMode);
    form.append("cropBox", JSON.stringify(faceCrop.cropBox));
    form.append("sourceWidth", String(faceCrop.sourceWidth));
    form.append("sourceHeight", String(faceCrop.sourceHeight));
    form.append("outputSize", String(faceCrop.outputSize));
    form.append("fallback", String(faceCrop.fallback));

    const response = await fetch("/analyze-frame", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "分析失败");
    renderResult(payload);
  } catch (error) {
    fields.preprocessStatus.textContent = "failed";
    fields.evidence.textContent = `分析失败：${readableError(error)}`;
  } finally {
    busy = false;
  }
}

function startPreviewLoop() {
  stopPreviewLoop();
  lastPreviewAt = 0;
  const tick = (timestamp) => {
    previewFrame = requestAnimationFrame(tick);
    if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (timestamp - lastPreviewAt < PREVIEW_INTERVAL_MS) return;
    lastPreviewAt = timestamp;
    try {
      const faceCrop = captureFaceInput();
      if (!faceCrop.ok) {
        latestFaceCrop = null;
        clearCanvas(faceCanvas);
        clearCanvas(overlayCanvas);
        fields.preprocessStatus.textContent = "no face";
        return;
      }
      latestFaceCrop = faceCrop;
      fields.preprocessStatus.textContent = faceCrop.fallback ? "crop fallback" : "live aligned";
    } catch (error) {
      latestFaceCrop = null;
      fields.preprocessStatus.textContent = "failed";
    }
  };
  previewFrame = requestAnimationFrame(tick);
}

function stopPreviewLoop() {
  if (previewFrame !== null) {
    cancelAnimationFrame(previewFrame);
    previewFrame = null;
  }
}

function captureFaceInput() {
  const landmarker = faceLandmarker;
  if (!landmarker) {
    throw new Error("FaceLandmarker 尚未就绪");
  }
  const sourceWidth = video.videoWidth || 640;
  const sourceHeight = video.videoHeight || 480;
  const detection = landmarker.detectForVideo(video, performance.now());
  const landmarks = detection.faceLandmarks?.[0] || [];
  if (!landmarks.length) {
    return {
      ok: false,
      message: "未检测到人脸，请正对摄像头或改善光照。",
    };
  }

  const cropBox = landmarksToCropBox(landmarks, sourceWidth, sourceHeight);
  const faceInput = buildAlignedFaceInput(landmarks, cropBox, sourceWidth, sourceHeight);
  drawFaceInput(faceInput);
  drawCropPreview();
  drawOverlay(cropBox, sourceWidth, sourceHeight);
  fields.preprocessStatus.textContent = faceInput.fallback ? "crop fallback" : "aligned";
  return faceInput;
}

function buildAlignedFaceInput(landmarks, cropBox, sourceWidth, sourceHeight) {
  try {
    const leftEye = averageLandmarks(landmarks, LEFT_EYE_LANDMARKS, sourceWidth, sourceHeight);
    const rightEye = averageLandmarks(landmarks, RIGHT_EYE_LANDMARKS, sourceWidth, sourceHeight);
    const mouthCenter = averageLandmarks(landmarks, MOUTH_LANDMARKS, sourceWidth, sourceHeight);
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const eyeDistance = Math.hypot(dx, dy);
    const eyeMid = {
      x: (leftEye.x + rightEye.x) / 2,
      y: (leftEye.y + rightEye.y) / 2,
    };
    const eyeToMouth = Math.hypot(mouthCenter.x - eyeMid.x, mouthCenter.y - eyeMid.y);
    if (
      !Number.isFinite(eyeDistance) ||
      !Number.isFinite(eyeToMouth) ||
      eyeDistance < MIN_EYE_DISTANCE ||
      eyeToMouth < MIN_EYE_DISTANCE
    ) {
      return buildCropFallback(cropBox, sourceWidth, sourceHeight);
    }
    return {
      ok: true,
      cropBox,
      sourceWidth,
      sourceHeight,
      outputSize: ALIGN_OUTPUT_SIZE,
      preprocessMode: "mediapipe_affine_align",
      fallback: false,
      transform: {
        angle: Math.atan2(dy, dx),
        scale: averageNumbers([
          TARGET_EYE_DISTANCE / eyeDistance,
          TARGET_EYE_MOUTH_DISTANCE / eyeToMouth,
        ]),
        eyeMid,
        targetEyeMid: {
          x: ALIGN_OUTPUT_SIZE / 2,
          y: TARGET_EYE_Y,
        },
      },
    };
  } catch (error) {
    return buildCropFallback(cropBox, sourceWidth, sourceHeight);
  }
}

function buildCropFallback(cropBox, sourceWidth, sourceHeight) {
  return {
    ok: true,
    cropBox,
    sourceWidth,
    sourceHeight,
    outputSize: Math.max(cropBox.width, cropBox.height),
    preprocessMode: "mediapipe_face_crop",
    fallback: true,
  };
}

function drawFaceInput(faceInput) {
  if (!faceInput.fallback && faceInput.transform) {
    drawAlignedFace(faceInput.transform);
    return;
  }
  drawFaceCrop(faceInput.cropBox);
}

function drawAlignedFace(transform) {
  canvas.width = ALIGN_OUTPUT_SIZE;
  canvas.height = ALIGN_OUTPUT_SIZE;
  const context = canvas.getContext("2d");
  context.save();
  context.fillStyle = "#808080";
  context.fillRect(0, 0, ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE);
  const cos = Math.cos(-transform.angle);
  const sin = Math.sin(-transform.angle);
  const scale = transform.scale;
  const a = scale * cos;
  const b = scale * sin;
  const c = -scale * sin;
  const d = scale * cos;
  const e = transform.targetEyeMid.x - (a * transform.eyeMid.x + c * transform.eyeMid.y);
  const f = transform.targetEyeMid.y - (b * transform.eyeMid.x + d * transform.eyeMid.y);
  context.setTransform(a, b, c, d, e, f);
  context.drawImage(video, 0, 0);
  context.restore();
}

function drawFaceCrop(cropBox) {
  canvas.width = cropBox.width;
  canvas.height = cropBox.height;
  const context = canvas.getContext("2d");
  context.drawImage(
    video,
    cropBox.x,
    cropBox.y,
    cropBox.width,
    cropBox.height,
    0,
    0,
    cropBox.width,
    cropBox.height,
  );
}

function averageLandmarks(landmarks, indices, sourceWidth, sourceHeight) {
  const points = indices.map((index) => landmarks[index]).filter(Boolean);
  if (!points.length) {
    throw new Error("缺少人脸关键点");
  }
  const total = points.reduce(
    (sum, point) => ({
      x: sum.x + clamp(point.x, 0, 1) * sourceWidth,
      y: sum.y + clamp(point.y, 0, 1) * sourceHeight,
    }),
    { x: 0, y: 0 },
  );
  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function averageNumbers(values) {
  const finiteValues = values.filter((value) => Number.isFinite(value) && value > 0);
  if (!finiteValues.length) {
    throw new Error("无法计算对齐尺度");
  }
  return finiteValues.reduce((total, value) => total + value, 0) / finiteValues.length;
}

async function cropToBlob() {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) throw new Error("无法生成裁剪后的人脸图片");
  return blob;
}

function landmarksToCropBox(landmarks, sourceWidth, sourceHeight) {
  const xs = landmarks.map((point) => clamp(point.x, 0, 1) * sourceWidth);
  const ys = landmarks.map((point) => clamp(point.y, 0, 1) * sourceHeight);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const pad = Math.max(width, height) * 0.25;
  const x = clamp(Math.floor(minX - pad), 0, sourceWidth - 1);
  const y = clamp(Math.floor(minY - pad), 0, sourceHeight - 1);
  const right = clamp(Math.ceil(maxX + pad), x + 1, sourceWidth);
  const bottom = clamp(Math.ceil(maxY + pad), y + 1, sourceHeight);
  return {
    x,
    y,
    width: right - x,
    height: bottom - y,
  };
}

function drawCropPreview() {
  const context = faceCanvas.getContext("2d");
  faceCanvas.width = canvas.width;
  faceCanvas.height = canvas.height;
  context.drawImage(canvas, 0, 0, faceCanvas.width, faceCanvas.height);
}

function drawOverlay(cropBox, sourceWidth, sourceHeight) {
  const context = overlayCanvas.getContext("2d");
  overlayCanvas.width = sourceWidth;
  overlayCanvas.height = sourceHeight;
  context.clearRect(0, 0, sourceWidth, sourceHeight);
  context.lineWidth = Math.max(4, Math.round(sourceWidth / 180));
  context.strokeStyle = "#1fd1a5";
  context.fillStyle = "rgba(31, 209, 165, 0.12)";
  context.fillRect(cropBox.x, cropBox.y, cropBox.width, cropBox.height);
  context.strokeRect(cropBox.x, cropBox.y, cropBox.width, cropBox.height);
}

function clearCanvas(targetCanvas) {
  const context = targetCanvas.getContext("2d");
  context.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
}

function renderNoFace(message) {
  fields.preprocessStatus.textContent = "no face";
  fields.rawEmotion.textContent = "-";
  fields.emotion.textContent = "-";
  fields.score.textContent = "-";
  fields.riskLevel.textContent = "-";
  fields.confidence.textContent = "-";
  fields.parseStatus.textContent = "-";
  fields.thinkText.textContent = "-";
  fields.rawResponse.textContent = "-";
  fields.evidence.textContent = message;
}

function renderResult(payload) {
  const features = payload.features || {};
  fields.emotion.textContent = payload.visualEmotion || payload.emotion || "-";
  fields.rawEmotion.textContent = features.rawEmotion || "-";
  fields.score.textContent = formatNumber(payload.visualScore ?? payload.score, 2);
  fields.riskLevel.textContent = payload.riskLevel || "-";
  fields.confidence.textContent = formatNumber(payload.confidence, 2);
  fields.runtimeStatus.textContent = "ready";
  fields.preprocessStatus.textContent = features.preprocess?.mode || "aligned";
  fields.parseStatus.textContent = features.parseStatus || "-";
  fields.thinkText.textContent = features.think || "-";
  fields.rawResponse.textContent = features.modelResponse || "-";
  fields.evidence.textContent = payload.evidence || "";
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

function readableError(error) {
  if (!error) return "未知错误";
  if (error.message) return error.message;
  return String(error);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", analyzeFrame);
autoToggle.addEventListener("change", () => {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (autoToggle.checked) {
    analyzeFrame();
    timer = setInterval(analyzeFrame, AUTO_ANALYZE_MS);
  }
});

checkService();
