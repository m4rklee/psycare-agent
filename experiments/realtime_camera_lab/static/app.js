const TASKS_VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const state = {
  stream: null,
  timer: null,
  analyzing: false,
  intervalMs: 750,
  faceLandmarker: null,
  mediaPipeReady: false,
  lastVideoTime: -1
};

const $ = (selector) => document.querySelector(selector);

const els = {
  serviceState: $("#serviceState"),
  preview: $("#cameraPreview"),
  previewEmpty: $("#previewEmpty"),
  startCamera: $("#startCamera"),
  stopCamera: $("#stopCamera"),
  analyzeOnce: $("#analyzeOnce"),
  autoAnalyze: $("#autoAnalyze"),
  cameraState: $("#cameraState"),
  emotionValue: $("#emotionValue"),
  riskValue: $("#riskValue"),
  scoreValue: $("#scoreValue"),
  confidenceValue: $("#confidenceValue"),
  timeValue: $("#timeValue"),
  faceValue: $("#faceValue"),
  landmarkValue: $("#landmarkValue"),
  evidenceValue: $("#evidenceValue"),
  featureList: $("#featureList")
};

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    setService(data.status === "UP" ? "实验服务正常" : `服务 ${data.status}`, data.status === "UP");
  } catch (error) {
    setService("实验服务不可用", false);
  }
}

function setService(text, ok) {
  els.serviceState.textContent = text;
  els.serviceState.classList.toggle("ok", ok);
  els.serviceState.classList.toggle("danger", !ok);
}

async function ensureFaceLandmarker() {
  if (state.faceLandmarker) return state.faceLandmarker;
  els.cameraState.textContent = "正在加载 MediaPipe FaceLandmarker 模型...";
  const visionModule = await import(TASKS_VISION_URL);
  const { FaceLandmarker, FilesetResolver } = visionModule;
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false
  });
  state.mediaPipeReady = true;
  return state.faceLandmarker;
}

async function startCamera() {
  clearStatus();
  try {
    await ensureFaceLandmarker();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: false
    });
    els.preview.srcObject = state.stream;
    els.previewEmpty.hidden = true;
    els.startCamera.disabled = true;
    els.stopCamera.disabled = false;
    els.analyzeOnce.disabled = false;
    els.autoAnalyze.disabled = false;
    els.cameraState.textContent = "FaceLandmarker 已就绪，摄像头画面仅在本地浏览器分析。";
  } catch (error) {
    els.cameraState.textContent = cameraErrorMessage(error);
  }
}

function stopCamera() {
  stopAutoAnalyze();
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.lastVideoTime = -1;
  els.preview.srcObject = null;
  els.previewEmpty.hidden = false;
  els.startCamera.disabled = false;
  els.stopCamera.disabled = true;
  els.analyzeOnce.disabled = true;
  els.autoAnalyze.checked = false;
  els.autoAnalyze.disabled = true;
  els.cameraState.textContent = "摄像头已停止。";
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") return "摄像头权限被拒绝，请在浏览器地址栏重新允许摄像头。";
  if (error?.name === "NotFoundError") return "没有检测到可用摄像头。";
  if (!navigator.mediaDevices?.getUserMedia) return "当前浏览器不支持 getUserMedia。";
  return "FaceLandmarker 或摄像头启动失败：" + readableError(error);
}

async function analyzeOnce() {
  if (!state.stream || state.analyzing) return;
  if (!els.preview.videoWidth || !els.preview.videoHeight) {
    els.cameraState.textContent = "摄像头画面尚未就绪。";
    return;
  }
  if (els.preview.currentTime === state.lastVideoTime) return;

  state.analyzing = true;
  state.lastVideoTime = els.preview.currentTime;
  try {
    const detection = state.faceLandmarker.detectForVideo(els.preview, performance.now());
    const localPayload = buildPayload(detection);
    const normalized = normalizePayload(localPayload);
    renderResult(normalized);
    els.cameraState.textContent = localPayload.faceDetected ? "Face Mesh 检测完成。" : "未检测到人脸。";
  } catch (error) {
    els.cameraState.textContent = "分析失败：" + readableError(error);
  } finally {
    state.analyzing = false;
  }
}

function normalizePayload(payload) {
  const features = {
    ...(payload.features || {}),
    faceDetected: !!payload.faceDetected,
    landmarkCount: Number(payload.landmarkCount || payload.features?.landmarkCount || 0)
  };
  const score = clamp(Number(payload.visualScore ?? payload.score ?? features.visualScore ?? 0), 0, 4.5);
  const emotion = payload.visualEmotion || payload.emotion || emotionFromScore(score);
  const confidence = clamp(Number(payload.confidence ?? (features.faceDetected ? 0.72 : 0.35)), 0, 1);
  features.visualScore = score;
  return {
    emotion,
    visualEmotion: emotion,
    score,
    visualScore: score,
    riskLevel: payload.riskLevel || riskFromScore(score),
    confidence,
    evidence: payload.evidence || defaultEvidence(features.faceDetected, features.landmarkCount, score),
    features,
    timestamp: new Date().toISOString()
  };
}

function defaultEvidence(faceDetected, landmarkCount, score) {
  if (!faceDetected) return "MediaPipe FaceLandmarker 未检测到人脸。";
  return `MediaPipe FaceLandmarker 浏览器端分析：landmarkCount=${landmarkCount}, visualScore=${score.toFixed(2)}。`;
}

function buildPayload(detection) {
  const landmarks = detection.faceLandmarks?.[0] || [];
  const blendshapes = categoriesToMap(detection.faceBlendshapes?.[0]?.categories || []);
  if (!landmarks.length) {
    return {
      faceDetected: false,
      landmarkCount: 0,
      score: 0.0,
      confidence: 0.32,
      evidence: "MediaPipe FaceLandmarker 未检测到人脸。",
      features: {
        faceDetected: false,
        landmarkCount: 0,
        facePresence: 0.0,
        visualScore: 0.0
      }
    };
  }

  const features = extractFaceMeshFeatures(landmarks, blendshapes);
  const score = scoreFeatures(features);
  const emotion = emotionFromScore(score);
  const confidence = clamp(0.58 + features.facePresence * 0.2 + features.landmarkStability * 0.16, 0.5, 0.92);
  const evidence = [
    "MediaPipe FaceLandmarker 浏览器端分析",
    `landmarkCount=${landmarks.length}`,
    `browTension=${features.browTension.toFixed(2)}`,
    `eyeTension=${features.eyeTension.toFixed(2)}`,
    `mouthDown=${features.mouthDown.toFixed(2)}`,
    `muscleTension=${features.muscleTension.toFixed(2)}`,
    `visualScore=${score.toFixed(2)}`
  ].join("; ");

  return {
    faceDetected: true,
    landmarkCount: landmarks.length,
    emotion,
    visualEmotion: emotion,
    score,
    visualScore: score,
    riskLevel: riskFromScore(score),
    confidence,
    evidence,
    features
  };
}

function categoriesToMap(categories) {
  const values = {};
  categories.forEach((category) => {
    values[category.categoryName] = category.score;
  });
  return values;
}

function extractFaceMeshFeatures(landmarks, blendshapes) {
  const leftEyeOpen = distance(landmarks[159], landmarks[145]);
  const rightEyeOpen = distance(landmarks[386], landmarks[374]);
  const mouthCenter = landmarks[13]?.y ?? 0.5;
  const mouthLeft = landmarks[61]?.y ?? mouthCenter;
  const mouthRight = landmarks[291]?.y ?? mouthCenter;
  const mouthDownGeometry = clamp(((mouthLeft + mouthRight) / 2 - mouthCenter) * 10, 0, 1);

  const browDown = maxBlend(blendshapes, ["browDownLeft", "browDownRight"]);
  const browInnerUp = blendshapes.browInnerUp || 0;
  const eyeSquint = maxBlend(blendshapes, ["eyeSquintLeft", "eyeSquintRight"]);
  const eyeBlink = maxBlend(blendshapes, ["eyeBlinkLeft", "eyeBlinkRight"]);
  const mouthFrown = maxBlend(blendshapes, ["mouthFrownLeft", "mouthFrownRight"]);
  const mouthPress = maxBlend(blendshapes, ["mouthPressLeft", "mouthPressRight"]);
  const jawOpen = blendshapes.jawOpen || 0;

  const eyeOpenness = clamp((leftEyeOpen + rightEyeOpen) * 18, 0, 1);
  const eyeTension = clamp(eyeSquint * 0.55 + eyeBlink * 0.35 + (1 - eyeOpenness) * 0.18, 0, 1);
  const browTension = clamp(browDown * 0.7 + browInnerUp * 0.35, 0, 1);
  const mouthDown = clamp(mouthFrown * 0.72 + mouthDownGeometry * 0.35, 0, 1);
  const muscleTension = clamp((browTension + eyeTension + mouthPress + mouthDown) / 4, 0, 1);
  const facePresence = clamp(0.7 + Math.min(landmarks.length, 478) / 478 * 0.3, 0, 1);
  const landmarkStability = clamp(1 - Math.abs((landmarks[10]?.x || 0.5) - (landmarks[152]?.x || 0.5)), 0, 1);

  return {
    faceDetected: true,
    landmarkCount: landmarks.length,
    browTension,
    browShadow: browTension,
    eyeTension,
    eyeOpenness,
    eyeDarkness: eyeTension,
    mouthDown,
    mouthPress,
    jawOpen,
    muscleTension,
    facePresence,
    landmarkStability,
    visualScore: 0
  };
}

function scoreFeatures(features) {
  const browScore = features.browTension > 0.55 ? 1.5 : features.browTension > 0.32 ? 0.8 : 0.0;
  const eyeScore = features.eyeTension > 0.55 ? 1.0 : features.eyeTension > 0.32 ? 0.5 : 0.0;
  const mouthScore = features.mouthDown > 0.45 ? 1.0 : features.mouthDown > 0.22 ? 0.5 : 0.0;
  const tensionScore = features.muscleTension > 0.48 ? 1.5 : features.muscleTension > 0.30 ? 0.8 : 0.0;
  const lowPresenceScore = features.facePresence < 0.55 ? 0.4 : 0.0;
  const score = clamp(browScore + eyeScore + mouthScore + tensionScore + lowPresenceScore, 0, 4.5);
  features.visualScore = score;
  return score;
}

function emotionFromScore(score) {
  if (score >= 4.0) return "HIGH_RISK";
  if (score >= 3.0) return "DEPRESSED";
  if (score >= 2.0) return "ANXIETY";
  return "NORMAL";
}

function riskFromScore(score) {
  if (score >= 4.0) return "HIGH";
  if (score >= 3.0) return "MEDIUM";
  return "LOW";
}

function toggleAutoAnalyze() {
  if (els.autoAnalyze.checked) {
    analyzeOnce();
    state.timer = window.setInterval(analyzeOnce, state.intervalMs);
    els.cameraState.textContent = "自动 Face Mesh 分析已开启。";
    return;
  }
  stopAutoAnalyze();
  els.cameraState.textContent = "自动分析已关闭。";
}

function stopAutoAnalyze() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
}

function renderResult(result) {
  const features = result.features || {};
  els.emotionValue.textContent = result.visualEmotion || result.emotion || "--";
  els.riskValue.textContent = result.riskLevel ?? "--";
  els.scoreValue.textContent = formatNumber(result.visualScore ?? result.score);
  els.confidenceValue.textContent = formatPercent(result.confidence);
  els.timeValue.textContent = result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : "--";
  els.faceValue.textContent = features.faceDetected ? "Detected" : "No face";
  els.landmarkValue.textContent = String(features.landmarkCount ?? "--");
  els.evidenceValue.textContent = result.evidence ?? "无分析依据。";
  renderFeatures(features);
}

function renderFeatures(features) {
  const items = [
    ["browTension", features.browTension],
    ["eyeTension", features.eyeTension],
    ["mouthDown", features.mouthDown],
    ["muscleTension", features.muscleTension],
    ["eyeOpenness", features.eyeOpenness],
    ["mouthPress", features.mouthPress],
    ["facePresence", features.facePresence],
    ["landmarkStability", features.landmarkStability]
  ];
  els.featureList.innerHTML = items.map(([name, value]) => (
    `<div><dt>${name}</dt><dd>${formatNumber(value)}</dd></div>`
  )).join("");
}

function distance(left, right) {
  if (!left || !right) return 0;
  return Math.hypot(left.x - right.x, left.y - right.y, (left.z || 0) - (right.z || 0));
}

function maxBlend(blendshapes, names) {
  return Math.max(...names.map((name) => blendshapes[name] || 0));
}

function formatNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "--";
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "--";
}

function readableError(error) {
  const message = String(error?.message || error || "未知错误");
  return message.length > 180 ? `${message.slice(0, 180)}...` : message;
}

function clearStatus() {
  els.cameraState.textContent = "";
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

window.addEventListener("beforeunload", stopCamera);
els.startCamera.addEventListener("click", startCamera);
els.stopCamera.addEventListener("click", stopCamera);
els.analyzeOnce.addEventListener("click", analyzeOnce);
els.autoAnalyze.addEventListener("change", toggleAutoAnalyze);

checkHealth();
