const CLIP_MS = 2500;
const LOOP_GAP_MS = 800;
const video = document.querySelector("#camera");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const recordBtn = document.querySelector("#recordBtn");
const autoToggle = document.querySelector("#autoToggle");
const serviceStatus = document.querySelector("#serviceStatus");
const fields = {
  emotion: document.querySelector("#emotion"),
  microExpression: document.querySelector("#microExpression"),
  score: document.querySelector("#score"),
  riskLevel: document.querySelector("#riskLevel"),
  confidence: document.querySelector("#confidence"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  probabilities: document.querySelector("#probabilities"),
  aus: document.querySelector("#aus"),
  evidence: document.querySelector("#evidence"),
};

let stream = null;
let autoRunning = false;
let busy = false;

async function checkService() {
  try {
    const [health, models] = await Promise.all([fetch("/health"), fetch("/models")]);
    if (!health.ok || !models.ok) throw new Error("service unavailable");
    const modelData = await models.json();
    serviceStatus.textContent = "服务就绪";
    fields.runtimeStatus.textContent = modelData.runtimeStatus || "-";
    if (!modelData.runtimeReady) {
      fields.evidence.textContent = `${modelData.runtimeStatus}: ${modelData.checkpointPath}`;
    }
  } catch (error) {
    serviceStatus.textContent = "服务不可用";
  }
}

async function startCamera() {
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    fields.evidence.textContent = "摄像头已开启。";
  } catch (error) {
    fields.evidence.textContent = `无法开启摄像头：${error.message}`;
  }
}

function stopCamera() {
  autoRunning = false;
  autoToggle.checked = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
}

async function recordAndAnalyze() {
  if (!stream || busy) return;
  busy = true;
  recordBtn.disabled = true;
  try {
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    });
    const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
    recorder.start();
    fields.evidence.textContent = "正在录制 2.5 秒短视频...";
    await wait(CLIP_MS);
    recorder.stop();
    await stopped;
    const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "video/webm" });
    await uploadClip(blob);
  } catch (error) {
    fields.evidence.textContent = `分析失败：${error.message}`;
  } finally {
    busy = false;
    recordBtn.disabled = false;
  }
}

async function uploadClip(blob) {
  if (!blob || blob.size === 0) throw new Error("录制片段为空");
  const form = new FormData();
  const ext = blob.type.includes("mp4") ? "mp4" : "webm";
  form.append("file", blob, `clip.${ext}`);
  const response = await fetch("/analyze-clip", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "分析失败");
  renderResult(payload);
}

async function autoLoop() {
  if (autoRunning) return;
  autoRunning = true;
  while (autoRunning && stream) {
    await recordAndAnalyze();
    await wait(LOOP_GAP_MS);
  }
}

function pickMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function renderResult(payload) {
  fields.emotion.textContent = payload.visualEmotion || payload.emotion || "-";
  fields.microExpression.textContent = payload.microExpression || payload.features?.microExpression || "-";
  fields.score.textContent = formatNumber(payload.visualScore ?? payload.score);
  fields.riskLevel.textContent = payload.riskLevel || "-";
  fields.confidence.textContent = formatNumber(payload.confidence);
  fields.runtimeStatus.textContent = "ready";
  fields.evidence.textContent = payload.evidence || "";
  renderBars(fields.probabilities, payload.features?.probabilities || {});
  renderBars(fields.aus, payload.features?.auPredictions || {});
}

function renderBars(container, values) {
  const entries = Object.entries(values).sort((a, b) => Number(b[1]) - Number(a[1]));
  container.innerHTML = "";
  for (const [label, value] of entries) {
    const row = document.createElement("div");
    row.className = "probability-row";
    const name = document.createElement("span");
    name.textContent = label;
    const bar = document.createElement("div");
    bar.className = "probability-bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(1, Number(value))) * 100}%`;
    const number = document.createElement("strong");
    number.textContent = formatNumber(value);
    bar.append(fill);
    row.append(name, bar, number);
    container.append(row);
  }
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
recordBtn.addEventListener("click", recordAndAnalyze);
autoToggle.addEventListener("change", () => {
  autoRunning = autoToggle.checked;
  if (autoRunning) autoLoop();
});

checkService();
