const video = document.querySelector("#camera");
const canvas = document.querySelector("#canvas");
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
  weightsStatus: document.querySelector("#weightsStatus"),
  probabilities: document.querySelector("#probabilities"),
  evidence: document.querySelector("#evidence"),
};

let stream = null;
let timer = null;
let busy = false;

async function checkService() {
  try {
    const [health, models] = await Promise.all([fetch("/health"), fetch("/models")]);
    if (!health.ok || !models.ok) throw new Error("service unavailable");
    const modelData = await models.json();
    serviceStatus.textContent = "服务就绪";
    fields.weightsStatus.textContent = modelData.weightsStatus || "-";
    if (!modelData.weightsPresent) {
      fields.evidence.textContent = `缺少 POSTER-Var 权重：${modelData.checkpointPath}`;
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
}

async function analyzeFrame() {
  if (!stream || busy || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  busy = true;
  try {
    const context = canvas.getContext("2d");
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
    if (!blob) throw new Error("无法生成图片帧");
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    const response = await fetch("/analyze-frame", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "分析失败");
    renderResult(payload);
  } catch (error) {
    fields.evidence.textContent = `分析失败：${error.message}`;
  } finally {
    busy = false;
  }
}

function renderResult(payload) {
  fields.emotion.textContent = payload.visualEmotion || payload.emotion || "-";
  fields.rawEmotion.textContent = payload.features?.rawEmotion || "-";
  fields.score.textContent = formatNumber(payload.visualScore ?? payload.score);
  fields.riskLevel.textContent = payload.riskLevel || "-";
  fields.confidence.textContent = formatNumber(payload.confidence);
  fields.weightsStatus.textContent = "ready";
  fields.evidence.textContent = payload.evidence || "";
  renderProbabilities(payload.features?.probabilities || {});
}

function renderProbabilities(probabilities) {
  const entries = Object.entries(probabilities).sort((a, b) => Number(b[1]) - Number(a[1]));
  fields.probabilities.innerHTML = "";
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
    fields.probabilities.append(row);
  }
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
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
    timer = setInterval(analyzeFrame, 2000);
  }
});

checkService();
