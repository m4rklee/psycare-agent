const state = {
  auth: { username: "student", password: "student123" },
  sessionId: null,
  sending: false,
  isAdmin: false,
  sessions: [],
  activeSessionId: null,
  sidebarOpen: window.innerWidth > 780,
  modelName: "multimodalAgent-qwen2.5-7b-ft:latest",
  latestReports: [],
  recording: false,
  mediaRecorder: null,
  recordedChunks: [],
  recordedAudioFile: null,
  recordingStream: null,
  discardRecording: false,
  attachmentNotice: "",
  videoStream: null,
  videoSegmentRecorder: null,
  videoSegmentChunks: [],
  videoSegmentRecording: false,
  videoDiscardSegment: false,
  videoSending: false,
  videoVad: null,
  videoVadReady: false,
  videoManualMode: true
};

const $ = (selector) => document.querySelector(selector);

const els = {
  loginForm: $("#loginForm"),
  username: $("#username"),
  password: $("#password"),
  loginState: $("#loginState"),
  accountPanel: $("#accountPanel"),
  activeAccount: $("#activeAccount"),
  activeRole: $("#activeRole"),
  switchAccount: $("#switchAccount"),
  studentView: $("#studentView"),
  adminView: $("#adminView"),
  historySidebar: $("#historySidebar"),
  historyList: $("#historyList"),
  historyEmpty: $("#historyEmpty"),
  sidebarToggle: $("#sidebarToggle"),
  mobileHistoryToggle: $("#mobileHistoryToggle"),
  historyOverlay: $("#historyOverlay"),
  messages: $("#messages"),
  chatForm: $("#chatForm"),
  messageInput: $("#messageInput"),
  audioInput: $("#audioInput"),
  micButton: $("#micButton"),
  imageInput: $("#imageInput"),
  videoInput: $("#videoInput"),
  videoChatButton: $("#videoChatButton"),
  videoChatOverlay: $("#videoChatOverlay"),
  closeVideoChat: $("#closeVideoChat"),
  videoChatPreview: $("#videoChatPreview"),
  videoChatCanvas: $("#videoChatCanvas"),
  videoChatState: $("#videoChatState"),
  videoVadState: $("#videoVadState"),
  videoEmotion: $("#videoEmotion"),
  videoRisk: $("#videoRisk"),
  videoConfidence: $("#videoConfidence"),
  videoEvidence: $("#videoEvidence"),
  videoPromptInput: $("#videoPromptInput"),
  startVideoChat: $("#startVideoChat"),
  videoTalkButton: $("#videoTalkButton"),
  endVideoChat: $("#endVideoChat"),
  videoReplyStream: $("#videoReplyStream"),
  attachmentState: $("#attachmentState"),
  clearAttachments: $("#clearAttachments"),
  newSessionButton: $("#newSessionButton"),
  sendButton: $("#sendButton"),
  adminRefresh: $("#adminRefresh"),
  adminStats: $("#adminStats"),
  queueCount: $("#queueCount"),
  adminReportRows: $("#adminReportRows"),
  excelRows: $("#excelRows"),
  emailRows: $("#emailRows"),
  knowledgeUploadForm: $("#knowledgeUploadForm"),
  knowledgeFile: $("#knowledgeFile"),
  knowledgeUploadState: $("#knowledgeUploadState"),
  detailOverlay: $("#detailOverlay"),
  detailKicker: $("#detailKicker"),
  detailTitle: $("#detailTitle"),
  detailMeta: $("#detailMeta"),
  detailBody: $("#detailBody"),
  closeDetail: $("#closeDetail")
};

const pipelineStatusText = {
  input: "正在接入多模态输入",
  fusion: "正在融合情绪信号",
  router: "正在判断对话意图",
  rag: "正在检索知识库",
  mcp: "正在同步工具结果",
  video: "正在分析视频表情",
  stream: "正在生成回复",
  done: "已完成",
  failed: "生成中断"
};

function authHeader() {
  return `Basic ${btoa(`${state.auth.username}:${state.auth.password}`)}`;
}

async function api(path, options = {}) {
  const headers = { Authorization: authHeader(), ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    throw new Error(await response.text() || `${response.status} ${response.statusText}`);
  }
  return response;
}

function displayModelName(model) {
  return "心理咨询AI";
}

function setModel(status) {
  state.modelName = status.model || state.modelName;
}

function selectedFiles() {
  const audioFile = state.recordedAudioFile || els.audioInput.files?.[0];
  return [
    ["audio", state.recordedAudioFile ? "录音" : "语音", audioFile],
    ["image", "图像", els.imageInput.files?.[0]],
    ["video", "视频", els.videoInput.files?.[0]]
  ].filter(([, , file]) => file);
}

function updateAttachments() {
  const files = selectedFiles();
  const hasStatus = Boolean(state.attachmentNotice || state.recording || files.length);
  els.clearAttachments.hidden = files.length === 0 && !state.attachmentNotice;
  els.attachmentState.hidden = !hasStatus;
  els.chatForm.classList.toggle("has-attachments", hasStatus);
  els.attachmentState.textContent = state.recording
    ? "正在录音，点击麦克风停止"
    : state.attachmentNotice || (files.length
    ? files.map(([, label, file]) => `${label} / ${file.name}`).join("    ")
    : "暂无附件");
  els.attachmentState.classList.toggle("active", hasStatus);
  els.micButton.classList.toggle("recording", state.recording);
  els.micButton.setAttribute("aria-label", state.recording ? "停止录音" : "开始录音");
  els.micButton.title = state.recording ? "停止录音" : "开始录音";
  els.micButton.disabled = state.sending;
}

function resetRecordingState() {
  if (state.recordingStream) {
    state.recordingStream.getTracks().forEach((track) => track.stop());
  }
  state.recording = false;
  state.mediaRecorder = null;
  state.recordedChunks = [];
  state.recordingStream = null;
}

function clearAttachments(options = {}) {
  if (state.recording) {
    state.discardRecording = true;
    state.mediaRecorder?.stop();
  }
  resetRecordingState();
  if (!options.keepNotice) state.attachmentNotice = "";
  state.recordedAudioFile = null;
  els.audioInput.value = "";
  els.imageInput.value = "";
  els.videoInput.value = "";
  updateAttachments();
}

function recordingMimeType() {
  if (!window.MediaRecorder) return "";
  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus"
  ];
  return preferred.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function audioExtension(type) {
  if (type.includes("mp4")) return "m4a";
  if (type.includes("ogg")) return "ogg";
  return "webm";
}

function showAttachmentNotice(text) {
  state.attachmentNotice = text;
  updateAttachments();
}

async function startRecording() {
  if (state.sending || state.isAdmin || state.recording) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showAttachmentNotice("当前浏览器不支持录音");
    return;
  }
  try {
    clearAttachments();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = recordingMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    state.recordingStream = stream;
    state.mediaRecorder = recorder;
    state.recordedChunks = [];
    state.discardRecording = false;
    state.recording = true;
    state.attachmentNotice = "";
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data?.size) state.recordedChunks.push(event.data);
    });
    recorder.addEventListener("stop", finishRecording);
    recorder.start();
    updateAttachments();
  } catch (error) {
    resetRecordingState();
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    showAttachmentNotice(denied ? "未获得麦克风权限" : "录音启动失败");
  }
}

function stopRecording() {
  if (!state.recording || !state.mediaRecorder) return;
  state.discardRecording = false;
  if (state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
}

function finishRecording() {
  if (state.discardRecording) {
    state.discardRecording = false;
    resetRecordingState();
    updateAttachments();
    return;
  }
  const chunks = [...state.recordedChunks];
  const type = state.mediaRecorder?.mimeType || recordingMimeType() || "audio/webm";
  resetRecordingState();
  if (!chunks.length) {
    showAttachmentNotice("录音为空，请重试");
    return;
  }
  const blob = new Blob(chunks, { type });
  if (!blob.size) {
    showAttachmentNotice("录音为空，请重试");
    return;
  }
  const extension = audioExtension(type);
  state.recordedAudioFile = new File([blob], `mic-recording-${Date.now()}.${extension}`, {
    type: blob.type || "audio/webm"
  });
  state.attachmentNotice = "";
  updateAttachments();
  if (!state.sending && !state.isAdmin) {
    els.chatForm.requestSubmit();
  }
}

function toggleRecording() {
  if (state.recording) {
    stopRecording();
  } else {
    void startRecording();
  }
}

function setVideoState(text) {
  els.videoChatState.textContent = text;
}

function setVideoVad(text) {
  els.videoVadState.textContent = text;
}

function resetVideoMetrics() {
  els.videoEmotion.textContent = "-";
  els.videoRisk.textContent = "-";
  els.videoConfidence.textContent = "-";
  els.videoEvidence.textContent = "开启后会分析当前人脸帧。";
}

function openVideoChat() {
  if (state.isAdmin) return;
  els.videoChatOverlay.hidden = false;
  document.body.classList.add("video-chat-open");
  resetVideoMetrics();
  setVideoState("准备开启摄像头");
  setVideoVad("等待启动");
  void startVideoChat();
}

async function startVideoChat() {
  if (state.videoStream) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setVideoState("当前浏览器不支持摄像头或录音");
    return;
  }
  try {
    state.videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: true
    });
    els.videoChatPreview.srcObject = state.videoStream;
    await els.videoChatPreview.play().catch(() => null);
    els.videoTalkButton.disabled = false;
    setVideoState("视频已开启");
    await startBrowserVad();
  } catch (error) {
    setVideoState(cameraErrorMessage(error));
    stopVideoChat();
  }
}

async function startBrowserVad() {
  try {
    const vadModule = await import("https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/bundle.min.js");
    const MicVAD = vadModule.MicVAD || vadModule.default?.MicVAD;
    if (!MicVAD) throw new Error("MicVAD unavailable");
    state.videoVad = await MicVAD.new({
      onSpeechStart: () => {
        setVideoVad("检测到说话");
        startVideoSegment();
      },
      onSpeechEnd: () => {
        setVideoVad("检测到停顿，正在发送");
        stopVideoSegment();
      }
    });
    state.videoVad.start();
    state.videoVadReady = true;
    state.videoManualMode = false;
    els.videoTalkButton.textContent = "自动分轮中";
    els.videoTalkButton.disabled = true;
    setVideoVad("VAD 已开启");
  } catch (error) {
    state.videoVadReady = false;
    state.videoManualMode = true;
    els.videoTalkButton.textContent = "开始一轮";
    els.videoTalkButton.disabled = !state.videoStream;
    setVideoVad("VAD 不可用，请手动开始一轮");
  }
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") return "摄像头或麦克风权限被拒绝";
  if (error?.name === "NotFoundError") return "没有检测到可用摄像头或麦克风";
  return "视频启动失败";
}

function startVideoSegment() {
  if (!state.videoStream || state.videoSegmentRecording || state.videoSending) return;
  const audioTracks = state.videoStream.getAudioTracks();
  if (!audioTracks.length) {
    setVideoState("未检测到麦克风音轨");
    return;
  }
  const audioStream = new MediaStream(audioTracks);
  const mimeType = recordingMimeType();
  const recorder = mimeType ? new MediaRecorder(audioStream, { mimeType }) : new MediaRecorder(audioStream);
  state.videoSegmentChunks = [];
  state.videoDiscardSegment = false;
  state.videoSegmentRecorder = recorder;
  state.videoSegmentRecording = true;
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data?.size) state.videoSegmentChunks.push(event.data);
  });
  recorder.addEventListener("stop", () => void finishVideoSegment(recorder.mimeType || mimeType || "audio/webm"));
  recorder.start();
  setVideoState("正在聆听");
  if (state.videoManualMode) els.videoTalkButton.textContent = "停止并发送";
}

function stopVideoSegment() {
  if (!state.videoSegmentRecording || !state.videoSegmentRecorder) return;
  if (state.videoSegmentRecorder.state !== "inactive") state.videoSegmentRecorder.stop();
}

async function finishVideoSegment(type) {
  const chunks = [...state.videoSegmentChunks];
  const discard = state.videoDiscardSegment;
  state.videoSegmentChunks = [];
  state.videoSegmentRecorder = null;
  state.videoSegmentRecording = false;
  state.videoDiscardSegment = false;
  if (state.videoManualMode) els.videoTalkButton.textContent = "开始一轮";
  if (discard) {
    setVideoState("视频已关闭");
    return;
  }
  if (!chunks.length || state.videoSending) {
    setVideoState(chunks.length ? "正在处理上一轮" : "没有检测到语音");
    return;
  }
  const audio = new File([new Blob(chunks, { type })], `video-turn-${Date.now()}.${audioExtension(type)}`, {
    type: type || "audio/webm"
  });
  const frame = await captureVideoFrame();
  if (!frame) {
    setVideoState("当前视频帧不可用");
    return;
  }
  await sendVideoTurn(audio, frame);
}

function captureVideoFrame() {
  return new Promise((resolve) => {
    const video = els.videoChatPreview;
    if (!video.videoWidth || !video.videoHeight) {
      resolve(null);
      return;
    }
    const canvas = els.videoChatCanvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      resolve(blob ? new File([blob], `poster-frame-${Date.now()}.jpg`, { type: "image/jpeg" }) : null);
    }, "image/jpeg", 0.86);
  });
}

function renderVideoReply(text) {
  els.videoReplyStream.textContent = text || "等待本轮回复";
}

function updateVideoMetricsFromMeta(eventData) {
  if (eventData.visualEmotion) els.videoEmotion.textContent = eventData.visualEmotion;
  if (eventData.visualRiskLevel) els.videoRisk.textContent = eventData.visualRiskLevel;
  if (eventData.visualConfidence !== undefined) els.videoConfidence.textContent = Number(eventData.visualConfidence).toFixed(2);
  if (eventData.visualEvidence) els.videoEvidence.textContent = eventData.visualEvidence;
}

async function sendVideoTurn(audioFile, frameFile) {
  state.videoSending = true;
  els.videoTalkButton.disabled = true;
  setVideoState("正在分析视频表情");
  renderVideoReply("正在分析视频表情...");
  clearEmpty();
  const assistant = addMessage("assistant", "");
  setPipelineStatus(assistant, "video");
  let displayedUserCard = null;
  let output = "";
  let streamErrored = false;
  try {
    const response = await sendVideoChat(els.videoPromptInput.value.trim(), audioFile, frameFile);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") {
          const changedSession = state.sessionId !== eventData.sessionId;
          state.sessionId = eventData.sessionId;
          state.activeSessionId = eventData.sessionId;
          if (eventData.displayText && !displayedUserCard) displayedUserCard = addMessage("user", eventData.displayText, assistant);
          updateVideoMetricsFromMeta(eventData);
          if (changedSession) void loadChatSessions(eventData.sessionId);
          setPipelineStatus(assistant, "rag");
          setVideoState("正在生成回复");
        }
        if (eventData.type === "token") {
          output += eventData.content;
          updateAssistant(assistant, output);
          renderVideoReply(output);
          setPipelineStatus(assistant, "stream");
        }
        if (eventData.type === "agent") {
          addAgentSidecarCard(eventData, assistant);
        }
        if (eventData.type === "error") {
          streamErrored = true;
          output = eventData.content || "视频聊天暂时不可用。";
          updateAssistant(assistant, output);
          renderVideoReply(output);
          setPipelineStatus(assistant, "failed");
          setVideoState("本轮失败");
        }
      });
    }
    if (!displayedUserCard && !streamErrored) displayedUserCard = addMessage("user", els.videoPromptInput.value.trim() || "视频语音输入", assistant);
    if (!output) {
      output = streamErrored ? output : "模型暂时没有返回内容。";
      updateAssistant(assistant, output);
      renderVideoReply(output);
    }
    if (!streamErrored) {
      setPipelineStatus(assistant, "done");
      setVideoState("本轮完成");
    }
  } catch (error) {
    updateAssistant(assistant, "视频聊天请求失败，请确认 POSTER++ 服务和后端已启动。");
    renderVideoReply("视频聊天请求失败，请确认 POSTER++ 服务和后端已启动。");
    setPipelineStatus(assistant, "failed");
    setVideoState("请求失败");
  } finally {
    state.videoSending = false;
    els.videoTalkButton.disabled = !state.videoStream || !state.videoManualMode;
    if (state.sessionId) void loadChatSessions(state.sessionId);
  }
}

function sendVideoChat(message, audioFile, frameFile) {
  const body = new FormData();
  body.append("message", message || "学生正在进行视频心理支持对话。");
  if (state.sessionId) body.append("sessionId", state.sessionId);
  body.append("audio", audioFile);
  body.append("frame", frameFile);
  body.append("preprocessMode", "browser-video-frame");
  body.append("fallback", "true");
  return api("/api/chat/video/stream", { method: "POST", body });
}

function toggleManualVideoTurn() {
  if (!state.videoStream || state.videoSending || !state.videoManualMode) return;
  if (state.videoSegmentRecording) stopVideoSegment();
  else startVideoSegment();
}

function stopVideoChat() {
  if (state.videoSegmentRecording) {
    state.videoDiscardSegment = true;
    stopVideoSegment();
  }
  state.videoVad?.pause?.();
  state.videoVad?.destroy?.();
  state.videoVad = null;
  state.videoVadReady = false;
  state.videoManualMode = true;
  if (state.videoStream) state.videoStream.getTracks().forEach((track) => track.stop());
  state.videoStream = null;
  els.videoChatPreview.srcObject = null;
  els.videoTalkButton.disabled = true;
  els.videoTalkButton.textContent = "开始一轮";
  setVideoState("视频已关闭");
  setVideoVad("VAD 未启动");
}

function closeVideoChat() {
  stopVideoChat();
  els.videoChatOverlay.hidden = true;
  document.body.classList.remove("video-chat-open");
}

function setPipelineStatus(card, key) {
  if (!card) return;
  const meta = card.querySelector(".message-meta");
  if (!meta) return;
  const text = pipelineStatusText[key] || "";
  meta.textContent = text;
  meta.hidden = !text;
}

function addMessage(role, content = "", beforeNode = null) {
  const card = document.createElement("article");
  card.className = `message-card ${role}`;
  card.dataset.raw = content;
  const label = role === "user" ? "学生输入" : displayModelName(state.modelName);
  card.innerHTML = `
    <header><span>${label}</span></header>
    <div class="message-content"></div>
    ${role === "assistant" ? '<p class="message-meta" hidden></p>' : ""}
  `;
  setMarkdownContent(card.querySelector(".message-content"), content);
  if (beforeNode) {
    els.messages.insertBefore(card, beforeNode);
  } else {
    els.messages.append(card);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
  return card;
}

function updateAssistant(card, text) {
  card.dataset.raw = text;
  setMarkdownContent(card.querySelector(".message-content"), text);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addAgentSidecarCard(eventData, beforeNode = null) {
  if (!eventData?.content) return null;
  const card = document.createElement("article");
  card.className = "agent-sidecar-card";
  card.dataset.raw = eventData.content;
  const agents = Array.isArray(eventData.agentDispatchedAgents) && eventData.agentDispatchedAgents.length
    ? eventData.agentDispatchedAgents.join(" / ")
    : "多 Agent";
  const status = eventData.agentTimeoutOccurred
    ? "部分超时"
    : eventData.agentDecompositionError
    ? "分解回退"
    : "已完成";
  const runId = eventData.agentRunId ? `Run ${String(eventData.agentRunId).slice(0, 8)}` : "旧链路";
  card.innerHTML = `
    <details>
      <summary>
        <span>旁路旧链路回复</span>
        <small>${escapeHtml(agents)} · ${escapeHtml(status)} · ${escapeHtml(runId)}</small>
      </summary>
      <div class="agent-sidecar-content"></div>
    </details>
  `;
  setMarkdownContent(card.querySelector(".agent-sidecar-content"), eventData.content);
  if (beforeNode) {
    els.messages.insertBefore(card, beforeNode);
  } else {
    els.messages.append(card);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
  return card;
}

function renderEmptyConversation() {
  els.messages.innerHTML = `
    <section class="empty-state">
      <h2>心理咨询助手</h2>
      <div class="sample-list">
        <button type="button" data-prompt="我最近压力很大，晚上总是睡不着，想先把原因说清楚。">焦虑倾诉</button>
        <button type="button" data-prompt="这周一直很低落，做什么都提不起劲。">低落陪伴</button>
        <button type="button" data-prompt="帮我把今晚的复习任务拆成一个具体计划。">普通聊天</button>
      </div>
    </section>
  `;
}

function clearEmpty() {
  els.messages.querySelector(".empty-state")?.remove();
}

function startNewSession() {
  state.sessionId = null;
  state.activeSessionId = null;
  closeVideoChat();
  clearAttachments();
  renderEmptyConversation();
  renderHistory();
  closeMobileSidebar();
  els.messageInput.focus();
}

function setSidebar(open) {
  state.sidebarOpen = open;
  els.studentView.classList.toggle("sidebar-open", open);
  els.studentView.classList.toggle("sidebar-collapsed", !open);
  els.sidebarToggle.setAttribute("aria-label", open ? "收起历史记录" : "展开历史记录");
  els.sidebarToggle.title = open ? "收起历史记录" : "展开历史记录";
}

function openMobileSidebar() {
  els.studentView.classList.add("mobile-sidebar-open");
  els.historyOverlay.hidden = false;
}

function closeMobileSidebar() {
  els.studentView.classList.remove("mobile-sidebar-open");
  els.historyOverlay.hidden = true;
}

async function loadChatSessions(activeId = state.activeSessionId) {
  if (state.isAdmin) return;
  try {
    const response = await api("/api/chat/sessions");
    state.sessions = await response.json();
    state.activeSessionId = activeId;
    renderHistory();
  } catch (error) {
    state.sessions = [];
    renderHistory("历史记录读取失败");
  }
}

function renderHistory(errorText = "") {
  els.historyList.innerHTML = "";
  els.historyEmpty.hidden = state.sessions.length > 0;
  els.historyEmpty.textContent = errorText || "暂无历史记录";
  state.sessions.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item ${item.sessionId === state.activeSessionId ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(item.title || "未命名会话")}</strong>
      <span>${escapeHtml(formatDate(item.updatedAt))}</span>
    `;
    button.addEventListener("click", () => loadConversation(item.sessionId));
    els.historyList.append(button);
  });
}

async function loadConversation(sessionId) {
  if (state.sending) return;
  try {
    const response = await api(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
    const data = await response.json();
    state.sessionId = data.sessionId;
    state.activeSessionId = data.sessionId;
    clearAttachments();
    renderConversationMessages(data.messages || []);
    renderHistory();
    closeMobileSidebar();
    els.messageInput.focus();
  } catch (error) {
    renderHistory("会话读取失败");
  }
}

function renderConversationMessages(messages) {
  els.messages.innerHTML = "";
  if (!messages.length) {
    renderEmptyConversation();
    return;
  }
  messages.forEach((message) => {
    const role = String(message.role || "").toLowerCase() === "user" ? "user" : "assistant";
    addMessage(role, message.content || "");
  });
}

function parseSse(buffer, onEvent) {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() || "";
  for (const block of blocks) {
    const data = block.split("\n").find((line) => line.startsWith("data:"));
    if (data) onEvent(JSON.parse(data.slice(5)));
  }
  return rest;
}

function hasRecordedAudio(files) {
  return files.some(([key, , file]) => key === "audio" && file === state.recordedAudioFile);
}

function visibleMultimodalInput(message, files) {
  return [
    message || "学生上传了多模态内容",
    ...files
      .filter(([key]) => key !== "audio")
      .map(([, label, file]) => `${label}: ${file.name}`)
  ].join("\n");
}

async function sendChat(event) {
  event.preventDefault();
  if (state.sending || state.isAdmin) return;
  if (state.recording) {
    showAttachmentNotice("正在录音，点击麦克风停止后会自动发送");
    return;
  }
  const message = els.messageInput.value.trim();
  const files = selectedFiles();
  if (!message && !files.length) return;

  state.sending = true;
  els.sendButton.disabled = true;
  els.micButton.disabled = true;
  els.messageInput.value = "";
  resizeMessageInput();
  clearEmpty();

  const waitingForTranscript = hasRecordedAudio(files);
  let displayedUserCard = waitingForTranscript ? null : addMessage("user", files.length ? visibleMultimodalInput(message, files) : message);
  const assistant = addMessage("assistant", "");
  setPipelineStatus(assistant, "input");

  try {
    const response = files.length ? await sendMultimodal(message, files) : await sendText(message);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let output = "";
    let streamErrored = false;
    setPipelineStatus(assistant, files.length ? "fusion" : "router");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") {
          const changedSession = state.sessionId !== eventData.sessionId;
          state.sessionId = eventData.sessionId;
          state.activeSessionId = eventData.sessionId;
          if (changedSession) void loadChatSessions(eventData.sessionId);
          if (waitingForTranscript && eventData.displayText && !displayedUserCard) {
            displayedUserCard = addMessage("user", eventData.displayText, assistant);
          }
          setPipelineStatus(assistant, "rag");
        }
        if (eventData.type === "token") {
          output += eventData.content;
          updateAssistant(assistant, output);
          setPipelineStatus(assistant, "stream");
        }
        if (eventData.type === "agent") {
          addAgentSidecarCard(eventData, assistant);
        }
        if (eventData.type === "error") {
          streamErrored = true;
          output = eventData.content || "模型暂时没有返回内容。";
          updateAssistant(assistant, output);
          setPipelineStatus(assistant, "failed");
        }
      });
    }

    if (waitingForTranscript && !displayedUserCard && !streamErrored) {
      displayedUserCard = addMessage("user", message || "语音转录暂时不可用", assistant);
    }
    if (!output) updateAssistant(assistant, "模型暂时没有返回内容。");
    setPipelineStatus(assistant, "mcp");
    setTimeout(() => setPipelineStatus(assistant, "done"), 280);
  } catch (error) {
    if (waitingForTranscript && !displayedUserCard) {
      displayedUserCard = addMessage("user", message || "语音消息发送失败", assistant);
    }
    updateAssistant(assistant, "请求失败，请确认后端和 Ollama 已启动。");
    setPipelineStatus(assistant, "failed");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
    clearAttachments();
    if (state.sessionId) void loadChatSessions(state.sessionId);
    els.messageInput.focus();
  }
}

function resizeMessageInput() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(els.messageInput.scrollHeight, 180)}px`;
}

function sendText(message) {
  return api("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId: state.sessionId, message })
  });
}

function sendMultimodal(message, files) {
  const body = new FormData();
  body.append("message", message || "学生上传了多模态内容，希望获得支持。");
  if (state.sessionId) body.append("sessionId", state.sessionId);
  files.forEach(([key, , file]) => body.append(key, file));
  return api("/api/chat/multimodal/stream", { method: "POST", body });
}

function formatDate(value) {
  return value ? new Date(value).toLocaleString() : "";
}

function riskTone(risk) {
  if (risk === "HIGH" || risk === "FAILED") return "danger";
  if (risk === "MEDIUM" || risk === "PENDING") return "warn";
  if (risk === "LOW" || risk === "SUCCESS") return "ok";
  return "";
}

function statCard(label, value, kind) {
  const node = document.createElement("article");
  node.className = `stat-card ${kind || ""}`;
  node.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
  return node;
}

function renderAdminStats(reports, excelRecords, alerts) {
  els.adminStats.innerHTML = "";
  const high = reports.filter((item) => item.riskLevel === "HIGH").length;
  const medium = reports.filter((item) => item.riskLevel === "MEDIUM").length;
  const mailFailed = alerts.filter((item) => item.status === "FAILED").length;
  els.queueCount.textContent = high;
  els.adminStats.append(
    statCard("报告总数", reports.length),
    statCard("高风险", high, "danger"),
    statCard("需关注", medium, "warn"),
    statCard("邮件失败", mailFailed, mailFailed ? "danger" : "ok"),
    statCard("Excel 写入", excelRecords.length, "ok")
  );
}

function emptyRecord(text) {
  const node = document.createElement("p");
  node.className = "empty-record";
  node.textContent = text;
  return node;
}

function recordButton(title, badge, meta, summary, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "record-card";
  button.innerHTML = `
    <div><strong>${escapeHtml(title)}</strong><span class="${riskTone(badge)}">${escapeHtml(badge || "SKIPPED")}</span></div>
    <small>${escapeHtml(meta || "")}</small>
    <p>${escapeHtml(summary || "无摘要")}</p>
  `;
  button.addEventListener("click", onClick);
  return button;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setMarkdownContent(element, content) {
  element.innerHTML = renderMarkdown(content);
}

function renderMarkdown(value) {
  const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    if (lines[index].trim().startsWith("```")) {
      index += 1;
      const code = [];
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const block = [];
    while (index < lines.length && !lines[index].trim().startsWith("```")) {
      block.push(lines[index]);
      index += 1;
    }
    html.push(renderMarkdownBlocks(block));
  }

  return html.join("").trim();
}

function renderMarkdownBlocks(lines) {
  const html = [];
  let paragraph = [];
  let listType = null;

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  }

  function closeList() {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const quote = line.match(/^\s*>\s?(.*)$/);

    if (!trimmed) {
      flushParagraph();
      closeList();
      return;
    }

    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        html.push(`<${nextType}>`);
        listType = nextType;
      }
      html.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
      return;
    }

    closeList();
    if (quote) {
      flushParagraph();
      html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      return;
    }

    paragraph.push(line);
  });

  flushParagraph();
  closeList();
  return html.join("");
}

function renderInlineMarkdown(value) {
  const codeSpans = [];
  let html = escapeHtml(value).replace(/`([^`]+)`/g, (_, code) => {
    const index = codeSpans.push(`<code>${code}</code>`) - 1;
    return `\u0000CODE${index}\u0000`;
  });

  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+|\/[^\s)]*)\)/g, (_, label, href) => (
    `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`
  ));
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  html = html.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeSpans[Number(index)] || "");
  return html;
}

function renderReportRows(reports) {
  els.adminReportRows.innerHTML = "";
  if (!reports.length) {
    els.adminReportRows.append(emptyRecord("暂无风险记录"));
    return;
  }
  reports.slice(0, 24).forEach((item) => {
    els.adminReportRows.append(recordButton(
      `${item.username} / ${item.emotion}`,
      item.riskLevel,
      `${item.intent} · ${formatDate(item.createdAt)}`,
      item.summary,
      () => item.sessionId ? openConversation(item) : openRecord("报告详情", item)
    ));
  });
}

function renderExcelRows(records) {
  els.excelRows.innerHTML = "";
  if (!records.length) {
    els.excelRows.append(emptyRecord("暂无 Excel 记录"));
    return;
  }
  records.slice(0, 24).forEach((item) => {
    els.excelRows.append(recordButton(
      `#${item.reportId} / ${item.username}`,
      item.excelStatus,
      `${item.emotion} · ${item.riskLevel} · ${formatDate(item.createdAt)}`,
      item.summary || item.content,
      () => openRecord("Excel 写入", item)
    ));
  });
}

function renderEmailRows(records) {
  els.emailRows.innerHTML = "";
  if (!records.length) {
    els.emailRows.append(emptyRecord("暂无预警邮件"));
    return;
  }
  records.slice(0, 24).forEach((item) => {
    els.emailRows.append(recordButton(
      `#${item.reportId} / ${item.recipient}`,
      item.status,
      `${item.riskLevel} · ${item.attempts} 次 · ${formatDate(item.updatedAt)}`,
      item.errorMessage || item.summary,
      () => openRecord("邮件预警", item)
    ));
  });
}

function detailRow(label, value) {
  const row = document.createElement("div");
  row.className = "detail-row";
  row.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "无")}</strong>`;
  return row;
}

function openRecord(title, record) {
  els.detailOverlay.hidden = false;
  els.detailKicker.textContent = "记录详情";
  els.detailTitle.textContent = title;
  els.detailMeta.textContent = formatDate(record.createdAt || record.updatedAt);
  els.detailBody.innerHTML = "";
  Object.entries(record).forEach(([key, value]) => {
    if (value !== null && value !== undefined && typeof value !== "object") {
      els.detailBody.append(detailRow(key, value));
    }
  });
}

async function openConversation(report) {
  els.detailOverlay.hidden = false;
  els.detailKicker.textContent = `${report.username} / ${report.sessionId}`;
  els.detailTitle.textContent = "完整对话";
  els.detailMeta.textContent = "管理员视图";
  els.detailBody.innerHTML = `<p class="empty-record">读取中...</p>`;
  try {
    const response = await api(`/api/admin/conversations/${encodeURIComponent(report.sessionId)}`);
    const data = await response.json();
    els.detailBody.innerHTML = "";
    data.messages.forEach((message) => {
      const card = document.createElement("article");
      card.className = `conversation-card ${message.role.toLowerCase()}`;
      card.innerHTML = `<header><strong>${message.role}</strong><span>${formatDate(message.createdAt)}</span></header><div class="conversation-content"></div>`;
      setMarkdownContent(card.querySelector(".conversation-content"), message.content);
      els.detailBody.append(card);
    });
  } catch (error) {
    els.detailBody.innerHTML = `<p class="empty-record">读取失败</p>`;
  }
}

function closeDetail() {
  els.detailOverlay.hidden = true;
}

async function loadReports() {
  const response = await api("/api/admin/reports");
  return response.json();
}

async function loadExcelRecords() {
  const response = await api("/api/admin/excel-records");
  return response.json();
}

async function loadAlertRecords() {
  const response = await api("/api/admin/alerts");
  return response.json();
}

async function loadAdminData() {
  const [reports, excelRecords, alerts] = await Promise.all([
    loadReports(),
    loadExcelRecords(),
    loadAlertRecords()
  ]);
  state.latestReports = reports;
  renderAdminStats(reports, excelRecords, alerts);
  renderReportRows(reports);
  renderExcelRows(excelRecords);
  renderEmailRows(alerts);
}

async function uploadKnowledge(event) {
  event.preventDefault();
  const file = els.knowledgeFile.files?.[0];
  if (!file) {
    els.knowledgeUploadState.textContent = "请选择文件";
    return;
  }
  const body = new FormData();
  body.append("file", file);
  els.knowledgeUploadState.textContent = "入库中";
  try {
    const response = await api("/api/admin/knowledge/file", { method: "POST", body });
    const data = await response.json();
    els.knowledgeUploadState.textContent = `${data.source} / ${data.chunks} 个片段`;
    els.knowledgeFile.value = "";
  } catch (error) {
    els.knowledgeUploadState.textContent = "入库失败";
  }
}

function showLoggedOut() {
  state.isAdmin = false;
  state.sessions = [];
  state.sessionId = null;
  state.activeSessionId = null;
  els.loginForm.hidden = false;
  els.accountPanel.hidden = true;
  els.studentView.hidden = false;
  els.adminView.hidden = true;
  renderEmptyConversation();
  renderHistory();
}

function isAdmin(profile) {
  return profile.roles?.some((role) => role.authority === "ROLE_ADMIN");
}

async function loadProfile() {
  const response = await api("/api/profile");
  const profile = await response.json();
  state.isAdmin = isAdmin(profile);
  const accountName = state.isAdmin ? (profile.displayName || profile.username) : profile.username;
  els.loginForm.hidden = true;
  els.accountPanel.hidden = false;
  els.activeAccount.textContent = accountName;
  els.activeRole.textContent = state.isAdmin ? "管理员账号" : "学生账号";

  if (state.isAdmin) {
    els.studentView.hidden = true;
    els.adminView.hidden = false;
    await loadAdminData();
  } else {
    els.studentView.hidden = false;
    els.adminView.hidden = true;
    await loadChatSessions();
  }
  els.loginState.textContent = "登录成功";
}

async function loadAgentStatus() {
  const response = await api("/api/agent/status");
  setModel(await response.json());
}

async function login(event) {
  event?.preventDefault();
  state.auth.username = els.username.value.trim();
  state.auth.password = els.password.value;
  try {
    await loadProfile();
    await loadAgentStatus();
  } catch (error) {
    showLoggedOut();
    els.loginState.textContent = "账号或密码错误";
  }
}

els.loginForm.addEventListener("submit", login);
els.switchAccount.addEventListener("click", () => {
  closeVideoChat();
  showLoggedOut();
  els.username.focus();
});
els.sidebarToggle.addEventListener("click", () => setSidebar(!state.sidebarOpen));
els.mobileHistoryToggle.addEventListener("click", openMobileSidebar);
els.historyOverlay.addEventListener("click", closeMobileSidebar);
els.chatForm.addEventListener("submit", sendChat);
els.messageInput.addEventListener("input", resizeMessageInput);
els.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.chatForm.requestSubmit();
  }
});
els.audioInput.addEventListener("change", updateAttachments);
els.micButton.addEventListener("click", toggleRecording);
els.imageInput.addEventListener("change", updateAttachments);
els.videoInput.addEventListener("change", updateAttachments);
els.videoChatButton.addEventListener("click", openVideoChat);
els.startVideoChat.addEventListener("click", startVideoChat);
els.videoTalkButton.addEventListener("click", toggleManualVideoTurn);
els.endVideoChat.addEventListener("click", closeVideoChat);
els.closeVideoChat.addEventListener("click", closeVideoChat);
els.clearAttachments.addEventListener("click", clearAttachments);
els.newSessionButton.addEventListener("click", startNewSession);
els.adminRefresh.addEventListener("click", loadAdminData);
els.knowledgeUploadForm.addEventListener("submit", uploadKnowledge);
els.closeDetail.addEventListener("click", closeDetail);
els.detailOverlay.addEventListener("click", (event) => {
  if (event.target === els.detailOverlay) closeDetail();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.detailOverlay.hidden) closeDetail();
  if (event.key === "Escape" && !els.videoChatOverlay.hidden) closeVideoChat();
  if (event.key === "Escape") closeMobileSidebar();
});
document.addEventListener("click", (event) => {
  const prompt = event.target.closest("[data-prompt]");
  if (prompt && !state.isAdmin) {
    els.messageInput.value = prompt.dataset.prompt;
    resizeMessageInput();
    els.messageInput.focus();
  }
});

setSidebar(state.sidebarOpen);
renderEmptyConversation();
renderHistory();
resizeMessageInput();
login();
