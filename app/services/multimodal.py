import json
import logging
import math
from io import BytesIO
from dataclasses import dataclass

import httpx
from fastapi import UploadFile
from openai import AsyncOpenAI
from PIL import Image

from app.core.config import Settings
from app.models.enums import EmotionLabel, RiskLevel
from app.services.assessment import PsychologicalAssessmentService, PsychologyAssessment

MAX_VISUAL_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
TRANSCRIPTION_ERROR_MESSAGE = "语音转写失败，请检查 Whisper 配置或稍后重试。"
BROWSER_RECORDING_MOCK_ERROR_MESSAGE = (
    "当前 Whisper 仍为 mock 模式，浏览器录音无法生成真实转录；"
    "请配置 WHISPER_MODE=openai 和 OPENAI_API_KEY 后重建 app。"
)
POSTER_PP_ERROR_MESSAGE = "POSTER++ 视频表情分析失败，请确认模型服务已启动并稍后重试。"

logger = logging.getLogger(__name__)


class WhisperTranscriptionError(RuntimeError):
    pass


class PosterPPAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class MultimodalSignal:
    modality: str
    emotion: EmotionLabel
    score: float
    confidence: float
    evidence: str


@dataclass(frozen=True)
class MultimodalAnalysis:
    model_text: str
    signals: list[MultimodalSignal]
    fused_assessment: PsychologyAssessment
    summary: str
    display_text: str

    def emotion_tags_json(self) -> str:
        return json.dumps(
            [
                {
                    "modality": signal.modality,
                    "emotion": signal.emotion.value,
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "evidence": signal.evidence,
                }
                for signal in self.signals
            ],
            ensure_ascii=False,
        )


class WhisperClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def transcribe(self, file: UploadFile) -> str:
        mode = self.settings.whisper_mode.lower().strip()
        if mode == "mock":
            if self.is_browser_recording(file):
                raise WhisperTranscriptionError(BROWSER_RECORDING_MOCK_ERROR_MESSAGE)
            return self.mock_transcript(file)
        if mode != "openai":
            raise WhisperTranscriptionError(f"Unsupported WHISPER_MODE: {self.settings.whisper_mode}")
        api_key = self.settings.whisper_api_key or self.settings.openai_api_key
        if not api_key:
            raise WhisperTranscriptionError("WHISPER_MODE=openai requires WHISPER_API_KEY or OPENAI_API_KEY.")
        try:
            data = await file.read(MAX_AUDIO_BYTES + 1)
            if len(data) > MAX_AUDIO_BYTES:
                raise WhisperTranscriptionError("Audio file exceeds Whisper upload limit.")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.settings.whisper_base_url,
                timeout=60.0,
                max_retries=0,
            )
            response = await client.audio.transcriptions.create(
                model=self.settings.whisper_model,
                file=(
                    self.safe_upload_filename(file),
                    data,
                    file.content_type or "application/octet-stream",
                ),
            )
            text = response.text.strip()
            if not text:
                raise WhisperTranscriptionError("Whisper returned empty transcript.")
            return text
        except WhisperTranscriptionError:
            raise
        except Exception as exc:
            logger.warning(
                "Whisper transcription request failed: error_type=%s status_code=%s body=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                self._safe_error_body(getattr(exc, "body", None)),
            )
            raise WhisperTranscriptionError("OpenAI Whisper request failed.") from exc

    def safe_upload_filename(self, file: UploadFile | None) -> str:
        filename = (getattr(file, "filename", "") or "").lower()
        content_type = (getattr(file, "content_type", "") or "").lower()
        if content_type in {"audio/webm", "video/webm"} or filename.endswith(".webm"):
            return "audio.webm"
        if content_type in {"audio/mpeg", "audio/mp3"} or filename.endswith(".mp3"):
            return "audio.mp3"
        if content_type in {"audio/wav", "audio/x-wav", "audio/wave"} or filename.endswith(".wav"):
            return "audio.wav"
        if content_type in {"audio/mp4", "audio/x-m4a"} or filename.endswith((".m4a", ".mp4")):
            return "audio.m4a"
        if content_type.startswith("audio/"):
            return "audio.bin"
        return "audio.webm"

    def _safe_error_body(self, body: object) -> str:
        if body is None:
            return ""
        text = str(body).replace(self.settings.whisper_api_key, "[redacted]")
        if self.settings.openai_api_key:
            text = text.replace(self.settings.openai_api_key, "[redacted]")
        return text[:500]

    def mock_transcript(self, file: UploadFile | None) -> str:
        filename = (getattr(file, "filename", "") or "").lower()
        if any(word in filename for word in ("risk", "crisis", "崩溃")):
            return "语音转写提示：我感觉自己快撑不下去了。"
        if any(word in filename for word in ("sad", "depress", "低落")):
            return "语音转写提示：我最近情绪很低落。"
        if any(word in filename for word in ("anxious", "stress", "焦虑")):
            return "语音转写提示：我最近有些焦虑，睡眠也不太好。"
        return "语音转写提示：学生上传了一段语音，希望继续心理支持对话。"

    def is_browser_recording(self, file: UploadFile | None) -> bool:
        filename = (getattr(file, "filename", "") or "").lower()
        return filename.startswith(("mic-recording-", "video-turn-"))


class MediaPipeClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def analyze(self, file: UploadFile, modality: str) -> MultimodalSignal:
        filename = file.filename or "upload"
        content_type = file.content_type or "unknown"
        try:
            raw = await file.read(MAX_VISUAL_BYTES + 1)
        except Exception:
            return self._unsupported(filename, content_type)
        if len(raw) > MAX_VISUAL_BYTES:
            return self._unsupported(filename, content_type, "文件超过 8MB")
        if self.settings.mediapipe_mode == "http":
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.post(
                        self.settings.mediapipe_url,
                        files={"file": (filename, raw, file.content_type or "application/octet-stream")},
                    )
                    response.raise_for_status()
                    payload = response.json()
                return self._from_mediapipe_json(payload)
            except Exception:
                pass
        return self._analyze_image_bytes(raw, filename, content_type)

    def _from_mediapipe_json(self, payload: dict) -> MultimodalSignal:
        raw_emotion = payload.get("emotion") or payload.get("visualEmotion") or "NORMAL"
        emotion = self._parse_emotion(str(raw_emotion))
        score = float(payload.get("score", payload.get("visualScore", self._score_for_emotion(emotion))))
        confidence = float(payload.get("confidence", 0.82))
        evidence = str(payload.get("evidence", "MediaPipe Face Mesh 服务返回视觉情绪结果。"))
        if "features" in payload:
            evidence += f" features={payload['features']}"
        return MultimodalSignal("visual", emotion, score, confidence, evidence)

    def _analyze_image_bytes(self, raw: bytes, filename: str, content_type: str) -> MultimodalSignal:
        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            return self._unsupported(filename, content_type)
        features = self._extract_features(image)
        brow_score = 1.5 if features["brow_shadow"] > 0.34 else 0.8 if features["brow_shadow"] > 0.26 else 0.0
        eye_score = 1.0 if features["eye_darkness"] > 0.42 else 0.5 if features["eye_darkness"] > 0.32 else 0.0
        mouth_score = 1.0 if features["mouth_down"] > 0.16 else 0.5 if features["mouth_down"] > 0.08 else 0.0
        tension_score = 1.5 if features["muscle_tension"] > 0.38 else 0.8 if features["muscle_tension"] > 0.28 else 0.0
        low_energy_score = 0.8 if features["face_brightness"] < 0.36 and features["saturation"] < 0.32 else 0.0
        score = self._clamp(brow_score + eye_score + mouth_score + tension_score + low_energy_score, 0.0, 4.5)
        emotion = self._emotion_from_score(score)
        confidence = self._clamp(0.56 + features["face_coverage"] * 0.28 + features["symmetry"] * 0.12, 0.55, 0.9)
        evidence = (
            "MediaPipe 本地视觉分析："
            f"faceDetected={features['face_detected']}, faceCoverage={features['face_coverage']:.2f}, "
            f"browShadow={features['brow_shadow']:.2f}, eyeDarkness={features['eye_darkness']:.2f}, "
            f"mouthDown={features['mouth_down']:.2f}, muscleTension={features['muscle_tension']:.2f}, "
            f"brightness={features['face_brightness']:.2f}, saturation={features['saturation']:.2f}, "
            f"visualScore={score:.2f}。"
        )
        return MultimodalSignal("visual", emotion, score, confidence, evidence)

    def _unsupported(
        self, filename: str, content_type: str, reason: str = "无法作为图片解码"
    ) -> MultimodalSignal:
        return MultimodalSignal(
            "visual",
            EmotionLabel.NORMAL,
            0.0,
            0.45,
            f"视觉分析：当前文件 {filename} ({content_type}) {reason}；如需视频逐帧 Face Mesh，请启用 MEDIAPIPE_MODE=http 对接外部 MediaPipe 服务。",
        )

    def _extract_features(self, image: Image.Image) -> dict[str, float | bool]:
        width, height = image.size
        face = self._detect_face_region(image)
        face_stats = self._stats(image, face)
        upper = self._stats(image, self._slice(face, 0.18, 0.45, 0.08, 0.92))
        brow = self._stats(image, self._slice(face, 0.20, 0.34, 0.16, 0.84))
        eyes = self._stats(image, self._slice(face, 0.32, 0.50, 0.10, 0.90))
        mouth_center = self._stats(image, self._slice(face, 0.64, 0.82, 0.38, 0.62))
        mouth_corners = self._merge_stats(
            self._stats(image, self._slice(face, 0.64, 0.84, 0.18, 0.38)),
            self._stats(image, self._slice(face, 0.64, 0.84, 0.62, 0.82)),
        )
        left = self._stats(image, self._slice(face, 0.0, 1.0, 0.0, 0.5))["brightness"]
        right = self._stats(image, self._slice(face, 0.0, 1.0, 0.5, 1.0))["brightness"]
        area = face[2] * face[3]
        return {
            "face_detected": area < width * height * 0.92,
            "face_coverage": area / max(1, width * height),
            "symmetry": self._clamp(1.0 - abs(left - right) * 2.0, 0.0, 1.0),
            "face_brightness": face_stats["brightness"],
            "saturation": face_stats["saturation"],
            "brow_shadow": self._clamp((face_stats["brightness"] - brow["brightness"]) + brow["dark_ratio"] * 0.45, 0.0, 1.0),
            "eye_darkness": self._clamp(eyes["dark_ratio"] * 0.75 + (upper["contrast"] - face_stats["contrast"]) * 0.4, 0.0, 1.0),
            "mouth_down": self._clamp((mouth_corners["dark_ratio"] - mouth_center["dark_ratio"]) * 0.75 + (mouth_center["brightness"] - mouth_corners["brightness"]) * 0.45, 0.0, 1.0),
            "muscle_tension": self._clamp(face_stats["contrast"] * 0.7 + abs(left - right) * 0.7, 0.0, 1.0),
        }

    def _detect_face_region(self, image: Image.Image) -> tuple[int, int, int, int]:
        width, height = image.size
        min_x, min_y, max_x, max_y = width, height, -1, -1
        skin_pixels = 0
        step = max(1, min(width, height) // 240)
        pixels = image.load()
        for y in range(0, height, step):
            for x in range(0, width, step):
                if self._looks_like_skin(pixels[x, y]):
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    max_x, max_y = max(max_x, x), max(max_y, y)
                    skin_pixels += 1
        sampled = max(1, (width // step) * (height // step))
        if skin_pixels < sampled * 0.015 or max_x <= min_x or max_y <= min_y:
            return self._bounded((int(width * 0.20), int(height * 0.12), int(width * 0.60), int(height * 0.72)), width, height)
        pad_x = (max_x - min_x) // 5
        pad_y = (max_y - min_y) // 4
        return self._bounded((min_x - pad_x, min_y - pad_y, (max_x - min_x) + pad_x * 2, (max_y - min_y) + pad_y * 2), width, height)

    def _stats(self, image: Image.Image, rect: tuple[int, int, int, int]) -> dict[str, float]:
        x, y, width, height = self._bounded(rect, *image.size)
        step = max(1, min(width, height) // 80)
        brightness = saturation = squared = 0.0
        dark = count = 0
        pixels = image.load()
        for yy in range(y, y + height, step):
            for xx in range(x, x + width, step):
                r, g, b = pixels[xx, yy]
                value = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                sat = self._saturation(r, g, b)
                brightness += value
                saturation += sat
                squared += value * value
                dark += 1 if value < 0.30 else 0
                count += 1
        if count == 0:
            return {"brightness": 0.5, "saturation": 0.0, "contrast": 0.0, "dark_ratio": 0.0}
        mean = brightness / count
        variance = max(0.0, squared / count - mean * mean)
        return {
            "brightness": mean,
            "saturation": saturation / count,
            "contrast": math.sqrt(variance),
            "dark_ratio": dark / count,
        }

    def _slice(self, rect: tuple[int, int, int, int], top: float, bottom: float, left: float, right: float) -> tuple[int, int, int, int]:
        x, y, width, height = rect
        return (
            x + round(width * left),
            y + round(height * top),
            max(1, round(width * (right - left))),
            max(1, round(height * (bottom - top))),
        )

    def _bounded(self, rect: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x, y, width, height = rect
        bounded_x = max(0, min(x, image_width - 1))
        bounded_y = max(0, min(y, image_height - 1))
        return (
            bounded_x,
            bounded_y,
            max(1, min(width, image_width - bounded_x)),
            max(1, min(height, image_height - bounded_y)),
        )

    def _looks_like_skin(self, rgb: tuple[int, int, int]) -> bool:
        r, g, b = rgb
        return r > 70 and g > 45 and b > 35 and r > g and g >= b * 0.72 and max(rgb) - min(rgb) > 12

    def _merge_stats(self, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        return {key: (left[key] + right[key]) / 2.0 for key in left}

    def _saturation(self, r: int, g: int, b: int) -> float:
        maximum = max(r, g, b) / 255.0
        minimum = min(r, g, b) / 255.0
        return 0.0 if maximum == 0.0 else (maximum - minimum) / maximum

    def _parse_emotion(self, emotion: str) -> EmotionLabel:
        normalized = emotion.strip().upper()
        if normalized in {"高风险", "HIGH", "HIGH_RISK"}:
            return EmotionLabel.HIGH_RISK
        if normalized in {"低落", "DEPRESSED", "SAD"}:
            return EmotionLabel.DEPRESSED
        if normalized in {"焦虑", "ANXIETY", "ANXIOUS"}:
            return EmotionLabel.ANXIETY
        return EmotionLabel.NORMAL

    def _score_for_emotion(self, emotion: EmotionLabel) -> float:
        return {
            EmotionLabel.HIGH_RISK: 4.0,
            EmotionLabel.DEPRESSED: 3.0,
            EmotionLabel.ANXIETY: 2.0,
            EmotionLabel.NORMAL: 0.0,
        }[emotion]

    def _emotion_from_score(self, score: float) -> EmotionLabel:
        if score >= 4.0:
            return EmotionLabel.HIGH_RISK
        if score >= 3.0:
            return EmotionLabel.DEPRESSED
        if score >= 2.0:
            return EmotionLabel.ANXIETY
        return EmotionLabel.NORMAL

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


class MultimodalFusionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fuse(self, user_text: str, signals: list[MultimodalSignal], display_text: str | None = None) -> MultimodalAnalysis:
        visible_text = (display_text or user_text or "学生上传了多模态内容，希望获得支持。").strip()
        if not signals:
            assessment = PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.6, "No multimodal signal.")
            return MultimodalAnalysis(user_text, [], assessment, assessment.summary, visible_text)
        weights = {
            "text": self.settings.multimodal_text_weight,
            "audio": self.settings.multimodal_audio_weight,
            "visual": self.settings.multimodal_visual_weight,
        }
        fused_score = sum(signal.score * max(0.0, weights.get(signal.modality, 0.1)) for signal in signals)
        confidence = sum(signal.confidence * max(0.0, weights.get(signal.modality, 0.1)) for signal in signals)
        strongest = max(signals, key=lambda item: item.score * item.confidence)
        emotion = strongest.emotion
        risk = RiskLevel.HIGH if fused_score >= 2.0 else RiskLevel.MEDIUM if fused_score >= 1.0 else RiskLevel.LOW
        if any(signal.emotion == EmotionLabel.HIGH_RISK and signal.confidence >= 0.75 for signal in signals):
            risk = RiskLevel.HIGH
            emotion = EmotionLabel.HIGH_RISK
            fused_score = max(fused_score, 4.0)
        summary = (
            "多模态融合："
            + "，".join(f"{signal.modality}={signal.emotion.value}({signal.score:.1f})" for signal in signals)
            + f"，加权分数={fused_score:.2f}，风险={risk.value}。"
        )
        assessment = PsychologyAssessment(
            emotion,
            fused_score,
            risk,
            max(0.0, min(1.0, confidence)),
            summary,
        )
        model_text = "\n".join(
            [(user_text or "学生上传了多模态内容，希望获得支持。").strip(), "【多模态后台分析】", summary, self._signal_text(signals)]
        ).strip()
        return MultimodalAnalysis(model_text, signals, assessment, summary, visible_text)

    def _signal_text(self, signals: list[MultimodalSignal]) -> str:
        return "\n".join(f"{signal.modality}: {signal.evidence}" for signal in signals)


class MultimodalInputService:
    def __init__(
        self,
        whisper_client: WhisperClient,
        mediapipe_client: MediaPipeClient,
        assessment_service: PsychologicalAssessmentService,
        fusion_service: MultimodalFusionService,
    ) -> None:
        self.whisper_client = whisper_client
        self.mediapipe_client = mediapipe_client
        self.assessment_service = assessment_service
        self.fusion_service = fusion_service
        self.settings = fusion_service.settings

    async def analyze(
        self,
        message: str,
        audio: UploadFile | None,
        image: UploadFile | None,
        video: UploadFile | None,
    ) -> MultimodalAnalysis:
        user_text = (message or "").strip()
        audio_transcript = ""
        signals: list[MultimodalSignal] = []
        if user_text:
            assessment = await self.assessment_service.assess(user_text)
            signals.append(
                MultimodalSignal(
                    "text",
                    assessment.emotion,
                    assessment.emotion_score,
                    assessment.confidence,
                    "文本情绪模型：" + assessment.summary,
                )
            )
        if audio:
            transcript = await self.whisper_client.transcribe(audio)
            audio_transcript = transcript.strip()
            assessment = await self.assessment_service.assess(transcript)
            signals.append(
                MultimodalSignal(
                    "audio",
                    assessment.emotion,
                    assessment.emotion_score,
                    min(0.9, assessment.confidence),
                    "Whisper 转写后情绪分析：" + transcript,
                )
            )
        if image:
            try:
                signals.append(await self.mediapipe_client.analyze(image, "image"))
            except Exception:
                pass
        if video:
            try:
                signals.append(await self.mediapipe_client.analyze(video, "video"))
            except Exception:
                pass
        display_text = audio_transcript or user_text or "学生上传了多模态内容，希望获得支持。"
        return self.fusion_service.fuse(user_text, signals, display_text)

    async def analyze_video_chat(
        self,
        message: str,
        audio: UploadFile,
        frame: UploadFile,
        poster_metadata: dict[str, str | int | None] | None = None,
    ) -> MultimodalAnalysis:
        user_text = (message or "").strip()
        transcript = await self.whisper_client.transcribe(audio)
        signals: list[MultimodalSignal] = []
        if user_text:
            assessment = await self.assessment_service.assess(user_text)
            signals.append(
                MultimodalSignal(
                    "text",
                    assessment.emotion,
                    assessment.emotion_score,
                    assessment.confidence,
                    "文本情绪模型：" + assessment.summary,
                )
            )
        audio_assessment = await self.assessment_service.assess(transcript)
        signals.append(
            MultimodalSignal(
                "audio",
                audio_assessment.emotion,
                audio_assessment.emotion_score,
                min(0.9, audio_assessment.confidence),
                "Whisper 转写后情绪分析：" + transcript,
            )
        )
        signals.append(await self.analyze_poster_pp_frame(frame, poster_metadata or {}))
        return self.fusion_service.fuse(user_text or transcript, signals, transcript.strip())

    async def analyze_poster_pp_frame(
        self,
        frame: UploadFile,
        metadata: dict[str, str | int | None],
    ) -> MultimodalSignal:
        filename = frame.filename or "video-frame.jpg"
        content_type = frame.content_type or "image/jpeg"
        try:
            raw = await frame.read(MAX_VISUAL_BYTES + 1)
            if not raw:
                raise PosterPPAnalysisError("Video frame is empty.")
            if len(raw) > MAX_VISUAL_BYTES:
                raise PosterPPAnalysisError("Video frame exceeds size limit.")
            data: dict[str, str | tuple[str, bytes, str]] = {
                "file": (filename, raw, content_type),
            }
            form_data = {
                key: str(value)
                for key, value in metadata.items()
                if value not in (None, "")
            }
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{self.settings.poster_pp_url.rstrip('/')}/analyze-frame",
                    files=data,
                    data=form_data,
                )
                response.raise_for_status()
                payload = response.json()
            return self._poster_pp_signal(payload)
        except PosterPPAnalysisError:
            raise
        except Exception as exc:
            raise PosterPPAnalysisError("POSTER++ analysis failed.") from exc

    def _poster_pp_signal(self, payload: dict) -> MultimodalSignal:
        raw_emotion = payload.get("visualEmotion") or payload.get("emotion") or "NORMAL"
        emotion = MediaPipeClient(self.settings)._parse_emotion(str(raw_emotion))
        score = float(payload.get("visualScore", payload.get("score", 0.0)))
        confidence = float(payload.get("confidence", 0.0))
        evidence = str(payload.get("evidence", "POSTER++ 视频表情分析完成。"))
        features = payload.get("features")
        if features:
            evidence += f" features={features}"
        return MultimodalSignal("visual", emotion, score, confidence, evidence)
