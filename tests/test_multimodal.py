from app.core.config import Settings
from app.models.enums import EmotionLabel, RiskLevel
import pytest

from app.services.multimodal import (
    BROWSER_RECORDING_MOCK_ERROR_MESSAGE,
    MAX_AUDIO_BYTES,
    POSTER_PP_ERROR_MESSAGE,
    TRANSCRIPTION_ERROR_MESSAGE,
    MediaPipeClient,
    MultimodalFusionService,
    MultimodalInputService,
    MultimodalSignal,
    PosterPPAnalysisError,
    WhisperClient,
    WhisperTranscriptionError,
)
from app.api.chat import stream_multimodal, stream_video_chat


class LargeUpload:
    filename = "large-video.mp4"
    content_type = "video/mp4"

    async def read(self, size: int = -1) -> bytes:
        return b"x" * size


class NamedUpload:
    content_type = "audio/webm"

    def __init__(self, filename: str) -> None:
        self.filename = filename

    async def read(self, size: int = -1) -> bytes:
        return b"audio"


class ImageUpload:
    filename = "frame.jpg"
    content_type = "image/jpeg"

    async def read(self, size: int = -1) -> bytes:
        return b"image"


class LargeAudioUpload:
    filename = "risk-audio.webm"
    content_type = "audio/webm"

    def __init__(self) -> None:
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return b"x" * size


class BrokenWhisperClient:
    async def transcribe(self, file) -> str:
        raise WhisperTranscriptionError("audio failed")


class TranscriptWhisperClient:
    async def transcribe(self, file) -> str:
        return "我最近有些焦虑，睡眠也不太好。"


class BrokenMediaPipeClient:
    async def analyze(self, file, modality: str):
        raise RuntimeError(f"{modality} failed")


class KeywordAssessmentService:
    async def assess(self, text: str):
        from app.services.assessment import PsychologyAssessment

        if "低落" in text:
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.2, RiskLevel.MEDIUM, 0.82, "low")
        if "焦虑" in text:
            return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.78, "anxious")
        return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.7, "normal")


def test_multimodal_fusion_uses_java_thresholds() -> None:
    service = MultimodalFusionService(Settings(ai_provider="mock"))
    analysis = service.fuse(
        "我很难受",
        [
            MultimodalSignal("text", EmotionLabel.ANXIETY, 2.2, 0.78, "text"),
            MultimodalSignal("visual", EmotionLabel.HIGH_RISK, 4.0, 0.74, "visual"),
        ],
    )
    assert analysis.fused_assessment.risk == RiskLevel.HIGH
    assert analysis.fused_assessment.emotion == EmotionLabel.HIGH_RISK


def test_high_risk_visual_signal_forces_high_risk() -> None:
    service = MultimodalFusionService(Settings(ai_provider="mock"))
    analysis = service.fuse(
        "",
        [MultimodalSignal("visual", EmotionLabel.HIGH_RISK, 1.0, 0.8, "visual")],
    )
    assert analysis.fused_assessment.risk == RiskLevel.HIGH
    assert analysis.fused_assessment.emotion == EmotionLabel.HIGH_RISK


def test_mediapipe_json_accepts_visual_aliases() -> None:
    client = MediaPipeClient(Settings(ai_provider="mock"))
    signal = client._from_mediapipe_json(
        {"visualEmotion": "SAD", "visualScore": 3.2, "confidence": 0.8, "features": {"mouth": "down"}}
    )
    assert signal.modality == "visual"
    assert signal.emotion == EmotionLabel.DEPRESSED
    assert signal.score == 3.2
    assert "features=" in signal.evidence


@pytest.mark.asyncio
async def test_visual_upload_over_8mb_returns_unsupported_signal() -> None:
    client = MediaPipeClient(Settings(ai_provider="mock"))
    signal = await client.analyze(LargeUpload(), "video")  # type: ignore[arg-type]
    assert signal.modality == "visual"
    assert signal.emotion == EmotionLabel.NORMAL
    assert signal.score == 0.0
    assert "超过 8MB" in signal.evidence
    assert "MEDIAPIPE_MODE=http" in signal.evidence


@pytest.mark.asyncio
async def test_whisper_mock_matches_java_filename_rules() -> None:
    client = WhisperClient(Settings(ai_provider="mock"))
    assert await client.transcribe(NamedUpload("risk-note.webm")) == "语音转写提示：我感觉自己快撑不下去了。"
    assert await client.transcribe(NamedUpload("sad-note.webm")) == "语音转写提示：我最近情绪很低落。"
    assert await client.transcribe(NamedUpload("anxious-note.webm")) == "语音转写提示：我最近有些焦虑，睡眠也不太好。"
    assert await client.transcribe(NamedUpload("voice.webm")) == "语音转写提示：学生上传了一段语音，希望继续心理支持对话。"


@pytest.mark.asyncio
async def test_whisper_mock_rejects_browser_recordings() -> None:
    client = WhisperClient(Settings(ai_provider="mock"))

    with pytest.raises(WhisperTranscriptionError, match="浏览器录音无法生成真实转录"):
        await client.transcribe(NamedUpload("mic-recording-1781489623647.webm"))

    with pytest.raises(WhisperTranscriptionError, match="浏览器录音无法生成真实转录"):
        await client.transcribe(NamedUpload("video-turn-1781489623647.webm"))


@pytest.mark.asyncio
async def test_whisper_openai_without_key_raises_configuration_error() -> None:
    client = WhisperClient(Settings(ai_provider="mock", whisper_mode="openai"))

    with pytest.raises(WhisperTranscriptionError):
        await client.transcribe(NamedUpload("crisis-note.webm"))


@pytest.mark.asyncio
async def test_whisper_openai_failure_raises_instead_of_mocking(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingTranscriptions:
        async def create(self, **kwargs):
            raise RuntimeError("provider down")

    class FailingAudio:
        transcriptions = FailingTranscriptions()

    class FailingOpenAI:
        audio = FailingAudio()

        def __init__(self, **kwargs) -> None:
            pass

    monkeypatch.setattr("app.services.multimodal.AsyncOpenAI", FailingOpenAI)
    client = WhisperClient(Settings(ai_provider="mock", whisper_mode="openai", whisper_api_key="test"))

    with pytest.raises(WhisperTranscriptionError):
        await client.transcribe(NamedUpload("crisis-note.webm"))


@pytest.mark.asyncio
async def test_whisper_openai_large_audio_reads_limit_plus_one_and_raises() -> None:
    client = WhisperClient(Settings(ai_provider="mock", whisper_mode="openai", whisper_api_key="test"))
    upload = LargeAudioUpload()

    with pytest.raises(WhisperTranscriptionError):
        await client.transcribe(upload)  # type: ignore[arg-type]

    assert upload.read_sizes == [MAX_AUDIO_BYTES + 1]


@pytest.mark.asyncio
async def test_multimodal_audio_failure_is_not_ignored() -> None:
    service = MultimodalInputService(
        BrokenWhisperClient(),
        BrokenMediaPipeClient(),
        KeywordAssessmentService(),
        MultimodalFusionService(Settings(ai_provider="mock")),
    )

    with pytest.raises(WhisperTranscriptionError):
        await service.analyze("我最近焦虑", NamedUpload("audio.webm"), LargeUpload(), LargeUpload())


@pytest.mark.asyncio
async def test_multimodal_ignores_failed_visual_modalities() -> None:
    service = MultimodalInputService(
        TranscriptWhisperClient(),
        BrokenMediaPipeClient(),
        KeywordAssessmentService(),
        MultimodalFusionService(Settings(ai_provider="mock")),
    )
    analysis = await service.analyze("我最近焦虑", None, LargeUpload(), LargeUpload())
    assert [signal.modality for signal in analysis.signals] == ["text"]
    assert analysis.fused_assessment.risk == RiskLevel.LOW


@pytest.mark.asyncio
async def test_multimodal_audio_display_text_uses_whisper_transcript() -> None:
    service = MultimodalInputService(
        TranscriptWhisperClient(),
        BrokenMediaPipeClient(),
        KeywordAssessmentService(),
        MultimodalFusionService(Settings(ai_provider="mock")),
    )

    analysis = await service.analyze("学生上传了多模态内容，希望获得支持。", NamedUpload("audio.webm"), None, None)

    assert analysis.display_text == "我最近有些焦虑，睡眠也不太好。"
    assert "Whisper 转写后情绪分析：我最近有些焦虑，睡眠也不太好。" in analysis.model_text


@pytest.mark.asyncio
async def test_multimodal_stream_returns_sse_error_for_transcription_failure() -> None:
    class FailingMultimodalService:
        async def analyze(self, message, audio, image, video):
            raise WhisperTranscriptionError("audio failed")

    response = await stream_multimodal(
        user=object(),  # type: ignore[arg-type]
        db=object(),  # type: ignore[arg-type]
        multimodal_service=FailingMultimodalService(),  # type: ignore[arg-type]
        message="学生上传了多模态内容，希望获得支持。",
        audio=NamedUpload("voice.webm"),  # type: ignore[arg-type]
    )
    body = "".join([chunk async for chunk in response.body_iterator])

    assert response.media_type == "text/event-stream"
    assert '"type":"error"' in body
    assert TRANSCRIPTION_ERROR_MESSAGE in body


@pytest.mark.asyncio
async def test_multimodal_stream_returns_specific_mock_recording_error() -> None:
    response = await stream_multimodal(
        user=object(),  # type: ignore[arg-type]
        db=object(),  # type: ignore[arg-type]
        multimodal_service=MultimodalInputService(
            WhisperClient(Settings(ai_provider="mock")),
            BrokenMediaPipeClient(),
            KeywordAssessmentService(),
            MultimodalFusionService(Settings(ai_provider="mock")),
        ),
        message="学生上传了多模态内容，希望获得支持。",
        audio=NamedUpload("mic-recording-1781489623647.webm"),  # type: ignore[arg-type]
    )
    body = "".join([chunk async for chunk in response.body_iterator])

    assert '"type":"error"' in body
    assert BROWSER_RECORDING_MOCK_ERROR_MESSAGE in body


@pytest.mark.asyncio
async def test_video_chat_combines_whisper_and_poster_pp_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "visualEmotion": "DEPRESSED",
                "visualScore": 3.72,
                "confidence": 0.9,
                "evidence": "POSTER++ RAF-DB 表情分类。",
                "features": {"rawEmotion": "sad"},
            }

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url, files, data):
            captured["url"] = url
            captured["data"] = data
            captured["filename"] = files["file"][0]
            return FakeResponse()

    monkeypatch.setattr("app.services.multimodal.httpx.AsyncClient", FakeAsyncClient)
    service = MultimodalInputService(
        TranscriptWhisperClient(),
        BrokenMediaPipeClient(),
        KeywordAssessmentService(),
        MultimodalFusionService(Settings(ai_provider="mock", POSTER_PP_URL="http://poster")),
    )

    analysis = await service.analyze_video_chat(
        "补充文字",
        NamedUpload("voice.webm"),
        ImageUpload(),  # type: ignore[arg-type]
        {"preprocessMode": "mediapipe_affine_align", "fallback": "false"},
    )

    assert captured["url"] == "http://poster/analyze-frame"
    assert captured["filename"] == "frame.jpg"
    assert captured["data"]["preprocessMode"] == "mediapipe_affine_align"
    assert analysis.display_text == "我最近有些焦虑，睡眠也不太好。"
    assert [signal.modality for signal in analysis.signals] == ["text", "audio", "visual"]
    assert analysis.signals[-1].emotion == EmotionLabel.DEPRESSED
    assert "POSTER++ RAF-DB 表情分类" in analysis.signals[-1].evidence


@pytest.mark.asyncio
async def test_video_stream_returns_sse_error_for_poster_pp_failure() -> None:
    class FailingVideoService:
        async def analyze_video_chat(self, message, audio, frame, poster_metadata):
            raise PosterPPAnalysisError("poster failed")

    response = await stream_video_chat(
        user=object(),  # type: ignore[arg-type]
        db=object(),  # type: ignore[arg-type]
        multimodal_service=FailingVideoService(),  # type: ignore[arg-type]
        audio=NamedUpload("voice.webm"),  # type: ignore[arg-type]
        frame=ImageUpload(),  # type: ignore[arg-type]
        message="hello",
    )
    body = "".join([chunk async for chunk in response.body_iterator])

    assert response.media_type == "text/event-stream"
    assert '"type":"error"' in body
    assert POSTER_PP_ERROR_MESSAGE in body
