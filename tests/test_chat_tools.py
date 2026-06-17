import asyncio
import json

import pytest

import app.services.chat as chat_module
from app.agent.orchestration.lead import LeadAgentResult
from app.core.config import Settings
from app.models.enums import EmotionLabel, IntentType, MessageRole, RiskLevel, ToolStatus
from app.schemas.api import ChatRequest
from app.services.ai import AiMessage
from app.services.chat import ChatService, PreparedConversation
from app.services.knowledge import AgenticRagResult
from app.services.assessment import PsychologyAssessment
from app.services.multimodal import MultimodalAnalysis, MultimodalSignal


class FailingToolService:
    async def handle(self, session, report_id: int) -> None:
        raise RuntimeError(f"tool failed for {report_id}")


class FakeReport:
    excel_status = ToolStatus.PENDING
    tool_error = None


class FakeSession:
    def __init__(self, report: FakeReport) -> None:
        self.report = report
        self.committed = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def scalar(self, statement):
        return self.report

    async def commit(self) -> None:
        self.committed = True


class FakeUser:
    id = 1
    username = "student"
    display_name = "Student Display"


class FakeChatSession:
    id = 1
    public_id = "session-1"


class EmptyAiClient:
    async def stream(self, messages):
        if False:
            yield ""


class SlowAiClient:
    async def stream(self, messages):
        await asyncio.sleep(1)
        yield "late"


class OneTokenAiClient:
    async def stream(self, messages):
        yield "ok"


class MultiChunkAiClient:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.prompts: list[str] = []

    async def stream(self, messages):
        self.prompts.append("\n".join(message.content for message in messages))
        for chunk in self.chunks:
            yield chunk


class FailingStreamAiClient:
    async def stream(self, messages):
        yield "半截"
        raise RuntimeError("stream failed")


class RiskIntentClassifier:
    async def classify(self, user_input: str, history: list[AiMessage]) -> IntentType:
        return IntentType.RISK


class ChatIntentClassifier:
    async def classify(self, user_input: str, history: list[AiMessage]) -> IntentType:
        return IntentType.CHAT


class RiskAssessmentService:
    async def assess(self, user_input: str, history: list[AiMessage]) -> PsychologyAssessment:
        return PsychologyAssessment(EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, 0.95, "risk")


class EmptyRagService:
    async def retrieve(self, db, user_input: str, history: list[AiMessage]) -> AgenticRagResult:
        return AgenticRagResult.empty()


class MemoryService:
    async def recent(self, session_id: str) -> list:
        return []

    async def refresh(self, session_id: str, messages: list) -> None:
        return None

    async def append(self, session_id: str, role, content: str) -> None:
        return None


class StaticChatService(ChatService):
    def __init__(self, ai_client, settings: Settings | None = None) -> None:
        super().__init__(
            ai_client=ai_client,
            intent_classifier=None,
            assessment_service=None,
            rag_service=None,
            tool_service=None,
            short_memory=None,
            settings=settings or Settings(ai_provider="mock"),
        )
        self.saved_messages: list[tuple[object, str]] = []

    async def prepare(self, db, user, request, multimodal_analysis):
        return PreparedConversation(
            user=FakeUser(),
            session=FakeChatSession(),
            main_reply="agent main reply",
        )

    async def save_message(self, db, user, chat_session, role, content):
        self.saved_messages.append((role, content))


class ToolSchedulingChatService(ChatService):
    def __init__(self) -> None:
        super().__init__(
            ai_client=OneTokenAiClient(),
            intent_classifier=RiskIntentClassifier(),
            assessment_service=RiskAssessmentService(),
            rag_service=EmptyRagService(),
            tool_service=None,
            short_memory=MemoryService(),
            settings=Settings(ai_provider="mock"),
        )
        self.scheduled_report_ids: list[int] = []
        self.agent_runtime = SuccessfulAgentRuntime()

    async def resolve_session(self, db, user, public_id, user_input):
        return FakeChatSession()

    async def recent_model_history(self, db, chat_session) -> list[AiMessage]:
        return []

    async def save_message(self, db, user, chat_session, role, content) -> None:
        return None

    async def save_report(self, db, user, chat_session, content, intent, assessment, multimodal_analysis):
        class Report:
            id = 42

        return Report()

    def schedule_tools(self, report_id: int) -> None:
        self.scheduled_report_ids.append(report_id)


class DisplayTextChatService(ChatService):
    def __init__(self) -> None:
        super().__init__(
            ai_client=OneTokenAiClient(),
            intent_classifier=ChatIntentClassifier(),
            assessment_service=RiskAssessmentService(),
            rag_service=EmptyRagService(),
            tool_service=None,
            short_memory=MemoryService(),
            settings=Settings(ai_provider="mock"),
            agent_runtime=SuccessfulAgentRuntime(),
        )
        self.saved_messages: list[tuple[object, str]] = []
        self.resolved_title = ""

    async def resolve_session(self, db, user, public_id, user_input):
        self.resolved_title = user_input
        return FakeChatSession()

    async def recent_model_history(self, db, chat_session) -> list[AiMessage]:
        return []

    async def save_message(self, db, user, chat_session, role, content) -> None:
        self.saved_messages.append((role, content))


class FailingAgentRuntime:
    async def run_sidecar(self, *args, **kwargs):
        raise RuntimeError("agent sidecar failed")


class SuccessfulAgentRuntime:
    async def run_sidecar(self, *args, **kwargs):
        return LeadAgentResult(
            run_id="run-123",
            complex_task=False,
            dispatched_agents=["consultation"],
            results=[],
            summary="旁路 Agent 已完成内部汇总。",
            timeout_occurred=False,
            decomposition_error=None,
        )


class SidecarFailureChatService(ChatService):
    def __init__(self) -> None:
        super().__init__(
            ai_client=OneTokenAiClient(),
            intent_classifier=ChatIntentClassifier(),
            assessment_service=RiskAssessmentService(),
            rag_service=EmptyRagService(),
            tool_service=None,
            short_memory=MemoryService(),
            settings=Settings(ai_provider="mock"),
            agent_runtime=FailingAgentRuntime(),
        )
        self.saved_messages: list[tuple[object, str]] = []

    async def resolve_session(self, db, user, public_id, user_input):
        return FakeChatSession()

    async def recent_model_history(self, db, chat_session) -> list[AiMessage]:
        return []

    async def save_message(self, db, user, chat_session, role, content) -> None:
        self.saved_messages.append((role, content))


class SidecarSuccessChatService(SidecarFailureChatService):
    def __init__(self, ai_client=None) -> None:
        super().__init__()
        if ai_client is not None:
            self.ai_client = ai_client
        self.agent_runtime = SuccessfulAgentRuntime()


def sse_payloads(events: list[str]) -> list[dict]:
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


@pytest.mark.asyncio
async def test_background_tool_failure_marks_report_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    report = FakeReport()
    sessions: list[FakeSession] = []

    def fake_session_factory() -> FakeSession:
        session = FakeSession(report)
        sessions.append(session)
        return session

    monkeypatch.setattr(chat_module, "AsyncSessionLocal", fake_session_factory)
    service = ChatService(
        ai_client=None,
        intent_classifier=None,
        assessment_service=None,
        rag_service=None,
        tool_service=FailingToolService(),
        short_memory=None,
    )

    await service._run_tools(7)

    assert report.excel_status == ToolStatus.FAILED
    assert report.tool_error == "tool failed for 7"
    assert sessions[-1].committed is True


@pytest.mark.asyncio
async def test_report_tools_are_not_scheduled_by_chatservice_prepare() -> None:
    service = ToolSchedulingChatService()

    prepared = await service.prepare(None, FakeUser(), ChatRequest(message="我不想活了"), None)

    assert prepared.main_reply == "旁路 Agent 已完成内部汇总。"
    assert service.scheduled_report_ids == []


@pytest.mark.asyncio
async def test_full_stream_does_not_schedule_report_tools_from_old_path() -> None:
    service = ToolSchedulingChatService()

    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="我不想活了"))]

    assert service.scheduled_report_ids == []
    assert sse_payloads(events)[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_multimodal_stream_meta_and_user_history_use_display_text() -> None:
    service = DisplayTextChatService()
    analysis = MultimodalAnalysis(
        model_text="【多模态后台分析】audio: Whisper 转写后情绪分析：我最近有些焦虑。",
        signals=[
            MultimodalSignal(
                "visual",
                EmotionLabel.DEPRESSED,
                3.72,
                0.9,
                "POSTER++ RAF-DB 表情分类。",
            )
        ],
        fused_assessment=PsychologyAssessment(EmotionLabel.ANXIETY, 0.8, RiskLevel.LOW, 0.8, "audio"),
        summary="audio",
        display_text="我最近有些焦虑。",
    )

    events = [
        event
        async for event in service.stream_chat(
            None,
            FakeUser(),
            ChatRequest(message="学生上传了多模态内容，希望获得支持。"),
            analysis,
        )
    ]
    payloads = sse_payloads(events)

    assert payloads[0]["type"] == "meta"
    assert payloads[0]["displayText"] == "我最近有些焦虑。"
    assert payloads[0]["visualEmotion"] == "DEPRESSED"
    assert payloads[0]["visualRiskLevel"] == "MEDIUM"
    assert payloads[0]["visualConfidence"] == 0.9
    assert service.resolved_title == "我最近有些焦虑。"
    assert service.saved_messages[0][1] == "我最近有些焦虑。"


@pytest.mark.asyncio
async def test_empty_legacy_sidecar_does_not_break_agent_main_reply() -> None:
    service = StaticChatService(EmptyAiClient())
    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="hi"))]
    payloads = sse_payloads(events)
    assert not any(payload["type"] == "error" for payload in payloads)
    assert "".join(payload["content"] for payload in payloads if payload["type"] == "token") == "agent main reply"
    assert payloads[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_slow_legacy_sidecar_does_not_break_agent_main_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    service = StaticChatService(SlowAiClient())
    service.model_stream_timeout_seconds = 0.01
    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="hi"))]
    payloads = sse_payloads(events)
    assert not any(payload["type"] == "error" for payload in payloads)
    assert "".join(payload["content"] for payload in payloads if payload["type"] == "token") == "agent main reply"
    assert payloads[-1]["type"] == "done"


def test_chat_history_limit_controls_model_window() -> None:
    service = StaticChatService(EmptyAiClient(), Settings(ai_provider="mock", chat_history_limit=2))
    history = [AiMessage("user", f"u{i}") for i in range(6)]
    assert [message.content for message in service.with_current_user(history, "current")] == [
        "u3",
        "u4",
        "u5",
        "current",
    ]
    messages = service.build_messages(FakeUser(), IntentType.CHAT, RiskLevel.LOW, AgenticRagResult.empty(), history)
    assert len(messages) == 5
    assert "学生显示名：student" in messages[0].content
    assert [message.content for message in messages[1:]] == ["u2", "u3", "u4", "u5"]


@pytest.mark.asyncio
async def test_agent_main_failure_returns_error() -> None:
    service = SidecarFailureChatService()

    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="你好"))]
    payloads = sse_payloads(events)

    assert payloads[-1]["type"] == "error"
    assert "多 Agent 主链路评估失败" in payloads[-1]["content"]
    assert not any(payload["type"] == "agent" for payload in payloads)
    assert not any(content == "ok" for _role, content in service.saved_messages)


@pytest.mark.asyncio
async def test_agent_main_success_emits_legacy_sidecar_stream_event() -> None:
    service = SidecarSuccessChatService()

    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="你好"))]
    payloads = sse_payloads(events)
    agent_payloads = [payload for payload in payloads if payload["type"] == "agent"]

    assert len(agent_payloads) == 1
    assert agent_payloads[0]["content"] == "ok"
    assert agent_payloads[0]["agentRunId"] == "run-123"
    assert agent_payloads[0]["agentDispatchedAgents"] == ["consultation"]
    assert agent_payloads[0]["agentTimeoutOccurred"] is False
    assert "".join(payload["content"] for payload in payloads if payload["type"] == "token") == "ok"
    assert payloads[-1]["type"] == "done"
    assert service.saved_messages[-1][1] == "ok"


@pytest.mark.asyncio
async def test_agent_main_stream_forwards_llm_chunks_without_rechunking() -> None:
    ai_client = MultiChunkAiClient(["你", "好，", "我在。"])
    service = SidecarSuccessChatService(ai_client)

    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="你好"))]
    payloads = sse_payloads(events)
    token_payloads = [payload for payload in payloads if payload["type"] == "token"]

    assert [payload["content"] for payload in token_payloads] == ["你", "好，", "我在。"]
    assert service.saved_messages[-1][1] == "你好，我在。"
    assert ai_client.prompts
    assert any("Agent 结果" in prompt for prompt in ai_client.prompts)


@pytest.mark.asyncio
async def test_agent_main_stream_error_does_not_save_partial_reply() -> None:
    service = SidecarSuccessChatService(FailingStreamAiClient())

    events = [event async for event in service.stream_chat(None, FakeUser(), ChatRequest(message="你好"))]
    payloads = sse_payloads(events)

    assert any(payload["type"] == "token" and payload["content"] == "半截" for payload in payloads)
    assert payloads[-1]["type"] == "error"
    assert "stream failed" in payloads[-1]["content"]
    assert not any(role == MessageRole.ASSISTANT for role, _content in service.saved_messages)
