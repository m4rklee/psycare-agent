import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.infrastructure.runtime import AgentRuntime
from app.agent.orchestration.lead import LeadAgentResult
from app.core.config import Settings, get_settings
from app.core.database import AsyncSessionLocal
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.models.enums import IntentType, MessageRole, RiskLevel, ToolStatus
from app.schemas.api import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient, AiMessage
from app.services.assessment import PsychologicalAssessmentService, PsychologyAssessment
from app.services.knowledge import AgenticRagResult, AgenticRagService
from app.services.memory import MemoryMessage, ShortTermMemoryService
from app.services.multimodal import MultimodalAnalysis
from app.services.prompts import answer_system_prompt
from app.services.rules import IntentClassifier, sanitize
from app.services.tools import ToolOrchestrationService


@dataclass(frozen=True)
class PreparedConversation:
    user: UserAccount
    session: ChatSession
    main_reply: str
    display_text: str | None = None
    sidecar_reply: str | None = None
    agent_result: LeadAgentResult | None = None


def sse(event: str, data: ChatStreamEvent) -> str:
    return f"event: {event}\ndata: {data.model_dump_json(exclude_none=True)}\n\n"


class ChatService:
    def __init__(
        self,
        ai_client: AiClient,
        intent_classifier: IntentClassifier,
        assessment_service: PsychologicalAssessmentService,
        rag_service: AgenticRagService,
        tool_service: ToolOrchestrationService,
        short_memory: ShortTermMemoryService,
        settings: Settings | None = None,
        agent_runtime: AgentRuntime | None = None,
    ) -> None:
        self.ai_client = ai_client
        self.intent_classifier = intent_classifier
        self.assessment_service = assessment_service
        self.rag_service = rag_service
        self.tool_service = tool_service
        self.short_memory = short_memory
        self.settings = settings or get_settings()
        self.agent_runtime = agent_runtime
        self.model_stream_timeout_seconds = 45.0

    async def stream_chat(
        self,
        db: AsyncSession,
        user: UserAccount,
        request: ChatRequest,
        multimodal_analysis: MultimodalAnalysis | None = None,
    ) -> AsyncIterator[str]:
        try:
            prepared = await self.prepare(db, user, request, multimodal_analysis)
            yield sse(
                "meta",
                ChatStreamEvent(
                    type="meta",
                    sessionId=prepared.session.public_id,
                    content="",
                    displayText=prepared.display_text,
                    **self.visual_meta(multimodal_analysis),
                ),
            )
            if prepared.sidecar_reply:
                yield sse(
                    "agent",
                    ChatStreamEvent(
                        type="agent",
                        sessionId=prepared.session.public_id,
                        content=prepared.sidecar_reply,
                        agentRunId=prepared.agent_result.run_id if prepared.agent_result else None,
                        agentDispatchedAgents=prepared.agent_result.dispatched_agents if prepared.agent_result else None,
                        agentTimeoutOccurred=prepared.agent_result.timeout_occurred if prepared.agent_result else None,
                        agentDecompositionError=prepared.agent_result.decomposition_error if prepared.agent_result else None,
                    ),
                )
            assistant_reply = ""
            async for token in self.stream_main_reply_tokens(prepared):
                assistant_reply += token
                yield sse("token", ChatStreamEvent(type="token", sessionId=prepared.session.public_id, content=token))
            if not assistant_reply:
                yield sse(
                    "error",
                    ChatStreamEvent(
                        type="error",
                        sessionId=prepared.session.public_id,
                        content="多 Agent 没有返回内容，请稍后重试。",
                    ),
                )
            if assistant_reply:
                await self.save_message(db, prepared.user, prepared.session, MessageRole.ASSISTANT, assistant_reply)
            yield sse("done", ChatStreamEvent(type="done", sessionId=prepared.session.public_id, content=""))
        except Exception as exc:
            yield sse("error", ChatStreamEvent(type="error", sessionId=None, content=f"服务暂时不可用：{exc}"))

    async def prepare(
        self,
        db: AsyncSession,
        user: UserAccount,
        request: ChatRequest,
        multimodal_analysis: MultimodalAnalysis | None,
    ) -> PreparedConversation:
        original_input = request.message.strip()
        model_input = sanitize(multimodal_analysis.model_text if multimodal_analysis else original_input)
        display_input = self.display_input(original_input, multimodal_analysis)
        chat_session = await self.resolve_session(db, user, request.sessionId, display_input)
        previous_history = await self.recent_model_history(db, chat_session)
        await self.save_message(db, user, chat_session, MessageRole.USER, display_input)
        if multimodal_analysis:
            await self.save_message(db, user, chat_session, MessageRole.SYSTEM, self.multimodal_memory(multimodal_analysis))
        model_history = self.with_current_user(previous_history, model_input)
        agent_result = await self.run_agent_sidecar(
            db,
            user,
            chat_session,
            display_input,
            model_history,
            multimodal_analysis,
            defer_summary=True,
        )
        if agent_result is None or (not agent_result.results and agent_result.summary.strip() == "评估失败"):
            raise RuntimeError("多 Agent 主链路评估失败")
        sidecar_reply = await self.run_legacy_sidecar_reply(
            db,
            user,
            chat_session,
            model_input,
            display_input,
            model_history,
            multimodal_analysis,
        )
        return PreparedConversation(
            user=user,
            session=chat_session,
            main_reply=agent_result.summary,
            display_text=display_input,
            sidecar_reply=sidecar_reply,
            agent_result=agent_result,
        )

    async def run_legacy_sidecar_reply(
        self,
        db: AsyncSession,
        user: UserAccount,
        chat_session: ChatSession,
        model_input: str,
        display_input: str,
        model_history: list[AiMessage],
        multimodal_analysis: MultimodalAnalysis | None,
    ) -> str | None:
        try:
            messages = await self.prepare_legacy_messages(
                db,
                user,
                chat_session,
                model_input,
                display_input,
                model_history,
                multimodal_analysis,
            )
            return await self.complete_legacy_reply(messages)
        except Exception:
            return None

    async def prepare_legacy_messages(
        self,
        db: AsyncSession,
        user: UserAccount,
        chat_session: ChatSession,
        model_input: str,
        display_input: str,
        model_history: list[AiMessage],
        multimodal_analysis: MultimodalAnalysis | None,
    ) -> list[AiMessage]:
        intent = await self.intent_classifier.classify(model_input, model_history)
        if multimodal_analysis and multimodal_analysis.fused_assessment.risk == RiskLevel.HIGH:
            intent = IntentType.RISK
        elif (
            multimodal_analysis
            and multimodal_analysis.fused_assessment.risk == RiskLevel.MEDIUM
            and intent == IntentType.CHAT
        ):
            intent = IntentType.CONSULT

        assessment: PsychologyAssessment | None = None
        rag_result = AgenticRagResult.empty()
        if intent != IntentType.CHAT:
            rag_result = await self.rag_service.retrieve(db, model_input, model_history)
            assessment = (
                multimodal_analysis.fused_assessment
                if multimodal_analysis
                else await self.assessment_service.assess(model_input, model_history)
            )
            if intent == IntentType.RISK and assessment.risk != RiskLevel.HIGH:
                assessment = PsychologyAssessment(
                    assessment.emotion,
                    max(assessment.emotion_score, 4.0),
                    RiskLevel.HIGH,
                    assessment.confidence,
                    assessment.summary,
            )

        risk_level = assessment.risk if assessment else RiskLevel.LOW
        return self.build_messages(user, intent, risk_level, rag_result, model_history)

    async def complete_legacy_reply(self, messages: list[AiMessage]) -> str:
        tokens = []
        async for token in self.stream_model_tokens(messages):
            tokens.append(token)
        return "".join(tokens).strip()

    async def run_agent_sidecar(
        self,
        db: AsyncSession,
        user: UserAccount,
        chat_session: ChatSession,
        user_input: str,
        history: list[AiMessage],
        multimodal_analysis: MultimodalAnalysis | None,
        defer_summary: bool = False,
    ) -> LeadAgentResult | None:
        if self.agent_runtime is None:
            return None
        try:
            return await self.agent_runtime.run_sidecar(
                db,
                user,
                chat_session,
                user_input,
                history,
                multimodal_analysis.summary if multimodal_analysis else None,
                defer_summary=defer_summary,
            )
        except Exception:
            return None

    def display_input(self, original_input: str, multimodal_analysis: MultimodalAnalysis | None) -> str:
        if multimodal_analysis and multimodal_analysis.display_text.strip():
            return multimodal_analysis.display_text.strip()
        return original_input or "学生上传了多模态内容，希望获得支持。"

    def visual_meta(self, multimodal_analysis: MultimodalAnalysis | None) -> dict[str, str | float]:
        if not multimodal_analysis:
            return {}
        visual = next((signal for signal in multimodal_analysis.signals if signal.modality == "visual"), None)
        if not visual:
            return {}
        risk = RiskLevel.HIGH if visual.score >= 4.0 else RiskLevel.MEDIUM if visual.score >= 3.0 else RiskLevel.LOW
        return {
            "visualEmotion": visual.emotion.value,
            "visualRiskLevel": risk.value,
            "visualConfidence": visual.confidence,
            "visualEvidence": visual.evidence,
        }

    async def resolve_session(
        self, db: AsyncSession, user: UserAccount, public_id: str | None, user_input: str
    ) -> ChatSession:
        if public_id:
            chat_session = await db.scalar(
                select(ChatSession).where(ChatSession.public_id == public_id, ChatSession.user_id == user.id)
            )
            if not chat_session:
                raise ValueError("Session not found")
            return chat_session
        chat_session = ChatSession(
            public_id=uuid.uuid4().hex,
            user_id=user.id,
            title=user_input[:36] if len(user_input) > 36 else user_input,
        )
        db.add(chat_session)
        await db.commit()
        await db.refresh(chat_session)
        chat_session.user = user
        return chat_session

    async def save_message(
        self, db: AsyncSession, user: UserAccount, chat_session: ChatSession, role: MessageRole, content: str
    ) -> None:
        db.add(ChatMessage(user_id=user.id, session_id=chat_session.id, role=role, content=content))
        chat_session.updated_at = datetime.utcnow()
        await db.commit()
        await self.short_memory.append(chat_session.public_id, role, content)

    async def recent_model_history(self, db: AsyncSession, chat_session: ChatSession) -> list[AiMessage]:
        redis_history = await self.short_memory.recent(chat_session.public_id)
        if redis_history:
            return [self.to_ai_message(message.role, sanitize(message.content)) for message in redis_history]
        rows = (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == chat_session.id)
                .order_by(ChatMessage.created_at.desc())
                .limit(self.message_window_limit())
            )
        ).all()
        history = list(reversed(rows))
        await self.short_memory.refresh(
            chat_session.public_id,
            [MemoryMessage(message.role, message.content) for message in history],
        )
        return [self.to_ai_message(message.role, sanitize(message.content)) for message in history]

    def with_current_user(self, previous_history: list[AiMessage], current_input: str) -> list[AiMessage]:
        history = previous_history + [AiMessage("user", current_input)]
        return history[-self.message_window_limit() :]

    def to_ai_message(self, role: MessageRole, content: str) -> AiMessage:
        return AiMessage(
            {
                MessageRole.USER: "user",
                MessageRole.ASSISTANT: "assistant",
                MessageRole.SYSTEM: "system",
            }[role],
            content,
        )

    def build_messages(
        self,
        user: UserAccount,
        intent: IntentType,
        risk_level: RiskLevel,
        rag_result: AgenticRagResult,
        history: list[AiMessage],
    ) -> list[AiMessage]:
        return [answer_system_prompt(intent, risk_level, rag_result.context_block(), user.username)] + history[
            -self.message_window_limit() :
        ]

    async def stream_model_tokens(self, messages: list[AiMessage]) -> AsyncIterator[str]:
        iterator = self.ai_client.stream(messages).__aiter__()
        while True:
            try:
                yield await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=self.model_stream_timeout_seconds,
                )
            except StopAsyncIteration:
                break

    async def stream_main_reply_tokens(self, prepared: PreparedConversation) -> AsyncIterator[str]:
        if prepared.agent_result is not None:
            if self.agent_runtime is not None and hasattr(self.agent_runtime, "stream_final_answer"):
                iterator = self.agent_runtime.stream_final_answer(
                    prepared.display_text or "",
                    prepared.agent_result,
                ).__aiter__()
                while True:
                    try:
                        token = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=self.model_stream_timeout_seconds,
                        )
                    except StopAsyncIteration:
                        break
                    if token:
                        yield token
                return
            messages = self.agent_result_fallback_messages(prepared.display_text or "", prepared.agent_result)
            async for token in self.stream_model_tokens(messages):
                if token:
                    yield token
            return
        if prepared.main_reply:
            yield prepared.main_reply

    def agent_result_fallback_messages(self, user_input: str, result: LeadAgentResult) -> list[AiMessage]:
        return [
            AiMessage(
                "user",
                "你是校园心理健康支持助手。请基于下面 Agent 结果生成面向学生的最终回答，"
                "保持温和、清晰、可执行，不要输出 JSON。\n\n"
                f"用户问题：{user_input}\n\n"
                f"Agent 结果：\n{result.summary or self.agent_results_text(result)}",
            )
        ]

    def agent_results_text(self, result: LeadAgentResult) -> str:
        return "\n\n".join(item.content or item.error or "" for item in result.results).strip()

    def message_window_limit(self) -> int:
        return max(2, self.settings.chat_history_limit * 2)

    async def save_report(
        self,
        db: AsyncSession,
        user: UserAccount,
        chat_session: ChatSession,
        content: str,
        intent: IntentType,
        assessment: PsychologyAssessment,
        multimodal_analysis: MultimodalAnalysis | None,
    ) -> PsychologicalReport:
        report = PsychologicalReport(
            user_id=user.id,
            session_id=chat_session.id,
            content=content,
            intent=intent,
            emotion=assessment.emotion,
            emotion_score=assessment.emotion_score,
            risk_level=assessment.risk,
            confidence=assessment.confidence,
            summary=assessment.summary,
            emotion_tags=multimodal_analysis.emotion_tags_json() if multimodal_analysis else None,
        )
        db.add(report)
        await db.commit()
        await db.refresh(report)
        report.user = user
        report.session = chat_session
        return report

    def multimodal_memory(self, analysis: MultimodalAnalysis) -> str:
        modalities = "、".join(dict.fromkeys(signal.modality for signal in analysis.signals))
        evidence = "；".join(f"{signal.modality}={signal.evidence}" for signal in analysis.signals)
        return f"""【多模态分析记忆】
用户本轮上传了{modalities or "附件"}，后端已完成多模态情绪分析。后续如果用户追问“你是否根据图片/语音/视频分析”，应说明：我是基于后端多模态分析结果和你的文字一起判断，不是只凭文字猜测。不要否认已上传附件，也不要声称自己直接查看了原始文件。
分析摘要：{analysis.summary}
情绪标签：{analysis.emotion_tags_json()}
分析证据：{evidence or "无"}"""

    async def _run_tools(self, report_id: int) -> None:
        try:
            async with AsyncSessionLocal() as db:
                await self.tool_service.handle(db, report_id)
        except Exception as exc:
            await self._mark_tool_failure(report_id, exc)

    def schedule_tools(self, report_id: int) -> None:
        asyncio.create_task(self._run_tools(report_id))

    async def _mark_tool_failure(self, report_id: int, exc: Exception) -> None:
        try:
            async with AsyncSessionLocal() as db:
                report = await db.scalar(
                    select(PsychologicalReport).where(PsychologicalReport.id == report_id)
                )
                if not report:
                    return
                if report.excel_status == ToolStatus.PENDING:
                    report.excel_status = ToolStatus.FAILED
                report.tool_error = str(exc)[:500]
                await db.commit()
        except Exception:
            pass
