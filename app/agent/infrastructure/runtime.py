from sqlalchemy.ext.asyncio import AsyncSession
from collections.abc import AsyncIterator

from app.agent.capabilities import CapabilityExecutor, CapabilityRegistry
from app.agent.communication.types import AgentContext
from app.agent.memory.manager import LongTermMemoryManager
from app.agent.orchestration.lead import LeadAgent, LeadAgentResult
from app.core.config import Settings
from app.models.entities import ChatSession, UserAccount
from app.services.ai import AiClient, AiMessage
from app.services.knowledge import KnowledgeService
from app.services.tools import ToolOrchestrationService


class AgentRuntime:
    def __init__(
        self,
        settings: Settings,
        ai_client: AiClient,
        lead_agent: LeadAgent | None = None,
        memory_manager: LongTermMemoryManager | None = None,
        knowledge_service: KnowledgeService | None = None,
        tool_service: ToolOrchestrationService | None = None,
        capability_registry: CapabilityRegistry | None = None,
        capability_executor: CapabilityExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.ai_client = ai_client
        self.memory_manager = memory_manager or LongTermMemoryManager()
        self.knowledge_service = knowledge_service
        self.tool_service = tool_service
        self.capability_registry = capability_registry or CapabilityRegistry(settings.agent_skills_dir)
        self.capability_executor = capability_executor or CapabilityExecutor(
            self.capability_registry,
            timeout_seconds=settings.agent_capability_timeout_seconds,
            max_calls=settings.agent_max_capability_calls,
            enable_mcp_auto_call=settings.agent_enable_mcp_auto_call,
        )
        self.lead_agent = lead_agent or LeadAgent(
            ai_client,
            timeout_seconds=settings.agent_task_timeout_seconds,
            capability_executor=self.capability_executor,
        )

    async def run_sidecar(
        self,
        db: AsyncSession,
        user: UserAccount,
        chat_session: ChatSession,
        user_input: str,
        history: list[AiMessage],
        multimodal_summary: str | None = None,
        defer_summary: bool = False,
    ) -> LeadAgentResult | None:
        if not self.settings.agent_enable_background_orchestration:
            return None
        long_memory = [
            record.summary
            for record in await self.memory_manager.load(db, chat_session.id, limit=5)
        ]
        context = AgentContext(
            session_id=chat_session.id,
            public_session_id=chat_session.public_id,
            user_id=user.id,
            display_name=user.display_name,
            history=history,
            long_memory=long_memory,
            tools=self.capability_registry.tool_metadata(),
            skills=self.capability_registry.skill_metadata(),
            mcp_tools=self.capability_registry.mcp_metadata(),
            capability_context={
                "db": db,
                "user": user,
                "chat_session": chat_session,
                "tool_service": self.tool_service,
                "knowledge_service": self.knowledge_service,
                "history": history,
                "long_memory": long_memory,
                "session_id": chat_session.public_id,
                "multimodal_summary": multimodal_summary,
            },
        )
        result = await self.lead_agent.run(
            db,
            context,
            user_input,
            multimodal_summary,
            summarize=not defer_summary,
        )
        if user_input.strip():
            await self.memory_manager.save_summary(
                db,
                chat_session.id,
                f"用户最近表达摘要：{user_input.strip()}",
            )
        await db.commit()
        return result

    async def stream_final_answer(
        self,
        user_input: str,
        result: LeadAgentResult,
    ) -> AsyncIterator[str]:
        async for token in self.lead_agent.stream_final_answer(
            user_input,
            result.results,
            result.timeout_occurred,
        ):
            yield token
