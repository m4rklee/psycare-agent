from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import AgentTaskRecord, ChatMessage, ChatSession
from app.models.enums import MessageRole


@dataclass(frozen=True)
class AgentSessionState:
    session: ChatSession
    messages: list[ChatMessage]
    tasks: list[AgentTaskRecord]


class AgentSessionManager:
    async def load(self, db: AsyncSession, chat_session: ChatSession) -> AgentSessionState:
        messages = (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == chat_session.id, ChatMessage.role != MessageRole.SYSTEM)
                .order_by(ChatMessage.created_at.asc())
            )
        ).all()
        tasks = (
            await db.scalars(
                select(AgentTaskRecord)
                .where(AgentTaskRecord.session_id == chat_session.id)
                .order_by(AgentTaskRecord.created_at.asc())
            )
        ).all()
        return AgentSessionState(chat_session, list(messages), list(tasks))
