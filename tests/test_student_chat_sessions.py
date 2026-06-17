from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.database import Base
from app.models.entities import ChatMessage, ChatSession, UserAccount, UserAccountRole
from app.models.enums import MessageRole
from app.services.reports import ReportService


@pytest.mark.asyncio
async def test_student_sessions_are_limited_to_owner_and_sorted(tmp_path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        student, other = await seed_users(db)
        older = await add_session(db, student.id, "older", "较早会话", minutes_ago=20)
        newer = await add_session(db, student.id, "newer", "较新会话", minutes_ago=2)
        await add_session(db, other.id, "other", "其他学生会话", minutes_ago=1)

        rows = await ReportService().student_sessions(db, student.id)

    assert [row.sessionId for row in rows] == [newer.public_id, older.public_id]
    assert [row.title for row in rows] == ["较新会话", "较早会话"]


@pytest.mark.asyncio
async def test_student_conversation_requires_owner_and_excludes_system_messages(tmp_path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        student, other = await seed_users(db)
        owned = await add_session(db, student.id, "owned", "我的会话", minutes_ago=1)
        foreign = await add_session(db, other.id, "foreign", "别人的会话", minutes_ago=1)
        await add_message(db, owned.id, student.id, MessageRole.USER, "你好")
        await add_message(db, owned.id, student.id, MessageRole.SYSTEM, "hidden context")
        await add_message(db, owned.id, student.id, MessageRole.ASSISTANT, "你好，我在。")

        service = ReportService()
        response = await service.student_conversation(db, student.id, owned.public_id)
        foreign_response = await service.student_conversation(db, student.id, foreign.public_id)
        missing_response = await service.student_conversation(db, student.id, "missing")

    assert response is not None
    assert response.sessionId == "owned"
    assert [message.role for message in response.messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert [message.content for message in response.messages] == ["你好", "你好，我在。"]
    assert foreign_response is None
    assert missing_response is None


async def make_session_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


async def seed_users(db):
    student = UserAccount(username="student", password="x", display_name="student")
    student.roles = [UserAccountRole(role="ROLE_USER")]
    other = UserAccount(username="other", password="x", display_name="other")
    other.roles = [UserAccountRole(role="ROLE_USER")]
    db.add_all([student, other])
    await db.commit()
    await db.refresh(student)
    await db.refresh(other)
    return student, other


async def add_session(db, user_id: int, public_id: str, title: str, minutes_ago: int) -> ChatSession:
    when = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    chat_session = ChatSession(
        public_id=public_id,
        user_id=user_id,
        title=title,
        created_at=when,
        updated_at=when,
    )
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


async def add_message(db, session_id: int, user_id: int, role: MessageRole, content: str) -> None:
    db.add(ChatMessage(session_id=session_id, user_id=user_id, role=role, content=content))
    await db.commit()
