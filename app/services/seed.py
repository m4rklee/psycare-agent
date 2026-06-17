from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.entities import UserAccount, UserAccountRole
from app.services.knowledge import KnowledgeService


async def seed_initial_data(session: AsyncSession, knowledge_service: KnowledgeService) -> None:
    count = await session.scalar(select(func.count(UserAccount.id)))
    if not count:
        admin = UserAccount(
            username="admin",
            display_name="Counselor Admin",
            password=hash_password("admin123"),
        )
        admin.roles = [UserAccountRole(role="ROLE_ADMIN"), UserAccountRole(role="ROLE_USER")]
        student = UserAccount(
            username="student",
            display_name="student",
            password=hash_password("student123"),
        )
        student.roles = [UserAccountRole(role="ROLE_USER")]
        session.add_all([admin, student])
        await session.commit()
    await knowledge_service.ingest_classpath_if_empty(session)
    await knowledge_service.sync_chroma_from_db(session)
