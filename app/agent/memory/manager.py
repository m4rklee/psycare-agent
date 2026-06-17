import hashlib
import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import AgentMemorySummary


class LongTermMemoryManager:
    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def content_hash(self, text: str) -> str:
        return hashlib.sha256(self.normalize(text).encode("utf-8")).hexdigest()

    def compress(self, text: str, max_chars: int = 1200) -> str:
        normalized = self.normalize(text)
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def token_estimate(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    async def save_summary(
        self,
        session: AsyncSession,
        session_id: int,
        summary: str,
        version: int = 1,
    ) -> AgentMemorySummary:
        compressed = self.compress(summary)
        digest = self.content_hash(compressed)
        existing = await session.scalar(
            select(AgentMemorySummary).where(
                AgentMemorySummary.session_id == session_id,
                AgentMemorySummary.content_hash == digest,
            )
        )
        if existing:
            return existing
        now = datetime.now(UTC)
        record = AgentMemorySummary(
            session_id=session_id,
            summary=compressed,
            content_hash=digest,
            token_estimate=self.token_estimate(compressed),
            version=version,
            updated_at=now,
        )
        session.add(record)
        await session.flush()
        return record

    async def load(self, session: AsyncSession, session_id: int, limit: int = 5) -> list[AgentMemorySummary]:
        rows = (
            await session.scalars(
                select(AgentMemorySummary)
                .where(AgentMemorySummary.session_id == session_id)
                .order_by(AgentMemorySummary.created_at.desc())
                .limit(limit)
            )
        ).all()
        return list(rows)

    async def query(self, session: AsyncSession, session_id: int, text: str, limit: int = 5) -> list[AgentMemorySummary]:
        terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", text.lower()) if term]
        rows = await self.load(session, session_id, limit=50)
        if not terms:
            return rows[:limit]
        ranked = sorted(
            rows,
            key=lambda row: sum(1 for term in terms if term in row.summary.lower()),
            reverse=True,
        )
        return [row for row in ranked if any(term in row.summary.lower() for term in terms)][:limit]
