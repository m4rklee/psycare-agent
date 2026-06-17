import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.communication.types import AgentTask, AgentTaskResult, AgentTaskStatus
from app.models.entities import AgentTaskRecord


def _json(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), ensure_ascii=False, default=str)


class AgentTaskStore:
    async def create(self, session: AsyncSession, task: AgentTask) -> AgentTask:
        record = AgentTaskRecord(
            session_id=task.session_id,
            run_id=task.run_id,
            agent_name=task.assigned_agent,
            task_type=task.task_type,
            status=task.status.value,
            input_json=_json(task.payload),
        )
        session.add(record)
        await session.flush()
        return AgentTask(
            id=record.id,
            session_id=task.session_id,
            run_id=task.run_id,
            assigned_agent=task.assigned_agent,
            task_type=task.task_type,
            payload=task.payload,
            status=AgentTaskStatus(record.status),
        )

    async def mark_running(self, session: AsyncSession, task_id: int | None) -> None:
        record = await self._record(session, task_id)
        if record is None:
            return
        now = datetime.now(UTC)
        record.status = AgentTaskStatus.RUNNING.value
        record.started_at = now
        record.updated_at = now
        await session.flush()

    async def complete(self, session: AsyncSession, result: AgentTaskResult) -> None:
        record = await self._record(session, result.task_id)
        if record is None:
            return
        now = datetime.now(UTC)
        record.status = result.status.value
        record.output_json = _json({"content": result.content, "metadata": result.metadata})
        record.error = result.error[:500] if result.error else None
        record.finished_at = now
        record.updated_at = now
        await session.flush()

    async def _record(self, session: AsyncSession, task_id: int | None) -> AgentTaskRecord | None:
        if task_id is None:
            return None
        return await session.get(AgentTaskRecord, task_id)
