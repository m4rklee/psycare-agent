import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.communication.types import AgentContext, AgentTask, AgentTaskResult, AgentTaskStatus
from app.agent.infrastructure.task_store import AgentTaskStore

AgentHandler = Callable[[AgentContext, AgentTask], Awaitable[AgentTaskResult]]


class AgentTaskQueue:
    def __init__(self, store: AgentTaskStore | None = None) -> None:
        self.store = store or AgentTaskStore()
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()

    async def submit(self, db: AsyncSession, task: AgentTask) -> AgentTask:
        persisted = await self.store.create(db, task)
        await self._queue.put(persisted)
        return persisted

    async def run_all(
        self,
        db: AsyncSession,
        context: AgentContext,
        handlers: dict[str, AgentHandler],
        timeout_seconds: float,
        run_id: str,
    ) -> list[AgentTaskResult]:
        tasks: list[AgentTask] = []
        retained: list[AgentTask] = []
        while not self._queue.empty():
            task = await self._queue.get()
            if task.run_id == run_id:
                tasks.append(task)
            else:
                retained.append(task)
        for task in retained:
            await self._queue.put(task)
        if not tasks:
            return []
        for task in tasks:
            await self.store.mark_running(db, task.id)
        results = await asyncio.gather(
            *(self._execute_one(context, task, handlers, timeout_seconds) for task in tasks)
        )
        for result in results:
            await self.store.complete(db, result)
        return results

    async def _execute_one(
        self,
        context: AgentContext,
        task: AgentTask,
        handlers: dict[str, AgentHandler],
        timeout_seconds: float,
    ) -> AgentTaskResult:
        handler = handlers.get(task.assigned_agent)
        if handler is None:
            result = AgentTaskResult(
                task.id,
                task.assigned_agent,
                AgentTaskStatus.FAILED,
                "",
                error=f"No handler registered for agent: {task.assigned_agent}",
            )
            return result
        try:
            result = await asyncio.wait_for(handler(context, task), timeout=timeout_seconds)
        except TimeoutError:
            result = AgentTaskResult(
                task.id,
                task.assigned_agent,
                AgentTaskStatus.FAILED,
                "",
                error="timeout",
            )
        except Exception as exc:
            result = AgentTaskResult(
                task.id,
                task.assigned_agent,
                AgentTaskStatus.FAILED,
                "",
                error=str(exc),
            )
        return result
