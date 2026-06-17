from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.services.ai import AiMessage


class AgentTaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AgentContext:
    session_id: int
    public_session_id: str
    user_id: int
    display_name: str
    history: list[AiMessage]
    short_memory: list[AiMessage] = field(default_factory=list)
    long_memory: list[str] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(default_factory=dict)
    mcp_tools: dict[str, Any] = field(default_factory=dict)
    capability_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    id: int | None
    session_id: int
    run_id: str
    assigned_agent: str
    task_type: str
    payload: dict[str, Any]
    status: AgentTaskStatus = AgentTaskStatus.PENDING


@dataclass(frozen=True)
class AgentTaskResult:
    task_id: int | None
    agent_name: str
    status: AgentTaskStatus
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
