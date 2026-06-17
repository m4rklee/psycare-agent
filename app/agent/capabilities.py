import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.models.entities import PsychologicalReport
from app.models.enums import EmotionLabel, IntentType, RiskLevel
from app.agent.skills import LoadedSkill, SkillLoader


class CapabilityKind(StrEnum):
    SKILL = "skill"
    TOOL = "tool"
    MCP = "mcp"


class CapabilityCallStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    kind: CapabilityKind
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    auto_callable: bool = True
    side_effect: bool = False

    def public_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "metadata": self.metadata,
            "autoCallable": self.auto_callable,
            "sideEffect": self.side_effect,
        }


@dataclass(frozen=True)
class CapabilityCall:
    name: str
    input: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "input": self.input}


@dataclass(frozen=True)
class CapabilityCallResult:
    name: str
    kind: CapabilityKind | None
    status: CapabilityCallStatus
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value if self.kind else "",
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
        }


class CapabilityRegistry:
    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = skills_dir
        self.skills = SkillLoader(skills_dir).load_all()
        self.definitions: dict[str, CapabilityDefinition] = {}
        self._register_skills()
        self._register_builtin_tools()
        self._register_builtin_mcp_tools()

    def _register_skills(self) -> None:
        for skill in self.skills.values():
            self.definitions[skill.name] = CapabilityDefinition(
                name=skill.name,
                kind=CapabilityKind.SKILL,
                description=skill.description,
                metadata=skill.metadata,
                auto_callable=True,
                side_effect=False,
            )

    def _register_builtin_tools(self) -> None:
        for name, description in {
            "report.create_and_dispatch": "Create a psychological risk report and dispatch configured report/alert tools.",
            "report.excel_writer": "Write psychological reports to the configured report sink.",
            "report.alert_notifier": "Send or record high-risk alert notifications.",
        }.items():
            self.definitions[name] = CapabilityDefinition(
                name=name,
                kind=CapabilityKind.TOOL,
                description=description,
                auto_callable=name == "report.create_and_dispatch",
                side_effect=True,
            )

    def _register_builtin_mcp_tools(self) -> None:
        for name, description in {
            "multimodalAgent.excel.write_report": "Write a psychological report row through MCP.",
            "multimodalAgent.email.send_alert": "Send or record a high-risk counselor alert through MCP.",
        }.items():
            self.definitions[name] = CapabilityDefinition(
                name=name,
                kind=CapabilityKind.MCP,
                description=description,
                auto_callable=False,
                side_effect=True,
            )

    def get(self, name: str) -> CapabilityDefinition | None:
        return self.definitions.get(name)

    def skill(self, name: str) -> LoadedSkill | None:
        return self.skills.get(name)

    def metadata_for(self, kind: CapabilityKind) -> dict[str, dict[str, Any]]:
        return {
            name: definition.public_metadata()
            for name, definition in sorted(self.definitions.items())
            if definition.kind == kind
        }

    def skill_metadata(self) -> dict[str, dict[str, Any]]:
        return self.metadata_for(CapabilityKind.SKILL)

    def tool_metadata(self) -> dict[str, dict[str, Any]]:
        return self.metadata_for(CapabilityKind.TOOL)

    def mcp_metadata(self) -> dict[str, dict[str, Any]]:
        return self.metadata_for(CapabilityKind.MCP)


class CapabilityExecutor:
    def __init__(
        self,
        registry: CapabilityRegistry,
        timeout_seconds: float = 8.0,
        max_calls: int = 3,
        enable_mcp_auto_call: bool = False,
    ) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_calls = max_calls
        self.enable_mcp_auto_call = enable_mcp_auto_call

    async def execute_many(
        self,
        runtime_context: dict[str, Any],
        calls: list[CapabilityCall],
        allowlist: set[str],
    ) -> list[CapabilityCallResult]:
        limited = calls[: self.max_calls]
        return [
            await self.execute(runtime_context, call, allowlist)
            for call in limited
        ]

    async def execute(
        self,
        runtime_context: dict[str, Any],
        call: CapabilityCall,
        allowlist: set[str],
    ) -> CapabilityCallResult:
        definition = self.registry.get(call.name)
        if definition is None:
            return CapabilityCallResult(
                call.name,
                None,
                CapabilityCallStatus.BLOCKED,
                error=f"Unknown capability: {call.name}",
            )
        if call.name not in allowlist:
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.BLOCKED,
                error=f"Capability is not allowed for this agent: {call.name}",
            )
        if not definition.auto_callable or (definition.kind == CapabilityKind.MCP and not self.enable_mcp_auto_call):
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.BLOCKED,
                error=f"Capability is registered but not auto-callable: {call.name}",
            )
        if definition.kind == CapabilityKind.TOOL and call.name == "report.create_and_dispatch":
            return await self._create_and_dispatch_report(runtime_context, call)
        if definition.kind != CapabilityKind.SKILL:
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.BLOCKED,
                error=f"Only skills are executable in this phase: {call.name}",
            )
        skill = self.registry.skill(call.name)
        if skill is None:
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.FAILED,
                error=f"Skill handler not found: {call.name}",
            )
        try:
            output = await asyncio.wait_for(
                skill.run(runtime_context, call.input),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.FAILED,
                error="timeout",
            )
        except Exception as exc:
            return CapabilityCallResult(
                call.name,
                definition.kind,
                CapabilityCallStatus.FAILED,
                error=str(exc),
            )
        return CapabilityCallResult(
            call.name,
            definition.kind,
            CapabilityCallStatus.SUCCESS,
            output=output,
        )

    async def _create_and_dispatch_report(
        self,
        runtime_context: dict[str, Any],
        call: CapabilityCall,
    ) -> CapabilityCallResult:
        db = runtime_context.get("db")
        user = runtime_context.get("user")
        chat_session = runtime_context.get("chat_session")
        tool_service = runtime_context.get("tool_service")
        task = runtime_context.get("task")
        missing = [
            name
            for name, value in {
                "db": db,
                "user": user,
                "chat_session": chat_session,
                "tool_service": tool_service,
            }.items()
            if value is None
        ]
        if missing:
            return CapabilityCallResult(
                call.name,
                CapabilityKind.TOOL,
                CapabilityCallStatus.FAILED,
                error=f"Missing report runtime dependencies: {', '.join(missing)}",
            )
        risk_level = self._risk_level(call.input)
        if risk_level != RiskLevel.HIGH:
            return CapabilityCallResult(
                call.name,
                CapabilityKind.TOOL,
                CapabilityCallStatus.SUCCESS,
                output={
                    "status": "skipped",
                    "reason": "report dispatch only runs for high or emergency risk",
                    "risk_level": risk_level.value,
                },
            )
        content = str(
            call.input.get("content")
            or call.input.get("user_input")
            or getattr(task, "payload", {}).get("user_input", "")
        )
        summary = str(call.input.get("summary") or call.input.get("reason") or "DiagnosticAgent 识别到高风险心理安全线索。")
        report = PsychologicalReport(
            user_id=user.id,
            session_id=chat_session.id,
            content=content,
            intent=IntentType.RISK,
            emotion=EmotionLabel.HIGH_RISK,
            emotion_score=float(call.input.get("emotion_score") or 4.0),
            risk_level=RiskLevel.HIGH,
            confidence=float(call.input.get("confidence") or 0.9),
            summary=summary[:500],
            emotion_tags=runtime_context.get("multimodal_emotion_tags"),
        )
        db.add(report)
        await db.flush()
        await db.refresh(report)
        await tool_service.handle(db, report.id)
        return CapabilityCallResult(
            call.name,
            CapabilityKind.TOOL,
            CapabilityCallStatus.SUCCESS,
            output={
                "status": "created",
                "report_id": report.id,
                "risk_level": report.risk_level.value,
                "excel_status": report.excel_status.value,
                "email_status": report.email_status.value,
            },
        )

    def _risk_level(self, input_data: dict[str, Any]) -> RiskLevel:
        raw = str(input_data.get("risk_level") or input_data.get("riskLevel") or "").lower()
        if raw in {"high", "紧急", "emergency", "urgent", "高"}:
            return RiskLevel.HIGH
        return RiskLevel.LOW
