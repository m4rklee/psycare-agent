import json
from dataclasses import dataclass
from typing import Any

from app.agent.capabilities import CapabilityCall, CapabilityCallResult, CapabilityExecutor
from app.agent.communication.types import AgentContext, AgentTask, AgentTaskResult, AgentTaskStatus
from app.services.ai import AiClient, AiMessage


@dataclass(frozen=True)
class ReactStep:
    thought: str
    action: CapabilityCall | None = None
    final: str | None = None


class BaseAgent:
    name = "base"
    task_type = "general"
    CAPABILITY_ALLOWLIST: set[str] = set()

    def __init__(
        self,
        ai_client: AiClient | None = None,
        capability_executor: CapabilityExecutor | None = None,
    ) -> None:
        self.ai_client = ai_client
        self.capability_executor = capability_executor

    async def initialize(self) -> None:
        await self.load_capabilities()

    def role_prompt(self) -> str:
        raise NotImplementedError

    async def load_capabilities(self) -> list[str]:
        return []

    def format_input(self, context: AgentContext, task: AgentTask) -> str:
        current_input = str(task.payload.get("user_input") or "")
        subtask = str(task.payload.get("subtask_description") or "")
        subtask_text = f"\n\n子任务：{subtask}" if subtask else ""
        return f"{self.role_prompt()}{subtask_text}\n\n当前输入：{current_input}"

    async def process_with_llm(self, messages: list[AiMessage]) -> str:
        if self.ai_client is None:
            raise RuntimeError("LLM client is not configured.")
        return await self.ai_client.complete(messages)

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        formatted = self.format_input(context, task)
        capability_calls: list[CapabilityCall] = []
        capability_results: list[CapabilityCallResult] = []
        capability_observation_text = ""
        react_trace: list[dict[str, Any]] = []
        react_error: str | None = None
        stopped_reason = "final"
        final_input = self._final_input(formatted, capability_results)
        try:
            raw, stopped_reason, react_error = await self.run_react_loop(
                context,
                task,
                formatted,
                capability_calls,
                capability_results,
                react_trace,
            )
            capability_observation_text = self.format_capability_observations(capability_results)
            final_input = self._final_input(formatted, capability_results)
        except Exception as exc:
            return AgentTaskResult(
                task.id,
                self.name,
                AgentTaskStatus.FAILED,
                "",
                metadata={
                    "taskType": task.task_type,
                    "formattedInput": final_input,
                    **self._capability_metadata(
                        capability_calls,
                        capability_results,
                        react_error,
                        capability_observation_text,
                    ),
                    **self._react_metadata(react_trace, "llm_error", str(exc)),
                },
                error=str(exc),
            )
        result = self.parse_result(task, raw, final_input)
        result.metadata.update(
            self._capability_metadata(
                capability_calls,
                capability_results,
                react_error,
                capability_observation_text,
            )
        )
        result.metadata.update(self._react_metadata(react_trace, stopped_reason, react_error))
        return result

    def available_capability_names(self) -> set[str]:
        return set(self.CAPABILITY_ALLOWLIST)

    def capability_plan_prompt(
        self,
        context: AgentContext,
        task: AgentTask,
        formatted_input: str,
    ) -> list[AiMessage]:
        available = self.available_capability_metadata(context)
        available_text = json.dumps(available, ensure_ascii=False, default=str)
        return [
            AiMessage(
                "system",
                "你是 Agent 能力调用规划器。请只输出严格 JSON，不要 Markdown，不要解释。",
            ),
            AiMessage(
                "user",
                "请根据当前任务判断是否需要调用 skills。\n"
                "输出格式必须为：{\"calls\":[{\"name\":\"skill_name\",\"input\":{...}}]}。\n"
                "最多选择 3 个；如不需要调用，返回 {\"calls\":[]}。\n"
                "只能选择可用能力列表中的 name。\n\n"
                f"Agent：{self.name}\n"
                f"可用能力：{available_text}\n\n"
                f"任务输入：\n{formatted_input}",
            ),
        ]

    async def plan_capability_calls(
        self,
        context: AgentContext,
        task: AgentTask,
        formatted_input: str,
    ) -> list[CapabilityCall]:
        if self.ai_client is None:
            return []
        raw = await self.process_with_llm(self.capability_plan_prompt(context, task, formatted_input))
        return self.parse_capability_calls(raw)

    async def run_react_loop(
        self,
        context: AgentContext,
        task: AgentTask,
        formatted_input: str,
        capability_calls: list[CapabilityCall],
        capability_results: list[CapabilityCallResult],
        react_trace: list[dict[str, Any]],
    ) -> tuple[str, str, str | None]:
        max_actions = self._max_react_actions()
        react_error: str | None = None
        for step_index in range(max_actions + 1):
            force_final = step_index >= max_actions or not self._can_execute_capabilities()
            try:
                raw = await self.process_with_llm(
                    self.react_step_prompt(context, task, formatted_input, capability_results, force_final)
                )
                step = self.parse_react_step(raw)
            except Exception as exc:
                react_error = str(exc)
                final = await self.generate_react_final(context, task, formatted_input, capability_results)
                react_trace.append(
                    {
                        "step": step_index + 1,
                        "thought": "",
                        "status": "parse_error_fallback",
                        "error": react_error,
                    }
                )
                return final, "parse_error_fallback", react_error

            if step.final is not None:
                react_trace.append(
                    {
                        "step": step_index + 1,
                        "thought": step.thought,
                        "final": True,
                    }
                )
                return step.final, "final", react_error

            if step.action is None:
                react_error = "ReAct step must contain action or final."
                final = await self.generate_react_final(context, task, formatted_input, capability_results)
                react_trace.append(
                    {
                        "step": step_index + 1,
                        "thought": step.thought,
                        "status": "parse_error_fallback",
                        "error": react_error,
                    }
                )
                return final, "parse_error_fallback", react_error

            if force_final:
                react_error = "ReAct returned action when final was required."
                final = await self.generate_react_final(context, task, formatted_input, capability_results)
                react_trace.append(
                    {
                        "step": step_index + 1,
                        "thought": step.thought,
                        "action": step.action.to_dict(),
                        "status": "max_steps",
                        "error": react_error,
                    }
                )
                return final, "max_steps", react_error

            results = await self.execute_capability_calls(context, task, [step.action])
            capability_calls.append(step.action)
            capability_results.extend(results)
            result = results[0] if results else None
            react_trace.append(
                {
                    "step": step_index + 1,
                    "thought": step.thought,
                    "action": step.action.to_dict(),
                    "observation": result.to_dict() if result else {},
                    "status": result.status.value if result else "NO_RESULT",
                }
            )

        final = await self.generate_react_final(context, task, formatted_input, capability_results)
        return final, "max_steps", react_error

    def react_step_prompt(
        self,
        context: AgentContext,
        task: AgentTask,
        formatted_input: str,
        observations: list[CapabilityCallResult],
        force_final: bool = False,
    ) -> list[AiMessage]:
        available = self.available_capability_metadata(context)
        observations_text = json.dumps(
            [observation.to_dict() for observation in observations],
            ensure_ascii=False,
            default=str,
        )
        schema = (
            '{"thought":"简短内部判断","final":"最终结构化回答"}'
            if force_final
            else '{"thought":"简短内部判断","action":{"name":"skill_name","input":{...}}} 或 '
            '{"thought":"简短内部判断","final":"最终结构化回答"}'
        )
        instruction = (
            "现在必须输出 final，不要再输出 action。"
            if force_final
            else "你可以选择 1 个 action 调用一个 skill；如果信息足够，也可以直接输出 final。"
        )
        return [
            AiMessage(
                "system",
                "你是 Agent ReAct 控制器。请只输出严格 JSON，不要 Markdown，不要解释。",
            ),
            AiMessage(
                "user",
                f"{instruction}\n"
                f"输出格式：{schema}\n"
                "action.name 只能来自可用能力列表。\n\n"
                f"Agent：{self.name}\n"
                f"角色提示词：\n{self.role_prompt()}\n\n"
                f"可用能力：{json.dumps(available, ensure_ascii=False, default=str)}\n\n"
                f"任务输入：\n{formatted_input}\n\n"
                f"已有 Observation：\n{observations_text}",
            ),
        ]

    def parse_react_step(self, raw: str) -> ReactStep:
        data = self._json_object(raw, "ReAct step")
        thought = str(data.get("thought") or "").strip()
        if "final" in data and data.get("final") is not None:
            final = str(data.get("final") or "").strip()
            if final:
                return ReactStep(thought=thought, final=final)
            raise ValueError("ReAct final must be a non-empty string.")
        action = data.get("action")
        if isinstance(action, dict):
            name = str(action.get("name") or "").strip()
            input_data = action.get("input") or {}
            if name and isinstance(input_data, dict):
                return ReactStep(thought=thought, action=CapabilityCall(name, input_data))
        raise ValueError("ReAct step must contain a valid action or final.")

    def available_capability_metadata(self, context: AgentContext) -> dict[str, Any]:
        sources = (context.skills, context.tools, context.mcp_tools)
        available: dict[str, Any] = {}
        for name in sorted(self.available_capability_names()):
            for source in sources:
                if name in source:
                    available[name] = source[name]
                    break
        return available

    async def generate_react_final(
        self,
        context: AgentContext,
        task: AgentTask,
        formatted_input: str,
        observations: list[CapabilityCallResult],
    ) -> str:
        raw = await self.process_with_llm(
            self.react_step_prompt(context, task, formatted_input, observations, force_final=True)
        )
        step = self.parse_react_step(raw)
        if step.final is None:
            raise ValueError("Forced ReAct final did not contain final.")
        return step.final

    def parse_capability_calls(self, raw: str) -> list[CapabilityCall]:
        data = self._json_object(raw, "Capability plan")
        raw_calls = data.get("calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("Capability plan calls must be a list.")
        calls: list[CapabilityCall] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            input_data = item.get("input") or {}
            if name and isinstance(input_data, dict):
                calls.append(CapabilityCall(name, input_data))
        return calls

    async def execute_capability_calls(
        self,
        context: AgentContext,
        task: AgentTask,
        calls: list[CapabilityCall],
    ) -> list[CapabilityCallResult]:
        if self.capability_executor is None:
            return []
        runtime_context = dict(context.capability_context)
        runtime_context.setdefault("history", context.history)
        runtime_context.setdefault("long_memory", context.long_memory)
        runtime_context.setdefault("session_id", context.public_session_id)
        runtime_context.setdefault("task", task)
        return await self.capability_executor.execute_many(
            runtime_context,
            calls,
            self.available_capability_names(),
        )

    def format_capability_observations(self, results: list[CapabilityCallResult]) -> str:
        if not results:
            return ""
        lines = []
        for result in results:
            payload = json.dumps(result.output, ensure_ascii=False, default=str)
            error = f" error={result.error}" if result.error else ""
            lines.append(f"- {result.name} [{result.status.value}]{error}: {payload}")
        return "\n".join(lines)

    def _capability_metadata(
        self,
        calls: list[CapabilityCall],
        results: list[CapabilityCallResult],
        plan_error: str | None,
        observation_text: str,
    ) -> dict:
        return {
            "capabilityCalls": [call.to_dict() for call in calls],
            "capabilityResults": [result.to_dict() for result in results],
            "capabilityPlanError": plan_error,
            "capabilityObservationText": observation_text,
        }

    def _react_metadata(
        self,
        trace: list[dict[str, Any]],
        stopped_reason: str,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "reactTrace": trace,
            "reactStepCount": len(trace),
            "reactStoppedReason": stopped_reason,
            "reactError": error,
        }

    def _can_execute_capabilities(self) -> bool:
        return self.capability_executor is not None and bool(self.available_capability_names())

    def _max_react_actions(self) -> int:
        if self.capability_executor is None:
            return 0
        return max(0, self.capability_executor.max_calls)

    def _final_input(self, formatted_input: str, observations: list[CapabilityCallResult]) -> str:
        observation_text = self.format_capability_observations(observations)
        if not observation_text:
            return formatted_input
        return (
            f"{formatted_input}\n\n能力调用观察结果：\n{observation_text}\n\n"
            "请基于用户输入、上下文和以上能力调用结果，生成最终结构化回答。"
        )

    def _json_object(self, raw: str, label: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"{label} did not contain JSON.")
        data = json.loads(raw[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError(f"{label} JSON must be an object.")
        return data

    def parse_result(
        self,
        task: AgentTask,
        raw: str,
        formatted_input: str | None = None,
    ) -> AgentTaskResult:
        return AgentTaskResult(
            task.id,
            self.name,
            AgentTaskStatus.SUCCESS,
            raw,
            metadata={"taskType": task.task_type, "formattedInput": formatted_input or ""},
        )
