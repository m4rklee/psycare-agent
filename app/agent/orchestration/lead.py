import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.capabilities import CapabilityExecutor
from app.agent.communication.queue import AgentTaskQueue
from app.agent.communication.types import AgentContext, AgentTask, AgentTaskResult, AgentTaskStatus
from app.agent.orchestration.agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from app.services.ai import AiClient, AiMessage


class LeadDecompositionError(RuntimeError):
    pass


class LeadDecompositionParseError(ValueError):
    pass


@dataclass(frozen=True)
class LeadSubtask:
    description: str
    assigned_agent: str


@dataclass(frozen=True)
class LeadTaskDecomposition:
    subtasks: list[LeadSubtask]
    raw: str | None = None


@dataclass(frozen=True)
class LeadAgentResult:
    run_id: str
    complex_task: bool
    dispatched_agents: list[str]
    results: list[AgentTaskResult]
    summary: str
    timeout_occurred: bool
    decomposition_error: str | None = None


class LeadAgent:
    name = "lead"
    EXTERNAL_TO_INTERNAL = {
        "consultation_agent": ConsultationAgent.name,
        "diagnostic_agent": DiagnosticAgent.name,
        "research_agent": ResearchAgent.name,
    }

    def __init__(
        self,
        ai_client: AiClient | None = None,
        queue: AgentTaskQueue | None = None,
        timeout_seconds: float = 30.0,
        capability_executor: CapabilityExecutor | None = None,
    ) -> None:
        self.ai_client = ai_client
        self.queue = queue or AgentTaskQueue()
        self.timeout_seconds = timeout_seconds
        self.capability_executor = capability_executor
        self.agents = {
            ConsultationAgent.name: ConsultationAgent(ai_client, capability_executor),
            DiagnosticAgent.name: DiagnosticAgent(ai_client, capability_executor),
            ResearchAgent.name: ResearchAgent(ai_client, capability_executor),
        }

    def role_prompt(self) -> str:
        return """你是校园心理健康 Swarm 的 Lead Agent。你的职责是分析用户问题并分配给合适的 Worker Agent。

核心原则：
1. 尽量少分配任务：能用 1 个 Agent 解决的，不要用 2 个；能用 2 个的，不要用 3 个。
2. 优先使用 ConsultationAgent：对于常见心理困扰、情绪倾诉、压力、睡眠、人际困扰、轻度低落和健康科普，单独使用 ConsultationAgent 就足够。
3. 你只负责分配 Agent，不决定具体使用哪些工具/技能，Worker Agent 会自己选择。
4. 子任务应该相对独立，可以并行执行。

可用 Worker Agents：

1. ConsultationAgent（心理健康咨询支持）
擅长：情绪倾听、初步问询、常见心理困扰支持、生活方式建议、校园支持资源引导。
适用：焦虑、压力、睡眠、人际关系、学习压力、轻度情绪困扰、一般心理健康科普。

2. DiagnosticAgent（心理风险与模式分析）
擅长：复杂困扰模式分析、风险等级线索整理、明显加重或功能受损问题梳理、需要进一步求助的紧急程度判断。
适用：出现自伤/伤人/危机信号、明显加重或严重功能受损、多个困扰交织且用户明确询问风险或严重程度。

3. ResearchAgent（心理健康知识与证据支持）
擅长：权威心理健康知识、校园危机干预原则、心理支持方法、研究证据和资源建议。
适用：询问标准支持方法、危机处理原则、校园资源、研究证据或需要权威资料支持的问题。

任务分配策略：
简单问题 -> 1 个 Agent：ConsultationAgent。普通低落、压力、睡眠、人际、学习困扰，即使提到持续一段时间，也优先单 Agent 支持。
复杂风险或复杂困扰 -> 2 个 Agents：DiagnosticAgent + ConsultationAgent。只有明确高风险、明显加重/严重功能受损，或多个困扰交织且明确询问风险/严重程度时使用。
需要权威知识/资源/证据 -> 1 个 Agent：ResearchAgent。只有同时需要情绪支持时再加 ConsultationAgent。

输出严格 JSON，不要输出 Markdown，不要解释，不要包含 type 字段：
{
  "subtasks": [
    {
      "description": "具体说明这个 Agent 需要做什么",
      "assigned_agent": "consultation_agent"
    }
  ]
}

assigned_agent 只能是 consultation_agent、diagnostic_agent、research_agent。"""

    async def run(
        self,
        db: AsyncSession,
        context: AgentContext,
        user_input: str,
        multimodal_summary: str | None = None,
        summarize: bool = True,
    ) -> LeadAgentResult:
        run_id = str(uuid.uuid4())
        try:
            decomposition = await self.decompose_task(user_input, context)
            decomposition_error = None
        except LeadDecompositionError as exc:
            return LeadAgentResult(
                run_id=run_id,
                complex_task=False,
                dispatched_agents=[],
                results=[],
                summary="评估失败",
                timeout_occurred=False,
                decomposition_error=str(exc),
            )
        except LeadDecompositionParseError as exc:
            decomposition = self.fallback_decomposition(user_input)
            decomposition_error = str(exc)

        tasks = await self.create_tasks(db, context, decomposition, user_input, run_id, multimodal_summary)
        results, timeout_occurred = await self.wait_for_tasks(
            db,
            context,
            tasks,
            run_id,
            timeout_seconds=self.timeout_seconds,
        )
        summary = ""
        if summarize:
            summary = (
                results[0].content
                if self.should_return_single_agent_result(tasks, results, timeout_occurred)
                else await self.summarize_results(user_input, results, timeout_occurred)
            )
        dispatched = [task.assigned_agent for task in tasks]
        return LeadAgentResult(
            run_id=run_id,
            complex_task=len(tasks) > 1,
            dispatched_agents=dispatched,
            results=results,
            summary=summary,
            timeout_occurred=timeout_occurred,
            decomposition_error=decomposition_error,
        )

    def should_return_single_agent_result(
        self,
        tasks: list[AgentTask],
        results: list[AgentTaskResult],
        timeout_occurred: bool,
    ) -> bool:
        return (
            len(tasks) == 1
            and len(results) == 1
            and results[0].status == AgentTaskStatus.SUCCESS
            and bool(results[0].content.strip())
            and not timeout_occurred
        )

    async def decompose_task(self, user_input: str, context: AgentContext) -> LeadTaskDecomposition:
        if self.ai_client is None:
            raise LeadDecompositionError("LLM client is not configured.")
        try:
            raw = await self.ai_client.complete(
                [
                    AiMessage("system", self.role_prompt()),
                    AiMessage(
                        "user",
                        "请为下面的用户问题分配 Worker Agent。\n\n"
                        f"最近上下文：\n{self._format_history(context)}\n\n用户问题：\n{user_input}",
                    ),
                ]
            )
        except Exception as exc:
            raise LeadDecompositionError(str(exc)) from exc
        try:
            return self.parse_decomposition(raw)
        except LeadDecompositionParseError:
            raise
        except Exception as exc:
            raise LeadDecompositionParseError(str(exc)) from exc

    def parse_decomposition(self, raw: str) -> LeadTaskDecomposition:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise LeadDecompositionParseError("Lead decomposition did not contain JSON.")
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LeadDecompositionParseError(f"Invalid Lead decomposition JSON: {exc}") from exc
        subtasks_raw = payload.get("subtasks")
        if not isinstance(subtasks_raw, list) or not subtasks_raw:
            raise LeadDecompositionParseError("Lead decomposition requires non-empty subtasks.")
        subtasks: list[LeadSubtask] = []
        for item in subtasks_raw:
            if not isinstance(item, dict):
                raise LeadDecompositionParseError("Each subtask must be an object.")
            description = str(item.get("description") or "").strip()
            assigned_agent = str(item.get("assigned_agent") or "").strip()
            if not description or assigned_agent not in self.EXTERNAL_TO_INTERNAL:
                raise LeadDecompositionParseError("Subtask has invalid description or assigned_agent.")
            subtasks.append(LeadSubtask(description, assigned_agent))
        return LeadTaskDecomposition(subtasks, raw=raw)

    def fallback_decomposition(self, user_input: str) -> LeadTaskDecomposition:
        return LeadTaskDecomposition(
            [
                LeadSubtask(
                    "对用户当前心理健康问题进行初步问询和支持建议整理",
                    "consultation_agent",
                )
            ],
            raw=None,
        )

    async def create_tasks(
        self,
        db: AsyncSession,
        context: AgentContext,
        decomposition: LeadTaskDecomposition,
        user_input: str,
        run_id: str,
        multimodal_summary: str | None = None,
    ) -> list[AgentTask]:
        tasks: list[AgentTask] = []
        for subtask in decomposition.subtasks:
            internal_agent = self.EXTERNAL_TO_INTERNAL[subtask.assigned_agent]
            agent = self.agents[internal_agent]
            task = await self.queue.submit(
                db,
                AgentTask(
                    id=None,
                    session_id=context.session_id,
                    run_id=run_id,
                    assigned_agent=agent.name,
                    task_type=agent.task_type,
                    payload={
                        "run_id": run_id,
                        "user_input": user_input,
                        "subtask_description": subtask.description,
                        "external_agent": subtask.assigned_agent,
                        "multimodal_summary": multimodal_summary,
                    },
                ),
            )
            tasks.append(task)
        return tasks

    async def wait_for_tasks(
        self,
        db: AsyncSession,
        context: AgentContext,
        tasks: list[AgentTask],
        run_id: str,
        timeout_seconds: float = 30.0,
    ) -> tuple[list[AgentTaskResult], bool]:
        if not tasks:
            return [], False
        handlers = {name: agent.run for name, agent in self.agents.items()}
        results = await self.queue.run_all(db, context, handlers, timeout_seconds, run_id)
        timeout_occurred = any(result.error == "timeout" for result in results)
        return results, timeout_occurred

    async def summarize_results(
        self,
        user_input: str,
        results: list[AgentTaskResult],
        timeout_occurred: bool,
    ) -> str:
        if not results:
            return "评估失败"
        prompt = self.final_answer_prompt(user_input, results, timeout_occurred)
        if self.ai_client is None:
            return self._fallback_summary(results, timeout_occurred)
        try:
            summary = await self.ai_client.complete([AiMessage("user", prompt)])
            return summary.strip() or self._fallback_summary(results, timeout_occurred)
        except Exception:
            return self._fallback_summary(results, timeout_occurred)

    async def stream_final_answer(
        self,
        user_input: str,
        results: list[AgentTaskResult],
        timeout_occurred: bool,
    ) -> AsyncIterator[str]:
        if not results:
            yield "评估失败"
            return
        if self.ai_client is None:
            yield self._fallback_summary(results, timeout_occurred)
            return
        prompt = self.final_answer_prompt(user_input, results, timeout_occurred)
        async for token in self.ai_client.stream([AiMessage("user", prompt)]):
            if token:
                yield token

    def final_answer_prompt(
        self,
        user_input: str,
        results: list[AgentTaskResult],
        timeout_occurred: bool,
    ) -> str:
        if len(results) == 1 and results[0].status == AgentTaskStatus.SUCCESS and results[0].content.strip():
            return self._single_agent_final_prompt(user_input, results[0])
        return self._swarm_summary_prompt(user_input, results, timeout_occurred)

    def _single_agent_final_prompt(self, user_input: str, result: AgentTaskResult) -> str:
        metadata = json.dumps(self._summary_metadata(result.metadata), ensure_ascii=False, default=str)
        return f"""你是校园心理健康支持助手。请基于 Worker Agent 的结果，生成面向学生可直接阅读的最终回答。

用户问题：{user_input}

Worker Agent：{result.agent_name}

Worker 原始回答：
{result.content}

关键 metadata：
{metadata}

要求：
1. 保留 Worker 原始回答的核心信息和心理健康边界，不新增没有依据的结论。
2. 可以沿用 Worker 原始回答的模块结构，例如【回答】【核心建议】【免责声明】。
3. 语言要温和、清晰、可执行，适合校园心理咨询场景。
4. 不要输出 JSON，不要解释处理过程。
"""

    def _swarm_summary_prompt(
        self,
        user_input: str,
        results: list[AgentTaskResult],
        timeout_occurred: bool,
    ) -> str:
        contributions_text = self._format_contributions(results)
        timeout_note = "\n\n注意：部分分析模块未完成或超时。" if timeout_occurred else ""
        return f"""你是校园心理健康 Swarm 的 Lead Agent，负责汇总多个专业 Agent 的分析结果。

用户问题：{user_input}

Agent 贡献：
{contributions_text}{timeout_note}

任务：
整合以上所有分析，生成一个面向学生可直接阅读的最终回答。回答要先承接用户感受，再自然融合风险识别、心理支持分析、知识依据和行动建议。

要求：
1. 综合所有 Agent 的观点。
2. 不要把回答写成内部报告；优先使用温和、清晰、可执行的学生端表达。
3. 保持心理健康建议的谨慎性，不做医学诊断，不替代专业心理咨询。
4. 如果有风险、模式或证据信息，把它们整合到回答中；高风险时必须突出安全和现实支持。
5. 给出【核心建议】。
6. 添加【免责声明】。
{"7. 如果有分析模块未完成，在答案中明确说明。" if timeout_occurred else ""}

输出格式：
【回答】
...

【风险评估】（如果相关）
...

【心理支持分析】（如果相关）
...

【知识证据】（如果相关）
...

【核心建议】
1. ...
2. ...

【免责声明】
..."""

    def is_complex(self, user_input: str, multimodal_summary: str | None = None) -> bool:
        return bool(multimodal_summary) or len(user_input) > 120

    def _format_history(self, context: AgentContext) -> str:
        if not context.history:
            return "无"
        return "\n".join(f"{message.role}: {message.content}" for message in context.history[-10:])

    def _format_contributions(self, results: list[AgentTaskResult]) -> str:
        lines = []
        for result in results:
            status = result.status.value
            content = result.content or result.error or "无结果"
            contribution = {
                "agent_id": result.agent_name,
                "status": status,
                "answer": content,
                "metadata": self._summary_metadata(result.metadata),
            }
            lines.append(f"**{result.agent_name}** ({status}):\n{json.dumps(contribution, ensure_ascii=False)}")
        return "\n".join(lines)

    def _summary_metadata(self, metadata: dict) -> dict:
        keys = (
            "suggestions",
            "disclaimer",
            "risk_level",
            "diagnosis_provided",
            "evidence_level",
            "literature_count",
            "evidence_provided",
            "sourceCount",
            "evidenceStrength",
            "keywords",
        )
        return {key: metadata[key] for key in keys if key in metadata}

    def _fallback_summary(self, results: list[AgentTaskResult], timeout_occurred: bool) -> str:
        timeout_text = "\n部分分析模块未完成或超时。" if timeout_occurred else ""
        return f"{self._format_contributions(results)}{timeout_text}" or "评估失败"
