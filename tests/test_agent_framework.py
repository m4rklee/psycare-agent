import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.capabilities import CapabilityCall, CapabilityExecutor, CapabilityRegistry
from app.agent.communication.queue import AgentTaskQueue
from app.agent.communication.types import AgentContext, AgentTask, AgentTaskResult, AgentTaskStatus
from app.agent.memory.manager import LongTermMemoryManager
from app.agent.orchestration.agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from app.agent.orchestration.base import BaseAgent
from app.agent.orchestration.lead import LeadAgent
from app.agent.orchestration.postprocess import DEFAULT_PSYCHOLOGY_DISCLAIMER
from app.agent.skills import SkillLoadError, SkillLoader
from app.core.database import Base
from app.models.entities import AgentMemorySummary, AgentTaskRecord, ChatSession, PsychologicalReport, UserAccount
from app.models.enums import ToolStatus
from app.services.ai import AiMessage, HeuristicAiClient


class FailingAgent(BaseAgent):
    name = "diagnostic"
    task_type = "diagnostic_detail"

    def role_prompt(self) -> str:
        return "fail"

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        raise RuntimeError("diagnostic failed")


class SlowAgent(BaseAgent):
    name = "diagnostic"
    task_type = "diagnostic_detail"

    def role_prompt(self) -> str:
        return "slow"

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        import asyncio

        await asyncio.sleep(1)
        return AgentTaskResult(task.id, self.name, AgentTaskStatus.SUCCESS, "late")


class LeadAiClient:
    def __init__(self, decomposition: str | Exception, summary: str = "内部汇总") -> None:
        self.decomposition = decomposition
        self.summary = summary
        self.summary_calls = 0
        self.prompts: list[str] = []

    async def complete(self, messages: list[AiMessage]) -> str:
        prompt = "\n".join(message.content for message in messages)
        self.prompts.append(prompt)
        if "Agent ReAct 控制器" in prompt:
            if "Agent：diagnostic" in prompt:
                final = "【风险评估】\n风险等级：低\n紧急程度：可先观察。\n\n【模式分析】\n主要困扰类别：压力相关。\n线索关联性：需继续了解。\n\n【可能影响因素】\n1. 压力累积\n   - 支持线索：当前表达。\n   - 不确定点：持续时间。\n\n【建议进一步了解】\n- 持续多久？\n\n【推理过程】\n仅做心理风险与模式分析。"
            elif "Agent：research" in prompt:
                final = "【资料检索结果】\n关键词：校园心理支持\n找到相关资料：1 条\n\n【证据摘要】\n1. 校园心理支持原则\n   - 核心发现：连接现实支持。\n   - 证据强度：中\n   - 支持建议：联系学校心理中心。\n\n【综合评估】\n- 证据强度：中\n- 主要结论：早期支持有帮助。\n- 局限性：测试数据。\n- 建议：连接校园支持。\n\n【适用边界】\n不能替代专业心理咨询。"
            else:
                final = "【回答】\n先做初步心理支持。\n\n【核心建议】\n1. 记录困扰。\n2. 联系校园支持。\n\n【免责声明】\n本内容不能替代专业心理咨询。"
            return json.dumps({"thought": "测试 worker final", "final": final}, ensure_ascii=False)
        if "负责汇总多个专业 Agent" in prompt:
            self.summary_calls += 1
            return self.summary
        if isinstance(self.decomposition, Exception):
            raise self.decomposition
        return self.decomposition


class FailingAiClient:
    async def complete(self, messages: list[AiMessage]) -> str:
        raise RuntimeError("llm failed")


class BadCapabilityPlanAiClient:
    async def complete(self, messages: list[AiMessage]) -> str:
        prompt = "\n".join(message.content for message in messages)
        if "Agent ReAct 控制器" in prompt and "现在必须输出 final" not in prompt:
            return "not json"
        final = (
            "【回答】\n能力规划失败后仍然生成回答。\n\n"
            "【核心建议】\n1. 继续温和梳理当前困扰。\n\n"
            "【免责声明】\n本内容不能替代专业心理咨询。"
        )
        return json.dumps({"thought": "回退生成最终回答", "final": final}, ensure_ascii=False)


class McpCapabilityPlanAiClient:
    async def complete(self, messages: list[AiMessage]) -> str:
        prompt = "\n".join(message.content for message in messages)
        if (
            "Agent ReAct 控制器" in prompt
            and "multimodalAgent.excel.write_report" not in prompt.split("已有 Observation：", 1)[-1]
            and "现在必须输出 final" not in prompt
        ):
            return '{"thought":"测试未授权 MCP 工具的阻断路径","action":{"name":"multimodalAgent.excel.write_report","input":{"reportId":1}}}'
        final = (
            "【回答】\nMCP 写入工具已被阻断，最终回答不受影响。\n\n"
            "【核心建议】\n1. 不自动执行有副作用工具。\n\n"
            "【免责声明】\n本内容不能替代专业心理咨询。"
        )
        return json.dumps({"thought": "基于阻断观察生成最终回答", "final": final}, ensure_ascii=False)


class RecordingToolService:
    def __init__(self) -> None:
        self.report_ids: list[int] = []

    async def handle(self, session, report_id: int) -> None:
        self.report_ids.append(report_id)
        report = await session.get(PsychologicalReport, report_id)
        report.excel_status = ToolStatus.SUCCESS
        report.email_status = ToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_lead_agent_simple_input_dispatches_only_consultation(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        ai_client = LeadAiClient(
            '{"subtasks":[{"description":"初步问询和支持建议","assigned_agent":"consultation_agent"}]}'
        )
        lead = LeadAgent(
            ai_client=ai_client,
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "最近有点焦虑")
        await db.commit()
        rows = (await db.scalars(select(AgentTaskRecord))).all()

    assert result.complex_task is False
    assert len(result.run_id) == 36
    assert result.dispatched_agents == ["consultation"]
    assert result.summary == result.results[0].content
    assert "【回答】" in result.summary
    assert "【核心建议】" in result.summary
    assert "【免责声明】" in result.summary
    assert "先做初步心理支持" in result.summary
    assert "【回答】" in result.results[0].metadata["rawContent"]
    assert ai_client.summary_calls == 0
    assert [row.agent_name for row in rows] == ["consultation"]
    assert rows[0].run_id == result.run_id
    assert rows[0].status == AgentTaskStatus.SUCCESS.value


@pytest.mark.asyncio
async def test_lead_agent_identity_question_returns_consultation_answer_without_swarm_summary(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=HeuristicAiClient(),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "你是谁")

    assert result.dispatched_agents == ["consultation"]
    assert result.summary == result.results[0].content
    assert "【回答】" in result.summary
    assert "【核心建议】" in result.summary
    assert "【免责声明】" in result.summary
    assert "【风险评估】" not in result.summary
    assert "【心理支持分析】" not in result.summary
    assert "【知识证据】" not in result.summary


@pytest.mark.asyncio
async def test_lead_agent_low_mood_for_one_week_stays_single_consultation(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=HeuristicAiClient(),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "这周一直很低落，做什么都提不起劲。")

    assert result.complex_task is False
    assert result.dispatched_agents == ["consultation"]
    assert result.summary == result.results[0].content
    assert "【回答】" in result.summary
    assert "你现在的低落感值得被认真看见" in result.summary
    assert "【风险评估】" not in result.summary
    assert "【心理支持分析】" not in result.summary


@pytest.mark.asyncio
async def test_lead_agent_complex_input_dispatches_three_agents(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        ai_client = LeadAiClient(
            '{"subtasks":['
            '{"description":"评估风险等级","assigned_agent":"diagnostic_agent"},'
            '{"description":"提供支持建议","assigned_agent":"consultation_agent"}]}',
            summary=(
                "【回答】\n我会先认真接住你现在的状态，再综合风险和支持建议。\n\n"
                "【风险评估】\n暂未发现紧急风险。\n\n"
                "【心理支持分析】\n需要同时关注压力和睡眠。\n\n"
                "【核心建议】\n1. 记录情绪变化。\n\n"
                "【免责声明】\n旧免责声明。"
            ),
        )
        lead = LeadAgent(
            ai_client=ai_client,
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "我不想活了，需要分析症状和建议")
        await db.commit()

    assert result.complex_task is True
    assert len(result.run_id) == 36
    assert result.dispatched_agents == ["diagnostic", "consultation"]
    assert [item.agent_name for item in result.results] == ["diagnostic", "consultation"]
    assert all(item.status == AgentTaskStatus.SUCCESS for item in result.results)
    assert result.summary.startswith("【回答】")
    assert "【风险评估】" in result.summary
    assert "【心理支持分析】" in result.summary
    assert "【核心建议】" in result.summary
    assert "【免责声明】" in result.summary
    assert "暂未发现紧急风险" in result.summary
    assert "需要同时关注压力和睡眠" in result.summary
    assert "旧免责声明" in result.summary
    assert ai_client.summary_calls == 1
    summary_prompt = next(prompt for prompt in ai_client.prompts if "负责汇总多个专业 Agent" in prompt)
    assert '"risk_level": "low"' in summary_prompt
    assert '"diagnosis_provided": true' in summary_prompt
    assert '"suggestions": ["记录困扰。", "联系校园支持。"]' in summary_prompt


@pytest.mark.asyncio
async def test_lead_agent_single_research_result_is_returned_directly(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        ai_client = LeadAiClient(
            '{"subtasks":[{"description":"整理校园心理资源和证据","assigned_agent":"research_agent"}]}',
            summary="不应该调用汇总",
        )
        lead = LeadAgent(ai_client=ai_client, queue=AgentTaskQueue(), timeout_seconds=1)

        result = await lead.run(db, context_for(chat_session), "学校心理危机干预原则有哪些？")

    assert result.complex_task is False
    assert result.dispatched_agents == ["research"]
    assert result.summary == result.results[0].content
    assert "【资料检索结果】" in result.summary
    assert ai_client.summary_calls == 0


@pytest.mark.asyncio
async def test_lead_agent_single_diagnostic_result_is_returned_directly(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        ai_client = LeadAiClient(
            '{"subtasks":[{"description":"单独评估心理风险模式","assigned_agent":"diagnostic_agent"}]}',
            summary="不应该调用汇总",
        )
        lead = LeadAgent(ai_client=ai_client, queue=AgentTaskQueue(), timeout_seconds=1)

        result = await lead.run(db, context_for(chat_session), "请帮我评估这个状态的风险。")

    assert result.complex_task is False
    assert result.dispatched_agents == ["diagnostic"]
    assert result.summary == result.results[0].content
    assert "【风险评估】" in result.summary
    assert ai_client.summary_calls == 0


@pytest.mark.asyncio
async def test_lead_agent_research_mapping_uses_internal_agent_name(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=LeadAiClient(
                '{"subtasks":['
                '{"description":"整理校园心理危机干预原则","assigned_agent":"research_agent"},'
                '{"description":"提供沟通支持建议","assigned_agent":"consultation_agent"}]}'
            ),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "学校心理危机干预原则有哪些？")

    assert result.dispatched_agents == ["research", "consultation"]
    assert [item.agent_name for item in result.results] == ["research", "consultation"]


@pytest.mark.asyncio
async def test_lead_agent_llm_failure_returns_assessment_failed_without_tasks(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(ai_client=LeadAiClient(RuntimeError("client unavailable")), queue=AgentTaskQueue())

        result = await lead.run(db, context_for(chat_session), "最近有点焦虑")
        rows = (await db.scalars(select(AgentTaskRecord))).all()

    assert result.summary == "评估失败"
    assert len(result.run_id) == 36
    assert result.results == []
    assert rows == []
    assert result.decomposition_error == "client unavailable"


@pytest.mark.asyncio
async def test_lead_agent_invalid_json_falls_back_to_consultation(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(ai_client=LeadAiClient("not json"), queue=AgentTaskQueue(), timeout_seconds=1)

        result = await lead.run(db, context_for(chat_session), "最近有点焦虑")
        rows = (await db.scalars(select(AgentTaskRecord))).all()

    assert result.dispatched_agents == ["consultation"]
    assert result.decomposition_error is not None
    assert rows[0].run_id == result.run_id
    assert rows[0].agent_name == "consultation"


@pytest.mark.asyncio
async def test_agent_queue_collects_sub_agent_failure_without_raising(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=LeadAiClient(
                '{"subtasks":['
                '{"description":"评估风险等级","assigned_agent":"diagnostic_agent"},'
                '{"description":"提供支持建议","assigned_agent":"consultation_agent"}]}'
            ),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )
        lead.agents["diagnostic"] = FailingAgent()

        result = await lead.run(db, context_for(chat_session), "我不想活了，需要分析症状和建议")
        await db.commit()
        rows = (await db.scalars(select(AgentTaskRecord).order_by(AgentTaskRecord.agent_name))).all()

    failed = [item for item in result.results if item.status == AgentTaskStatus.FAILED]
    assert failed[0].agent_name == "diagnostic"
    assert failed[0].error == "diagnostic failed"
    assert any(row.status == AgentTaskStatus.FAILED.value for row in rows)


@pytest.mark.asyncio
async def test_agent_queue_marks_worker_timeout_in_lead_result(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=LeadAiClient(
                '{"subtasks":[{"description":"评估风险等级","assigned_agent":"diagnostic_agent"}]}'
            ),
            queue=AgentTaskQueue(),
            timeout_seconds=0.01,
        )
        lead.agents["diagnostic"] = SlowAgent()

        result = await lead.run(db, context_for(chat_session), "我不想活了，需要分析症状和建议")

    assert result.timeout_occurred is True
    assert result.results[0].status == AgentTaskStatus.FAILED
    assert result.results[0].error == "timeout"


@pytest.mark.asyncio
async def test_agent_queue_runs_only_matching_run_id_and_retains_others(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    queue = AgentTaskQueue()
    async with session_factory() as db:
        chat_session = await seed_session(db)
        context = context_for(chat_session)
        await queue.submit(
            db,
            AgentTask(
                id=None,
                session_id=chat_session.id,
                run_id="run-a",
                assigned_agent="consultation",
                task_type="consultation_intake",
                payload={"user_input": "run a"},
            ),
        )
        await queue.submit(
            db,
            AgentTask(
                id=None,
                session_id=chat_session.id,
                run_id="run-b",
                assigned_agent="consultation",
                task_type="consultation_intake",
                payload={"user_input": "run b"},
            ),
        )

        async def handler(_context: AgentContext, task: AgentTask) -> AgentTaskResult:
            return AgentTaskResult(task.id, task.assigned_agent, AgentTaskStatus.SUCCESS, task.payload["user_input"])

        run_a_results = await queue.run_all(
            db,
            context,
            {"consultation": handler},
            timeout_seconds=1,
            run_id="run-a",
        )
        rows_after_a = (
            await db.scalars(select(AgentTaskRecord).order_by(AgentTaskRecord.run_id))
        ).all()
        row_state_after_a = [(row.run_id, row.status) for row in rows_after_a]
        run_b_results = await queue.run_all(
            db,
            context,
            {"consultation": handler},
            timeout_seconds=1,
            run_id="run-b",
        )

    assert [result.content for result in run_a_results] == ["run a"]
    assert [run_id for run_id, _status in row_state_after_a] == ["run-a", "run-b"]
    assert [status for _run_id, status in row_state_after_a] == [
        AgentTaskStatus.SUCCESS.value,
        AgentTaskStatus.PENDING.value,
    ]
    assert [result.content for result in run_b_results] == ["run b"]


@pytest.mark.asyncio
async def test_long_term_memory_summary_is_deduped_by_hash(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        manager = LongTermMemoryManager()

        first = await manager.save_summary(db, chat_session.id, " 用户说最近压力很大 ")
        second = await manager.save_summary(db, chat_session.id, "用户说最近压力很大")
        await db.commit()
        count = len((await db.scalars(select(AgentMemorySummary))).all())

    assert first.id == second.id
    assert count == 1


def test_skill_loader_reads_markdown_and_async_handler(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (skill_dir / "skill.py").write_text(
        'metadata = {"name": "demo", "description": "Demo skill"}\n'
        "async def run(context, input):\n"
        '    return {"ok": True}\n',
        encoding="utf-8",
    )

    skills = SkillLoader(tmp_path / "skills").load_all()

    assert skills["demo"].instructions == "# demo\n"
    assert skills["demo"].metadata["description"] == "Demo skill"


def test_project_placeholder_skills_are_loadable() -> None:
    skills = SkillLoader("skills").load_all()
    expected = {
        "search_knowledge",
        "recommend_lifestyle",
        "assess_risk",
        "analyze_symptoms",
        "disease_code",
        "clinical_guideline",
        "deep_research",
        "search_history",
        "search_similar_cases",
    }

    assert expected.issubset(set(skills))


def test_capability_registry_loads_skills_and_registers_tool_metadata() -> None:
    registry = CapabilityRegistry("skills")

    assert "assess_risk" in registry.skill_metadata()
    assert "search_knowledge" in registry.skill_metadata()
    assert registry.tool_metadata()["report.create_and_dispatch"]["autoCallable"] is True
    assert "report.excel_writer" in registry.tool_metadata()
    assert "multimodalAgent.excel.write_report" in registry.mcp_metadata()
    assert registry.mcp_metadata()["multimodalAgent.excel.write_report"]["autoCallable"] is False


class FakeKnowledgeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, db, query: str, top_k: int):
        self.calls.append((query, top_k))
        return [
            SimpleNamespace(
                chunk_id=1,
                source="campus-policy.md",
                content=f"{query}：学校心理中心提供预约咨询、危机支持和辅导员联动资源。",
                score=0.91,
            ),
            SimpleNamespace(
                chunk_id=2,
                source="coping-guide.md",
                content="压力和睡眠困扰可先使用呼吸放松、作息稳定和现实支持。",
                score=0.72,
            ),
        ][:top_k]


@pytest.mark.asyncio
async def test_project_skills_run_as_real_async_handlers() -> None:
    skills = SkillLoader("skills").load_all()
    expected = {
        "search_knowledge",
        "recommend_lifestyle",
        "assess_risk",
        "analyze_symptoms",
        "disease_code",
        "clinical_guideline",
        "deep_research",
        "search_history",
        "search_similar_cases",
    }
    context = {
        "db": object(),
        "knowledge_service": FakeKnowledgeService(),
        "session_id": "session-1",
        "history": [
            AiMessage("user", "我最近压力很大，晚上睡不着"),
            AiMessage("assistant", "我们先看看睡眠和压力的触发点。"),
        ],
        "long_memory": ["用户曾提到考试压力和睡眠下降，需要校园心理支持。"],
    }
    input_data = {"user_input": "我最近焦虑睡不着，想了解学校心理危机干预资源", "query": "焦虑 睡眠 校园心理支持"}

    for name in expected:
        result = await skills[name].run(context, input_data)
        result_text = str(result)
        assert result["status"] == "success"
        assert result["skill"] == name
        assert "placeholder" not in result_text.lower()
        assert "暂无真实" not in result_text


@pytest.mark.asyncio
async def test_knowledge_skills_use_context_knowledge_service() -> None:
    skills = SkillLoader("skills").load_all()
    knowledge = FakeKnowledgeService()
    context = {"db": object(), "knowledge_service": knowledge}

    search_result = await skills["search_knowledge"].run(context, {"query": "焦虑 睡眠", "max_results": 2})
    guideline_result = await skills["clinical_guideline"].run(context, {"query": "危机干预", "max_results": 1})
    research_result = await skills["deep_research"].run(context, {"query": "校园心理支持", "max_results": 2})

    assert search_result["total_found"] == 2
    assert "campus-policy.md" in search_result["answer"]
    assert guideline_result["guideline_type"] == "campus_mental_health_support"
    assert "危机干预 校园心理支持" in knowledge.calls[1][0]
    assert research_result["status"] == "success"
    assert research_result["total_found"] >= 1
    assert research_result["evidence_strength"] == "中"
    assert len(research_result["queries"]) == 3


@pytest.mark.asyncio
async def test_capability_executor_executes_skill_and_blocks_mcp() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    context = {
        "db": object(),
        "knowledge_service": FakeKnowledgeService(),
        "history": [AiMessage("user", "我最近焦虑睡不着")],
        "long_memory": [],
        "session_id": "session-1",
    }

    executed = await executor.execute(
        context,
        CapabilityCall("assess_risk", {"user_input": "我不想活了"}),
        {"assess_risk"},
    )
    blocked = await executor.execute(
        context,
        CapabilityCall("multimodalAgent.excel.write_report", {"reportId": 1}),
        {"multimodalAgent.excel.write_report"},
    )
    unknown = await executor.execute(context, CapabilityCall("missing_skill", {}), {"missing_skill"})

    assert executed.status.value == "SUCCESS"
    assert executed.output["risk_level"] == "emergency"
    assert blocked.status.value == "BLOCKED"
    assert "not auto-callable" in (blocked.error or "")
    assert unknown.status.value == "BLOCKED"
    assert "Unknown capability" in (unknown.error or "")


@pytest.mark.asyncio
async def test_knowledge_skills_fail_soft_without_dependencies() -> None:
    skills = SkillLoader("skills").load_all()

    for name in ("search_knowledge", "clinical_guideline", "deep_research"):
        result = await skills[name].run({}, {"query": "焦虑"})
        assert result["status"] == "unavailable"
        assert "db" in result["error"]
        assert result["skill"] == name


@pytest.mark.asyncio
async def test_rule_skills_return_mental_health_structures() -> None:
    skills = SkillLoader("skills").load_all()

    risk = await skills["assess_risk"].run({}, {"user_input": "我不想活了，感觉撑不下去"})
    analysis = await skills["analyze_symptoms"].run({}, {"user_input": "我焦虑、睡不着，也不想上课"})
    category = await skills["disease_code"].run({}, {"user_input": "最近焦虑失眠，考试压力很大"})
    lifestyle = await skills["recommend_lifestyle"].run({}, {"user_input": "考试压力大，晚上睡不着"})

    assert risk["risk_level"] == "emergency"
    assert "现实支持" in risk["recommendation"]
    assert any("睡眠" in item for item in analysis["patterns"])
    assert any("学习" in item for item in analysis["patterns"])
    assert category["code"] == "PSY-STRESS-SLEEP"
    assert category["diagnostic"] is False
    assert "ICD" not in category["answer"]
    assert "sleep" in lifestyle["categories"]
    assert "study_stress" in lifestyle["categories"]


@pytest.mark.asyncio
async def test_memory_skills_search_context_history_and_long_memory() -> None:
    skills = SkillLoader("skills").load_all()
    context = {
        "session_id": "session-1",
        "history": [
            AiMessage("user", "我最近考试压力很大"),
            AiMessage("assistant", "你提到压力和睡眠都有影响。"),
            AiMessage("user", "和室友关系也很紧张"),
        ],
        "long_memory": [
            "用户曾讨论考试压力和睡眠下降，后续建议联系学校心理中心。",
            "用户曾提到社交孤独和室友沟通困难。",
        ],
    }

    history = await skills["search_history"].run(context, {"query": "室友", "limit": 2})
    similar = await skills["search_similar_cases"].run(context, {"query": "考试 睡眠", "max_results": 1})

    assert history["total_messages"] == 3
    assert history["total_found"] == 1
    assert "室友关系" in history["answer"]
    assert similar["total_found"] == 1
    assert "考试压力" in similar["answer"]


def test_skill_loader_reports_missing_handler(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# broken\n", encoding="utf-8")

    with pytest.raises(SkillLoadError, match="Missing skill.py"):
        SkillLoader(tmp_path / "skills").load_all()


def test_consultation_agent_prompt_uses_mental_health_boundary() -> None:
    prompt = ConsultationAgent().role_prompt()

    assert "校园心理健康咨询支持顾问" in prompt
    assert "search_knowledge" in prompt
    assert "search_similar_cases" in prompt
    assert "ICD-10" not in prompt
    assert "医生诊断" not in prompt


def test_consultation_agent_format_input_includes_session_context() -> None:
    agent = ConsultationAgent()
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[AiMessage("user", "我最近压力很大")],
        long_memory=["用户曾提到睡眠质量下降"],
    )
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={
            "user_input": "我最近焦虑睡不着",
            "subtask_description": "提供初步心理支持",
        },
    )

    formatted = agent.format_input(context, task)

    assert "会话ID：public-session" in formatted
    assert "子任务：提供初步心理支持" in formatted
    assert "最近上下文：" in formatted
    assert "user: 我最近压力很大" in formatted
    assert "长期记忆：" in formatted
    assert "用户曾提到睡眠质量下降" in formatted
    assert "用户问题：" in formatted
    assert "我最近焦虑睡不着" in formatted


def test_diagnostic_agent_prompt_uses_mental_health_boundary() -> None:
    prompt = DiagnosticAgent().role_prompt()

    assert "校园心理风险与模式分析 Agent" in prompt
    assert "assess_risk" in prompt
    assert "search_similar_cases" in prompt
    assert "ICD-10" not in prompt
    assert "VINDICATE" not in prompt
    assert "确诊" not in prompt


def test_diagnostic_agent_format_input_includes_session_context_and_contributions() -> None:
    agent = DiagnosticAgent()
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[AiMessage("user", "我最近压力很大")],
        long_memory=["用户曾提到睡眠质量下降"],
    )
    task = AgentTask(
        id=8,
        session_id=1,
        run_id="test-run",
        assigned_agent="diagnostic",
        task_type="diagnostic_detail",
        payload={
            "user_input": "我最近焦虑越来越严重",
            "subtask_description": "评估心理风险和困扰模式",
            "agent_contributions": ["consultation: 初步关注睡眠和压力"],
        },
    )

    formatted = agent.format_input(context, task)

    assert "会话ID：public-session" in formatted
    assert "子任务：评估心理风险和困扰模式" in formatted
    assert "最近上下文：" in formatted
    assert "user: 我最近压力很大" in formatted
    assert "长期记忆：" in formatted
    assert "用户曾提到睡眠质量下降" in formatted
    assert "其他 Agent 贡献：" in formatted
    assert "consultation: 初步关注睡眠和压力" in formatted
    assert "用户问题：" in formatted
    assert "我最近焦虑越来越严重" in formatted


def test_research_agent_prompt_uses_mental_health_evidence_boundary() -> None:
    prompt = ResearchAgent().role_prompt()

    assert "校园心理健康知识与证据支持 Agent" in prompt
    assert "clinical_guideline" in prompt
    assert "deep_research" in prompt
    assert "【资料检索结果】" in prompt
    assert "ICD-10" not in prompt
    assert "确诊" not in prompt
    assert "治疗决策" not in prompt


def test_research_agent_format_input_includes_session_context_and_contributions() -> None:
    agent = ResearchAgent()
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[AiMessage("user", "我最近压力很大")],
        long_memory=["用户曾提到睡眠质量下降"],
    )
    task = AgentTask(
        id=9,
        session_id=1,
        run_id="test-run",
        assigned_agent="research",
        task_type="symptom_research",
        payload={
            "user_input": "学校心理危机干预原则有哪些？",
            "subtask_description": "整理校园心理危机干预原则和证据依据",
            "agent_contributions": ["diagnostic: 关注安全风险和现实支持"],
        },
    )

    formatted = agent.format_input(context, task)

    assert "会话ID：public-session" in formatted
    assert "子任务：整理校园心理危机干预原则和证据依据" in formatted
    assert "最近上下文：" in formatted
    assert "user: 我最近压力很大" in formatted
    assert "长期记忆：" in formatted
    assert "用户曾提到睡眠质量下降" in formatted
    assert "其他 Agent 贡献：" in formatted
    assert "diagnostic: 关注安全风险和现实支持" in formatted
    assert "用户问题：" in formatted
    assert "学校心理危机干预原则有哪些？" in formatted


@pytest.mark.asyncio
async def test_consultation_agent_consult_uses_llm_and_extracts_sections() -> None:
    agent = ConsultationAgent(HeuristicAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
    )
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "我最近焦虑睡不着"},
    )

    result = await agent.consult(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert "【回答】" in result.content
    assert "【核心建议】" in result.content
    assert "【免责声明】" in result.content
    assert "我理解你现在想处理的是" in result.content
    assert "【回答】" in result.metadata["rawContent"]
    assert "焦虑" in result.metadata["answer"]
    assert "学校心理中心" in result.metadata["coreSuggestions"]
    assert result.metadata["suggestions"] == [
        "先记录最困扰你的具体场景，以及它出现的频率和持续时间。",
        "做一个短暂稳定动作：放慢呼吸、离开刺激源，或联系一个现实中能回应你的人。",
        "如果困扰持续两周以上，或明显影响睡眠、上课、饮食，建议尽快联系学校心理中心。",
    ]
    assert "不能替代专业心理咨询" in result.metadata["disclaimer"]
    assert "search_knowledge" in result.metadata["availableSkills"]
    assert "会话ID：public-session" in result.metadata["formattedInput"]


def test_consultation_agent_parse_result_uses_default_disclaimer_when_missing() -> None:
    agent = ConsultationAgent()
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "我最近压力大"},
    )
    raw = "【回答】\n可以先把压力来源写下来。\n\n【核心建议】\n1. 先休息。\n2. 找同学聊聊。"

    result = agent.parse_result(task, raw, "formatted")

    assert result.content == raw
    assert result.metadata["suggestions"] == ["先休息。", "找同学聊聊。"]
    assert result.metadata["disclaimer"] == DEFAULT_PSYCHOLOGY_DISCLAIMER


@pytest.mark.asyncio
async def test_diagnostic_agent_diagnose_uses_llm_and_extracts_sections() -> None:
    agent = DiagnosticAgent(HeuristicAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
        long_memory=["用户曾提到近期睡眠质量下降"],
    )
    task = AgentTask(
        id=8,
        session_id=1,
        run_id="test-run",
        assigned_agent="diagnostic",
        task_type="diagnostic_detail",
        payload={"user_input": "我最近焦虑越来越严重，已经持续两周"},
    )

    result = await agent.diagnose(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert "【风险评估】" in result.content
    assert "【模式分析】" in result.content
    assert "风险等级：中" in result.content
    assert "【风险评估】" in result.metadata["rawContent"]
    assert result.metadata["riskLevel"] == "中"
    assert result.metadata["risk_level"] == "medium"
    assert result.metadata["diagnosis_provided"] is True
    assert "学校心理中心" in result.metadata["urgency"]
    assert "压力、焦虑" in result.metadata["patternAnalysis"]
    assert "压力累积" in result.metadata["possibleFactors"]
    assert "持续了多久" in result.metadata["followUpQuestions"]
    assert "不做诊断结论" in result.metadata["reasoning"]
    assert "assess_risk" in result.metadata["availableSkills"]
    assert "会话ID：public-session" in result.metadata["formattedInput"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("【风险评估】\n风险等级：高\n紧急程度：尽快求助。", "high"),
        ("【风险评估】\n风险等级：紧急\n紧急程度：立即求助。", "high"),
        ("【风险评估】\n风险等级：中\n紧急程度：建议评估。", "medium"),
        ("【风险评估】\n风险等级：低\n紧急程度：可观察。", "low"),
        ("【模式分析】\n未提及风险等级。", "unknown"),
    ],
)
def test_diagnostic_agent_parse_result_uses_medix_style_risk_scan(raw: str, expected: str) -> None:
    agent = DiagnosticAgent()
    task = AgentTask(
        id=8,
        session_id=1,
        run_id="test-run",
        assigned_agent="diagnostic",
        task_type="diagnostic_detail",
        payload={"user_input": "测试"},
    )

    result = agent.parse_result(task, raw, "formatted")

    assert result.content == raw
    assert result.metadata["risk_level"] == expected
    assert result.metadata["diagnosis_provided"] is True


@pytest.mark.asyncio
async def test_research_agent_research_uses_llm_and_extracts_sections() -> None:
    agent = ResearchAgent(HeuristicAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
        long_memory=["用户曾提到近期睡眠质量下降"],
    )
    task = AgentTask(
        id=9,
        session_id=1,
        run_id="test-run",
        assigned_agent="research",
        task_type="symptom_research",
        payload={"user_input": "学校心理危机干预原则有哪些？"},
    )

    result = await agent.research(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert "【资料检索结果】" in result.content
    assert "【证据摘要】" in result.content
    assert "【综合评估】" in result.content
    assert "关键词：校园心理支持" in result.content
    assert "【资料检索结果】" in result.metadata["rawContent"]
    assert result.metadata["sourceCount"] == 3
    assert result.metadata["evidenceStrength"] == "中"
    assert result.metadata["evidence_level"] == "unknown"
    assert result.metadata["literature_count"] == result.content.count("文献") + result.content.count("资料")
    assert result.metadata["evidence_provided"] is True
    assert "校园心理支持" in result.metadata["keywords"]
    assert "学校心理中心" in result.metadata["evidenceSummary"]
    assert "早期识别" in result.metadata["mainConclusion"]
    assert "未接入真实检索" in result.metadata["limitations"]
    assert "校园支持" in result.metadata["recommendations"]
    assert "不能替代专业心理咨询" in result.metadata["applicability"]
    assert "clinical_guideline" in result.metadata["availableSkills"]
    assert "会话ID：public-session" in result.metadata["formattedInput"]


def test_research_agent_parse_result_adds_medix_style_evidence_fields() -> None:
    agent = ResearchAgent()
    task = AgentTask(
        id=9,
        session_id=1,
        run_id="test-run",
        assigned_agent="research",
        task_type="symptom_research",
        payload={"user_input": "测试"},
    )
    raw = (
        "【资料检索结果】\n关键词：压力支持\n找到相关资料：2 条\n\n"
        "【证据摘要】\n1. 文献A\n- 证据等级：A 级\n2. 资料B\n- 证据强度：强\n\n"
        "【综合评估】\n- 证据强度：强\n- 主要结论：需要支持。\n- 局限性：样本有限。\n- 建议：连接校园资源。"
    )

    result = agent.parse_result(task, raw, "formatted")

    assert result.content == raw
    assert result.metadata["evidence_level"] == "A"
    assert result.metadata["literature_count"] == raw.count("文献") + raw.count("资料")
    assert result.metadata["evidence_provided"] is True


@pytest.mark.asyncio
async def test_consultation_agent_calls_allowed_skills_before_final_answer() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = ConsultationAgent(HeuristicAiClient(), executor)
    context = context_with_capabilities(registry, knowledge_service=FakeKnowledgeService())
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "我最近焦虑睡不着"},
    )

    result = await agent.consult(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert [item["name"] for item in result.metadata["capabilityCalls"]] == [
        "recommend_lifestyle",
        "search_history",
    ]
    assert all(item["status"] == "SUCCESS" for item in result.metadata["capabilityResults"])
    assert "recommend_lifestyle" in result.metadata["capabilityObservationText"]
    assert [item["action"]["name"] for item in result.metadata["reactTrace"] if "action" in item] == [
        "recommend_lifestyle",
        "search_history",
    ]
    assert result.metadata["reactStoppedReason"] == "final"
    assert result.metadata["reactStepCount"] == 3


@pytest.mark.asyncio
async def test_diagnostic_agent_calls_risk_and_pattern_skills() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = DiagnosticAgent(HeuristicAiClient(), executor)
    context = context_with_capabilities(registry)
    task = AgentTask(
        id=8,
        session_id=1,
        run_id="test-run",
        assigned_agent="diagnostic",
        task_type="diagnostic_detail",
        payload={"user_input": "我不想活了，最近越来越严重"},
    )

    result = await agent.diagnose(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert [item["name"] for item in result.metadata["capabilityCalls"]] == [
        "assess_risk",
        "analyze_symptoms",
        "report.create_and_dispatch",
    ]
    assert result.metadata["capabilityResults"][0]["output"]["risk_level"] == "emergency"
    assert result.metadata["capabilityResults"][2]["status"] == "FAILED"
    assert "Missing report runtime dependencies" in result.metadata["capabilityResults"][2]["error"]
    assert "analyze_symptoms" in result.metadata["capabilityObservationText"]
    assert [item["action"]["name"] for item in result.metadata["reactTrace"] if "action" in item] == [
        "assess_risk",
        "analyze_symptoms",
        "report.create_and_dispatch",
    ]
    assert result.metadata["reactStoppedReason"] == "final"


@pytest.mark.asyncio
async def test_diagnostic_agent_report_tool_creates_report_and_dispatches(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = DiagnosticAgent(HeuristicAiClient(), executor)
    tool_service = RecordingToolService()
    async with session_factory() as db:
        chat_session = await seed_session(db)
        user = await db.get(UserAccount, chat_session.user_id)
        history = [AiMessage("user", "我不想活了，最近越来越严重")]
        context = AgentContext(
            session_id=chat_session.id,
            public_session_id=chat_session.public_id,
            user_id=user.id,
            display_name=user.display_name,
            history=history,
            skills=registry.skill_metadata(),
            tools=registry.tool_metadata(),
            mcp_tools=registry.mcp_metadata(),
            capability_context={
                "db": db,
                "user": user,
                "chat_session": chat_session,
                "tool_service": tool_service,
                "history": history,
                "session_id": chat_session.public_id,
            },
        )
        task = AgentTask(
            id=8,
            session_id=chat_session.id,
            run_id="test-run",
            assigned_agent="diagnostic",
            task_type="diagnostic_detail",
            payload={"user_input": "我不想活了，最近越来越严重"},
        )

        result = await agent.diagnose(context, task)
        reports = (await db.scalars(select(PsychologicalReport))).all()

    assert result.status == AgentTaskStatus.SUCCESS
    assert [item["name"] for item in result.metadata["capabilityCalls"]] == [
        "assess_risk",
        "analyze_symptoms",
        "report.create_and_dispatch",
    ]
    assert result.metadata["capabilityResults"][2]["status"] == "SUCCESS"
    assert result.metadata["capabilityResults"][2]["output"]["status"] == "created"
    assert len(reports) == 1
    assert reports[0].risk_level.value == "HIGH"
    assert reports[0].excel_status == ToolStatus.SUCCESS
    assert tool_service.report_ids == [reports[0].id]


@pytest.mark.asyncio
async def test_research_agent_calls_knowledge_skills_with_fake_knowledge_service() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = ResearchAgent(HeuristicAiClient(), executor)
    knowledge = FakeKnowledgeService()
    context = context_with_capabilities(registry, knowledge_service=knowledge)
    task = AgentTask(
        id=9,
        session_id=1,
        run_id="test-run",
        assigned_agent="research",
        task_type="symptom_research",
        payload={"user_input": "学校心理危机干预原则和证据有哪些？"},
    )

    result = await agent.research(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert [item["name"] for item in result.metadata["capabilityCalls"]] == [
        "clinical_guideline",
        "deep_research",
    ]
    assert all(item["status"] == "SUCCESS" for item in result.metadata["capabilityResults"])
    assert any("危机干预" in call[0] for call in knowledge.calls)
    assert [item["action"]["name"] for item in result.metadata["reactTrace"] if "action" in item] == [
        "clinical_guideline",
        "deep_research",
    ]
    assert result.metadata["reactStoppedReason"] == "final"


@pytest.mark.asyncio
async def test_invalid_capability_plan_does_not_fail_agent() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = ConsultationAgent(BadCapabilityPlanAiClient(), executor)
    context = context_with_capabilities(registry)
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "我最近焦虑睡不着"},
    )

    result = await agent.consult(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert result.metadata["capabilityCalls"] == []
    assert result.metadata["capabilityResults"] == []
    assert "ReAct step did not contain JSON" in result.metadata["capabilityPlanError"]
    assert "ReAct step did not contain JSON" in result.metadata["reactError"]
    assert result.metadata["reactStoppedReason"] == "parse_error_fallback"


@pytest.mark.asyncio
async def test_mcp_capability_call_is_blocked_and_agent_still_succeeds() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)
    agent = ConsultationAgent(McpCapabilityPlanAiClient(), executor)
    context = context_with_capabilities(registry)
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "请写报告"},
    )

    result = await agent.consult(context, task)

    assert result.status == AgentTaskStatus.SUCCESS
    assert result.metadata["capabilityResults"][0]["status"] == "BLOCKED"
    assert "multimodalAgent.excel.write_report" in result.metadata["capabilityObservationText"]
    assert result.metadata["capabilityCalls"][0]["name"] == "multimodalAgent.excel.write_report"
    assert result.metadata["reactTrace"][0]["status"] == "BLOCKED"
    assert result.metadata["reactStoppedReason"] == "final"


@pytest.mark.asyncio
async def test_report_tool_is_blocked_for_non_diagnostic_agents() -> None:
    registry = CapabilityRegistry("skills")
    executor = CapabilityExecutor(registry, timeout_seconds=1, max_calls=3)

    result = await executor.execute(
        {},
        CapabilityCall("report.create_and_dispatch", {"risk_level": "high"}),
        ConsultationAgent.CAPABILITY_ALLOWLIST,
    )

    assert result.status.value == "BLOCKED"
    assert result.kind.value == "tool"
    assert "not allowed" in (result.error or "")


@pytest.mark.asyncio
async def test_consultation_agent_llm_failure_returns_failed_result() -> None:
    agent = ConsultationAgent(FailingAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
    )
    task = AgentTask(
        id=7,
        session_id=1,
        run_id="test-run",
        assigned_agent="consultation",
        task_type="consultation_intake",
        payload={"user_input": "我最近焦虑睡不着"},
    )

    result = await agent.consult(context, task)

    assert result.status == AgentTaskStatus.FAILED
    assert result.error == "llm failed"
    assert "formattedInput" in result.metadata


@pytest.mark.asyncio
async def test_diagnostic_agent_llm_failure_returns_failed_result() -> None:
    agent = DiagnosticAgent(FailingAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
    )
    task = AgentTask(
        id=8,
        session_id=1,
        run_id="test-run",
        assigned_agent="diagnostic",
        task_type="diagnostic_detail",
        payload={"user_input": "我最近焦虑越来越严重"},
    )

    result = await agent.diagnose(context, task)

    assert result.status == AgentTaskStatus.FAILED
    assert result.error == "llm failed"
    assert "formattedInput" in result.metadata


@pytest.mark.asyncio
async def test_research_agent_llm_failure_returns_failed_result() -> None:
    agent = ResearchAgent(FailingAiClient())
    context = AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=[],
    )
    task = AgentTask(
        id=9,
        session_id=1,
        run_id="test-run",
        assigned_agent="research",
        task_type="symptom_research",
        payload={"user_input": "学校心理危机干预原则有哪些？"},
    )

    result = await agent.research(context, task)

    assert result.status == AgentTaskStatus.FAILED
    assert result.error == "llm failed"
    assert "formattedInput" in result.metadata


@pytest.mark.asyncio
async def test_lead_agent_diagnostic_worker_returns_structured_llm_result(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=HeuristicAiClient(),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "我不想活了，最近焦虑越来越严重")

    diagnostic = next(item for item in result.results if item.agent_name == "diagnostic")
    assert diagnostic.status == AgentTaskStatus.SUCCESS
    assert "【风险评估】" in diagnostic.content
    assert "风险等级：高" in diagnostic.content
    assert diagnostic.metadata["riskLevel"] == "高"
    assert diagnostic.metadata["risk_level"] == "high"
    assert diagnostic.metadata["diagnosis_provided"] is True
    assert diagnostic.content != "已完成诊断信息框架整理。"


@pytest.mark.asyncio
async def test_lead_agent_research_worker_returns_structured_llm_result(tmp_path: Path) -> None:
    session_factory = await make_session_factory(tmp_path)
    async with session_factory() as db:
        chat_session = await seed_session(db)
        lead = LeadAgent(
            ai_client=HeuristicAiClient(),
            queue=AgentTaskQueue(),
            timeout_seconds=1,
        )

        result = await lead.run(db, context_for(chat_session), "学校心理危机干预原则和证据有哪些？")

    research = next(item for item in result.results if item.agent_name == "research")
    assert research.status == AgentTaskStatus.SUCCESS
    assert "【资料检索结果】" in research.content
    assert "关键词：校园心理支持" in research.content
    assert research.metadata["sourceCount"] == 3
    assert research.metadata["evidenceStrength"] == "中"
    assert research.content != "已完成研究建议框架整理。"


@pytest.mark.asyncio
async def test_agent_tables_are_in_metadata_and_create_on_sqlite(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        table_names = await connection.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
        agent_columns = await connection.run_sync(
            lambda sync_conn: {column["name"] for column in inspect(sync_conn).get_columns("agent_tasks")}
        )
        agent_indexes = await connection.run_sync(
            lambda sync_conn: {index["name"] for index in inspect(sync_conn).get_indexes("agent_tasks")}
        )
    await engine.dispose()

    assert "agent_tasks" in table_names
    assert "run_id" in agent_columns
    assert "ix_agent_tasks_run_id_status" in agent_indexes
    assert "agent_memory_summaries" in table_names


async def make_session_factory(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


async def seed_session(db) -> ChatSession:
    user = UserAccount(username="student", password="x", display_name="student")
    db.add(user)
    await db.flush()
    chat_session = ChatSession(public_id="session-1", user_id=user.id, title="hello")
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


def context_for(chat_session: ChatSession) -> AgentContext:
    return AgentContext(
        session_id=chat_session.id,
        public_session_id=chat_session.public_id,
        user_id=chat_session.user_id,
        display_name="student",
        history=[AiMessage("user", "hello")],
    )


def context_with_capabilities(
    registry: CapabilityRegistry,
    knowledge_service=None,
) -> AgentContext:
    history = [
        AiMessage("user", "我最近焦虑睡不着"),
        AiMessage("assistant", "我们可以先看睡眠和压力的触发点。"),
    ]
    long_memory = ["用户曾提到考试压力和睡眠下降，需要校园心理支持。"]
    return AgentContext(
        session_id=1,
        public_session_id="public-session",
        user_id=2,
        display_name="student",
        history=history,
        long_memory=long_memory,
        skills=registry.skill_metadata(),
        tools=registry.tool_metadata(),
        mcp_tools=registry.mcp_metadata(),
        capability_context={
            "db": object(),
            "knowledge_service": knowledge_service,
            "history": history,
            "long_memory": long_memory,
            "session_id": "public-session",
        },
    )
