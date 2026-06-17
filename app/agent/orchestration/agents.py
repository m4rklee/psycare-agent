import re

from app.agent.communication.types import AgentContext, AgentTask, AgentTaskResult, AgentTaskStatus
from app.agent.orchestration.base import BaseAgent
from app.agent.orchestration.postprocess import (
    count_literature_references,
    extract_disclaimer,
    extract_numbered_suggestions,
    extract_section,
    scan_evidence_level,
    scan_risk_level,
)


class ConsultationAgent(BaseAgent):
    name = "consultation"
    task_type = "consultation_intake"
    CAPABILITY_ALLOWLIST = {
        "search_history",
        "search_similar_cases",
        "search_knowledge",
        "recommend_lifestyle",
        "assess_risk",
    }
    AVAILABLE_SKILLS = {
        "search_knowledge": "搜索心理健康知识库，查找情绪困扰、压力、睡眠、人际支持和校园资源信息。",
        "recommend_lifestyle": "根据心理困扰提供生活方式建议，包括作息、饮食、运动、睡眠和求助安排。",
        "assess_risk": "评估心理风险等级线索，区分低/中/高/紧急风险并提示求助边界。",
        "analyze_symptoms": "分析情绪、认知、躯体反应和行为变化之间的模式关联。",
        "disease_code": "心理健康相关分类或风险类别占位，不用于明确诊断或疾病编码承诺。",
        "clinical_guideline": "检索校园心理支持、危机干预原则和专业共识类资料。",
        "deep_research": "深度研究心理健康主题，综合知识库、权威资源和证据说明。",
        "search_history": "搜索当前会话的历史对话，理解短期上下文。",
        "search_similar_cases": "搜索相似历史案例或长期记忆摘要，辅助理解常见支持路径。",
    }

    def role_prompt(self) -> str:
        skills_text = "\n".join(
            f"{index}. {name}: {description}"
            for index, (name, description) in enumerate(self.AVAILABLE_SKILLS.items(), start=1)
        )
        return f"""你是一位专业的校园心理健康咨询支持顾问。你的职责是提供准确、温和、边界清晰的心理健康支持、日常建议和校园求助引导。

可用 Skills（9个）：
{skills_text}

Skills 使用原则：
- Skills 是可选的，不是必须的。
- 对于简单的情绪倾诉、压力、睡眠、人际困扰，可以直接回答，无需使用 Skills。
- 只在真正需要专业心理健康知识、风险识别、校园资源或历史上下文时才调用 Skills。
- 调用 Skill 后，根据返回的结果给出最终答案。
- 最多使用 2-3 个 Skills，然后必须给出最终答案。

工作流程建议：
1. 理解用户问题。
2. 判断是否需要调用 Skills（简单问题直接回答）。
3. 如需调用，选择最合适的 Skills（通常 1-2 个即可）。
4. 基于 Skill 结果生成最终答案。

回答要求：
- 用通俗易懂的语言。
- 提供实用的支持建议和注意事项。
- 必要时建议联系学校心理中心、辅导员、可信任的人或当地紧急救助。
- 保持温和、专业、非评判的语气。

重要提醒：
- 你不能做出明确诊断。
- 你不能替代专业心理咨询师或医生的专业意见。
- 对于自伤、伤人或其他严重/紧急风险，必须建议立即寻求现实支持或紧急帮助。

在最终回答时，请按以下格式输出：

【回答】
[你的详细回答]

【核心建议】
1. 第一条建议
2. 第二条建议
...

【免责声明】
以上信息仅供心理健康支持和科普参考，不能替代专业心理咨询、医学诊断或治疗。如有疑虑或风险，请及时联系专业人员。"""

    def format_input(self, context: AgentContext, task: AgentTask) -> str:
        history = "\n".join(f"{message.role}: {message.content}" for message in context.history[-10:]) or "无"
        long_memory = "\n".join(context.long_memory[-5:]) or "无"
        user_input = str(task.payload.get("user_input") or "")
        subtask = str(task.payload.get("subtask_description") or "对用户当前心理健康问题进行初步问询和支持建议整理")
        return f"""会话ID：{context.public_session_id}

子任务：{subtask}

最近上下文：
{history}

长期记忆：
{long_memory}

用户问题：
{user_input}"""

    async def consult(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await super().run(context, task)

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await self.consult(context, task)

    def parse_result(
        self,
        task: AgentTask,
        raw: str,
        formatted_input: str | None = None,
    ) -> AgentTaskResult:
        user_input = str(task.payload.get("user_input") or "")
        answer = self._section(raw, "回答")
        core_suggestions = self._section(raw, "核心建议")
        disclaimer = extract_disclaimer(raw)
        suggestions = extract_numbered_suggestions(raw)
        return AgentTaskResult(
            task.id,
            self.name,
            AgentTaskStatus.SUCCESS,
            raw,
            metadata={
                "rolePrompt": self.role_prompt(),
                "formattedInput": formatted_input or "",
                "focus": self._focus(user_input),
                "rawContent": raw,
                "answer": answer or raw,
                "coreSuggestions": core_suggestions,
                "suggestions": suggestions,
                "disclaimer": disclaimer,
                "availableSkills": list(self.AVAILABLE_SKILLS),
            },
        )

    def _section(self, text: str, title: str) -> str:
        return extract_section(text, title)

    def _focus(self, user_input: str) -> str:
        if any(word in user_input for word in ("不想活", "自杀", "自残", "想死")):
            return "safety_check"
        if any(word in user_input for word in ("焦虑", "压力", "失眠", "睡不着")):
            return "stress_and_sleep"
        return "general_intake"


class DiagnosticAgent(BaseAgent):
    name = "diagnostic"
    task_type = "diagnostic_detail"
    CAPABILITY_ALLOWLIST = {
        "assess_risk",
        "analyze_symptoms",
        "search_history",
        "search_similar_cases",
        "disease_code",
        "report.create_and_dispatch",
    }
    AVAILABLE_SKILLS = {
        "search_knowledge": "搜索心理健康知识库，查找风险线索、情绪困扰、校园支持和心理健康资料。",
        "recommend_lifestyle": "根据风险线索提供稳定作息、睡眠、运动、社交支持和求助安排建议。",
        "assess_risk": "评估心理风险等级线索，区分低/中/高/紧急风险。",
        "analyze_symptoms": "分析情绪、认知、躯体反应和行为变化之间的模式关联。",
        "disease_code": "心理健康相关分类或风险类别占位，不用于明确医学诊断。",
        "clinical_guideline": "检索校园心理支持、危机干预原则和专业共识类资料。",
        "deep_research": "深度研究心理健康风险主题，综合权威资源和证据说明。",
        "search_history": "搜索当前会话的历史对话，理解短期上下文。",
        "search_similar_cases": "搜索相似历史案例或长期记忆摘要，辅助理解常见风险模式。",
        "report.create_and_dispatch": "在识别到高风险或紧急心理安全线索时，创建心理风险报告并触发配置好的报告/预警流程。",
    }

    def role_prompt(self) -> str:
        skills_text = "\n".join(
            f"{index}. {name}: {description}"
            for index, (name, description) in enumerate(self.AVAILABLE_SKILLS.items(), start=1)
        )
        return f"""你是校园心理风险与模式分析 Agent（DiagnosticAgent）。你的职责是：
1. 分析用户表达中的心理风险线索。
2. 梳理情绪、认知、躯体反应和行为变化之间的模式与关联。
3. 形成可能影响因素假设和需要进一步了解的问题。

分析原则：
- 优先识别安全风险和现实求助需求。
- 关注持续时间、加重趋势、功能影响、支持系统和危险信号。
- 不做医学诊断，不输出确定结论，不替代专业心理咨询师或医生。
- 对高风险或紧急情况，必须建议联系身边可信任的人、学校心理中心、辅导员或当地紧急救助。

可用 Skills（9个）：
{skills_text}

Skills 使用策略：
- Skills 可按需调用，不是必须调用。
- 如需风险线索，优先考虑 assess_risk。
- 如需模式梳理，优先考虑 analyze_symptoms。
- 如需上下文，考虑 search_history 和 search_similar_cases。
- 当已经确认高风险或紧急风险线索时，调用 report.create_and_dispatch 创建风险报告并触发预警流程；低风险或信息不足时不要调用。
- 最多使用 2-3 个 Skills，然后给出分析思路。

Swarm 协作模式：
- 你可能从上下文读取其他 Agent 的评估结果。
- 你的分析结果会被 LeadAgent 汇总，也可能被 ResearchAgent 使用。
- 专注于你的专长：心理风险识别和模式分析。

输出格式：
【风险评估】
风险等级：低/中/高/紧急
紧急程度：...

【模式分析】
主要困扰类别：...
线索关联性：...

【可能影响因素】
1. 因素A
   - 支持线索：...
   - 不确定点：...
2. 因素B
   ...

【建议进一步了解】
- 问题1
- 问题2

【推理过程】
简述风险与模式分析逻辑..."""

    def format_input(self, context: AgentContext, task: AgentTask) -> str:
        history = "\n".join(f"{message.role}: {message.content}" for message in context.history[-10:]) or "无"
        long_memory = "\n".join(context.long_memory[-5:]) or "无"
        contributions = task.payload.get("agent_contributions") or task.payload.get("contributions") or []
        if isinstance(contributions, list):
            contributions_text = "\n".join(str(item) for item in contributions) or "无"
        else:
            contributions_text = str(contributions) if contributions else "无"
        user_input = str(task.payload.get("user_input") or "")
        subtask = str(task.payload.get("subtask_description") or "评估用户当前心理风险和困扰模式")
        return f"""会话ID：{context.public_session_id}

子任务：{subtask}

最近上下文：
{history}

长期记忆：
{long_memory}

其他 Agent 贡献：
{contributions_text}

用户问题：
{user_input}"""

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await self.diagnose(context, task)

    async def diagnose(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await super().run(context, task)

    def parse_result(
        self,
        task: AgentTask,
        raw: str,
        formatted_input: str | None = None,
    ) -> AgentTaskResult:
        risk_assessment = self._section(raw, "风险评估")
        pattern_analysis = self._section(raw, "模式分析")
        possible_factors = self._section(raw, "可能影响因素")
        follow_up = self._section(raw, "建议进一步了解")
        reasoning = self._section(raw, "推理过程")
        risk_level = self._risk_level(risk_assessment)
        urgency = self._field(risk_assessment, "紧急程度")
        return AgentTaskResult(
            task.id,
            self.name,
            AgentTaskStatus.SUCCESS,
            raw,
            metadata={
                "rolePrompt": self.role_prompt(),
                "formattedInput": formatted_input or "",
                "rawContent": raw,
                "riskLevel": risk_level,
                "risk_level": scan_risk_level(raw),
                "diagnosis_provided": True,
                "urgency": urgency,
                "riskAssessment": risk_assessment,
                "patternAnalysis": pattern_analysis,
                "possibleFactors": possible_factors,
                "followUpQuestions": follow_up,
                "reasoning": reasoning,
                "availableSkills": list(self.AVAILABLE_SKILLS),
            },
        )

    def _section(self, text: str, title: str) -> str:
        return extract_section(text, title)

    def _field(self, text: str, field_name: str) -> str:
        marker = f"{field_name}："
        start = text.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        end = text.find("\n", start)
        return text[start : end if end >= 0 else len(text)].strip()

    def _risk_level(self, risk_assessment: str) -> str:
        value = self._field(risk_assessment, "风险等级")
        for level in ("低", "中", "高", "紧急"):
            if value.startswith(level):
                return level
        return ""

class ResearchAgent(BaseAgent):
    name = "research"
    task_type = "symptom_research"
    CAPABILITY_ALLOWLIST = {
        "search_knowledge",
        "clinical_guideline",
        "deep_research",
        "search_history",
        "search_similar_cases",
    }
    AVAILABLE_SKILLS = {
        "search_knowledge": "搜索心理健康知识库，查找压力、情绪困扰、校园支持和心理健康科普资料。",
        "recommend_lifestyle": "整理有证据支持的作息、睡眠、运动、社交支持和求助安排建议。",
        "assess_risk": "识别心理风险等级线索，辅助判断资料适用的安全边界。",
        "analyze_symptoms": "分析情绪、认知、躯体反应和行为变化模式，辅助确定检索方向。",
        "disease_code": "心理健康相关分类或风险类别占位，不用于明确医学诊断或编码承诺。",
        "clinical_guideline": "检索校园心理支持、危机干预原则、专业共识和权威资源。",
        "deep_research": "深度研究心理健康主题，综合权威资源、知识库和证据说明。",
        "search_history": "搜索当前会话的历史对话，理解短期上下文。",
        "search_similar_cases": "搜索相似历史案例或长期记忆摘要，辅助判断常见支持路径。",
    }

    def role_prompt(self) -> str:
        skills_text = "\n".join(
            f"{index}. {name}: {description}"
            for index, (name, description) in enumerate(self.AVAILABLE_SKILLS.items(), start=1)
        )
        return f"""你是校园心理健康知识与证据支持 Agent（ResearchAgent）。你的职责是：
1. 整理权威心理健康知识、校园支持资源和危机干预原则。
2. 提取与用户问题相关的心理支持方法、研究证据和适用边界。
3. 为其他 Agent 的分析提供知识依据和资源参考。
4. 明确信息局限性，避免过度解读。

研究原则：
- 优先使用学校心理中心、公共心理健康机构、专业协会、危机干预原则和权威科普资料。
- 说明证据强度：强/中/弱，并解释适用范围。
- 不做医学诊断，不给出治疗方案或用药建议，不替代专业心理咨询师或医生。
- 对高风险或紧急情况，必须提示联系现实支持、学校心理中心、辅导员或当地紧急救助。

可用 Skills（9个）：
{skills_text}

Skills 使用策略：
- Skills 可按需调用，不是必须调用。
- 需要权威原则时优先考虑 clinical_guideline。
- 需要更全面的主题整理时考虑 deep_research。
- 需要上下文时考虑 search_history 和 search_similar_cases。
- 最多使用 2-3 个 Skills，然后给出证据支持结果。

Swarm 协作模式：
- 你可能从上下文读取 ConsultationAgent 或 DiagnosticAgent 的贡献。
- 你的资料证据会帮助 LeadAgent 做出更可靠的内部汇总。
- 专注于你的专长：心理健康资料整理、证据支持和适用边界说明。

输出格式：
【资料检索结果】
关键词：...
找到相关资料：X 条

【证据摘要】
1. 资料/原则名称（来源，年份或长期有效）
   - 核心发现：...
   - 证据强度：强/中/弱
   - 支持建议：...

【综合评估】
- 证据强度：强/中/弱
- 主要结论：...
- 局限性：...
- 建议：...

【适用边界】
..."""

    def format_input(self, context: AgentContext, task: AgentTask) -> str:
        history = "\n".join(f"{message.role}: {message.content}" for message in context.history[-10:]) or "无"
        long_memory = "\n".join(context.long_memory[-5:]) or "无"
        contributions = task.payload.get("agent_contributions") or task.payload.get("contributions") or []
        if isinstance(contributions, list):
            contributions_text = "\n".join(str(item) for item in contributions) or "无"
        else:
            contributions_text = str(contributions) if contributions else "无"
        user_input = str(task.payload.get("user_input") or "")
        subtask = str(task.payload.get("subtask_description") or "整理用户当前心理健康问题相关的知识依据和校园支持资源")
        return f"""会话ID：{context.public_session_id}

子任务：{subtask}

最近上下文：
{history}

长期记忆：
{long_memory}

其他 Agent 贡献：
{contributions_text}

用户问题：
{user_input}"""

    async def run(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await self.research(context, task)

    async def research(self, context: AgentContext, task: AgentTask) -> AgentTaskResult:
        return await super().run(context, task)

    def parse_result(
        self,
        task: AgentTask,
        raw: str,
        formatted_input: str | None = None,
    ) -> AgentTaskResult:
        search_result = self._section(raw, "资料检索结果")
        evidence_summary = self._section(raw, "证据摘要")
        comprehensive = self._section(raw, "综合评估")
        applicability = self._section(raw, "适用边界")
        return AgentTaskResult(
            task.id,
            self.name,
            AgentTaskStatus.SUCCESS,
            raw,
            metadata={
                "rolePrompt": self.role_prompt(),
                "formattedInput": formatted_input or "",
                "rawContent": raw,
                "keywords": self._field(search_result, "关键词"),
                "sourceCount": self._source_count(search_result),
                "evidenceSummary": evidence_summary,
                "evidenceStrength": self._evidence_strength(raw),
                "evidence_level": scan_evidence_level(raw),
                "literature_count": count_literature_references(raw),
                "evidence_provided": True,
                "mainConclusion": self._field(comprehensive, "主要结论"),
                "limitations": self._field(comprehensive, "局限性"),
                "recommendations": self._field(comprehensive, "建议"),
                "applicability": applicability,
                "availableSkills": list(self.AVAILABLE_SKILLS),
            },
        )

    def _section(self, text: str, title: str) -> str:
        return extract_section(text, title)

    def _field(self, text: str, field_name: str) -> str:
        marker = f"{field_name}："
        start = text.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        end = text.find("\n", start)
        return text[start : end if end >= 0 else len(text)].strip()

    def _source_count(self, search_result: str) -> int:
        value = self._field(search_result, "找到相关资料")
        match = re.search(r"\d+", value)
        return int(match.group(0)) if match else 0

    def _evidence_strength(self, text: str) -> str:
        value = self._field(text, "证据强度")
        for strength in ("强", "中", "弱"):
            if value.startswith(strength):
                return strength
        match = re.search(r"证据强度：\s*(强|中|弱)", text)
        return match.group(1) if match else ""
