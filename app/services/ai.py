import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings

HIGH_RISK_WORDS = (
    "不想活",
    "活不下去",
    "撑不下去",
    "自杀",
    "自残",
    "轻生",
    "结束生命",
    "结束这一切",
    "伤害自己",
    "伤人",
    "杀了",
    "想死",
    "去死",
    "没有活着的意义",
    "不想存在",
    "消失算了",
    "suicide",
    "kill myself",
    "self harm",
    "end my life",
    "hurt myself",
    "hurt others",
    "want to die",
)

CONSULT_WORDS = (
    "焦虑",
    "压力",
    "压抑",
    "抑郁",
    "低落",
    "失眠",
    "睡不着",
    "崩溃",
    "难过",
    "孤独",
    "情绪",
    "心理",
    "心理咨询",
    "咨询师",
    "心累",
    "烦躁",
    "害怕",
    "恐惧",
    "内耗",
    "想哭",
    "不开心",
    "没动力",
    "痛苦",
    "沮丧",
    "绝望",
    "无助",
    "喘不过气",
    "panic attack",
    "anxiety",
    "anxious",
    "stress",
    "depress",
    "sad",
    "insomnia",
    "panic",
    "lonely",
    "breakup",
)


@dataclass(frozen=True)
class AiMessage:
    role: str
    content: str


class AiClient:
    async def complete(self, messages: list[AiMessage]) -> str:
        raise NotImplementedError

    async def stream(self, messages: list[AiMessage]) -> AsyncIterator[str]:
        raise NotImplementedError


class HeuristicAiClient(AiClient):
    async def complete(self, messages: list[AiMessage]) -> str:
        prompt = "\n".join(message.content for message in messages)
        prompt_lower = prompt.lower()
        input_text = self._last_user_message(messages)
        if "Agent ReAct 控制器" in prompt:
            return self._react_step(input_text, prompt)
        if "Agent 能力调用规划器" in prompt:
            return self._capability_plan(input_text, prompt)
        if "Agentic RAG planner" in prompt:
            return '{"reason":"围绕学生当前困扰、校园心理支持和安全边界进行检索","queries":["校园心理咨询 焦虑 压力 支持建议","学生 情绪困扰 应对方法","高校心理危机 安全处理 流程"]}'
        if "Agentic RAG evidence reviewer" in prompt:
            return '{"sufficient":true,"reason":"候选证据可以支持安全、通用的心理支持回答","followUpQueries":[]}'
        if "校园心理健康 Swarm 的 Lead Agent" in prompt and "subtasks" in prompt:
            return self._lead_decomposition(input_text)
        if "负责汇总多个专业 Agent 的分析结果" in prompt:
            return self._lead_summary(prompt)
        if "校园心理健康咨询支持顾问" in prompt and "【回答】" in prompt:
            return self._consultation_agent_answer(input_text)
        if "校园心理风险与模式分析 Agent" in prompt and "【风险评估】" in prompt:
            return self._diagnostic_agent_answer(input_text)
        if "校园心理健康知识与证据支持 Agent" in prompt and "【资料检索结果】" in prompt:
            return self._research_agent_answer(input_text)
        if "intent classifier" in prompt_lower or "意图分类器" in prompt:
            return self._classify(input_text)
        if "strict json" in prompt_lower or "严格 JSON" in prompt or '"emotion"' in prompt:
            return self._analyze(input_text)
        return self._answer(input_text, prompt)

    async def stream(self, messages: list[AiMessage]) -> AsyncIterator[str]:
        text = await self.complete(messages)
        for token in text:
            yield token

    def _classify(self, input_text: str) -> str:
        normalized = input_text.lower()
        current = self._current_input(input_text).lower()
        if self._has_high_risk_signal(current):
            return "RISK"
        if self._has_consult_signal(current) or self._has_consult_signal(normalized):
            return "CONSULT"
        return "CHAT"

    def _analyze(self, input_text: str) -> str:
        normalized = input_text.lower()
        current = self._current_input(input_text).lower()
        if self._has_high_risk_signal(current):
            return '{"emotion":"HIGH_RISK","emotionScore":4.0,"risk":"HIGH","confidence":0.92,"summary":"检测到明确的高风险自伤或危险信号"}'
        if self._contains_any(current, "抑郁", "低落", "压抑", "崩溃", "难过", "绝望", "depress", "hopeless"):
            return '{"emotion":"DEPRESSED","emotionScore":3.2,"risk":"MEDIUM","confidence":0.82,"summary":"检测到持续低落或压抑相关表达"}'
        if self._contains_any(current, "焦虑", "压力", "睡不着", "失眠", "紧张", "anxious", "stress", "insomnia"):
            return '{"emotion":"ANXIETY","emotionScore":2.2,"risk":"LOW","confidence":0.78,"summary":"检测到焦虑、压力或睡眠困扰相关表达"}'
        if self._has_consult_signal(normalized):
            return '{"emotion":"ANXIETY","emotionScore":2.0,"risk":"LOW","confidence":0.70,"summary":"结合上下文检测到心理咨询延续表达"}'
        return '{"emotion":"NORMAL","emotionScore":0.0,"risk":"LOW","confidence":0.70,"summary":"未检测到明显心理风险信号"}'

    def _answer(self, input_text: str, prompt: str) -> str:
        normalized = input_text.lower()
        if self._has_high_risk_signal(normalized):
            return (
                "我会认真对待你刚才说的这些话。现在最重要的不是把问题讲清楚，而是先确保你此刻是安全的。\n\n"
                "请你先做三件事：第一，离开任何可能让你伤害自己或他人的物品和环境；第二，马上联系一个现实中能到你身边的人，比如同学、室友、家人、辅导员或学校心理中心；第三，如果你已经处在马上会伤害自己或他人的危险里，请立刻拨打当地紧急救助电话。\n\n"
                "你不用一个人扛完这一刻。你可以先回复我一个很短的答案：你现在是一个人吗？身边有没有一个可以立刻联系到的人？"
            )
        if self._has_consult_signal(normalized) or "检索知识：" in prompt:
            return self._consult_answer(input_text)
        return (
            "我在。你可以把我当作一个校园心理支持助手来用：日常闲聊我会自然回应；如果你聊到压力、焦虑、睡眠、人际关系或学习困扰，"
            "我会先判断意图，再结合最近上下文和知识库给你更具体的建议。\n\n"
            "你现在想聊轻松一点的内容，还是想说说最近真正让你卡住的一件事？"
        )

    def _lead_decomposition(self, input_text: str) -> str:
        normalized = input_text.lower()
        current = self._current_question(input_text).lower()
        if self._contains_any(current, "指南", "证据", "研究", "资源", "干预原则", "循证"):
            if self._has_consult_signal(current) and self._contains_any(current, "同时", "也", "还想", "建议"):
                return (
                    '{"subtasks":['
                    '{"description":"整理与用户问题相关的心理健康知识、校园支持资源和证据依据",'
                    '"assigned_agent":"research_agent"},'
                    '{"description":"基于用户当前困扰提供初步心理支持和后续沟通建议",'
                    '"assigned_agent":"consultation_agent"}]}'
                )
            return (
                '{"subtasks":['
                '{"description":"整理与用户问题相关的心理健康知识、校园支持资源和证据依据",'
                '"assigned_agent":"research_agent"}]}'
            )
        if self._should_use_diagnostic_swarm(current):
            return (
                '{"subtasks":['
                '{"description":"评估用户当前心理风险、紧急程度和症状模式",'
                '"assigned_agent":"diagnostic_agent"},'
                '{"description":"提供初步心理支持、稳定化建议和求助引导",'
                '"assigned_agent":"consultation_agent"}]}'
            )
        if self._has_consult_signal(current) or self._has_consult_signal(normalized):
            return (
                '{"subtasks":[{"description":"对用户当前心理健康问题进行初步问询和支持建议整理",'
                '"assigned_agent":"consultation_agent"}]}'
            )
        return (
            '{"subtasks":[{"description":"对用户当前问题进行日常陪伴式回应和必要的心理健康边界判断",'
            '"assigned_agent":"consultation_agent"}]}'
        )

    def _lead_summary(self, prompt: str) -> str:
        return (
            "【回答】\n"
            "我已经综合了不同 Agent 的分析。你提到的状态值得被认真看见，先不要急着责备自己，我们可以同时关注安全、情绪变化和下一步支持。\n\n"
            "【风险评估】\n"
            "如果没有明确自伤、伤人或失控风险，可以先按一般心理困扰处理；如果出现这些信号，请优先联系现实中的可信任人员、学校心理中心或紧急救助。\n\n"
            "【核心建议】\n"
            "1. 继续保持温和、聚焦的支持性回应。\n"
            "2. 如出现明确风险信号，应优先引导现实支持和紧急求助。\n\n"
            "【免责声明】\n"
            "本内容不替代专业心理咨询或医疗诊断。"
        )

    def _capability_plan(self, input_text: str, prompt: str) -> str:
        current = self._current_question(input_text).lower()
        calls = []
        if self._has_high_risk_signal(current):
            if "assess_risk" in prompt:
                calls.append({"name": "assess_risk", "input": {"user_input": current}})
            if "analyze_symptoms" in prompt:
                calls.append({"name": "analyze_symptoms", "input": {"user_input": current}})
        elif self._contains_any(current, "指南", "证据", "研究", "资源", "干预原则", "循证"):
            if "clinical_guideline" in prompt:
                calls.append({"name": "clinical_guideline", "input": {"query": current, "max_results": 2}})
            if "deep_research" in prompt:
                calls.append({"name": "deep_research", "input": {"query": current, "max_results": 2}})
            elif "search_knowledge" in prompt:
                calls.append({"name": "search_knowledge", "input": {"query": current, "max_results": 2}})
        elif self._has_consult_signal(current):
            if "recommend_lifestyle" in prompt:
                calls.append({"name": "recommend_lifestyle", "input": {"user_input": current}})
            if "search_history" in prompt:
                calls.append({"name": "search_history", "input": {"query": current, "limit": 3}})
        return json.dumps({"calls": calls[:3]}, ensure_ascii=False)

    def _react_step(self, input_text: str, prompt: str) -> str:
        current = self._current_question(input_text).lower()
        observations = self._react_observations(prompt)
        force_final = "现在必须输出 final" in prompt
        if force_final:
            return self._react_final_for_agent(input_text, prompt, "已根据现有观察生成最终结构化回答。")

        if self._has_high_risk_signal(current):
            if "assess_risk" in prompt and "assess_risk" not in observations:
                return self._react_action(
                    "需要先识别安全风险和紧急程度。",
                    "assess_risk",
                    {"user_input": current},
                )
            if "analyze_symptoms" in prompt and "analyze_symptoms" not in observations:
                return self._react_action(
                    "已有风险观察后，需要进一步梳理困扰模式。",
                    "analyze_symptoms",
                    {"user_input": current},
                )
            if "report.create_and_dispatch" in prompt and "report.create_and_dispatch" not in observations:
                return self._react_action(
                    "已确认高风险线索，需要创建风险报告并触发预警流程。",
                    "report.create_and_dispatch",
                    {
                        "user_input": current,
                        "content": current,
                        "risk_level": "high",
                        "summary": "用户表达出明确自伤或生命安全风险线索。",
                        "confidence": 0.92,
                    },
                )
            return self._react_final_for_agent(input_text, prompt, "风险和模式信息已足够，可以整合回答。")

        if self._contains_any(current, "指南", "证据", "研究", "资源", "干预原则", "循证"):
            if "clinical_guideline" in prompt and "clinical_guideline" not in observations:
                return self._react_action(
                    "需要先获取校园心理支持原则或资源依据。",
                    "clinical_guideline",
                    {"query": current, "max_results": 2},
                )
            if "deep_research" in prompt and "deep_research" not in observations:
                return self._react_action(
                    "需要补充综合资料和适用边界。",
                    "deep_research",
                    {"query": current, "max_results": 2},
                )
            if "deep_research" in observations:
                return self._react_final_for_agent(input_text, prompt, "综合研究观察已足够，可以形成最终评估。")
            if "search_knowledge" in prompt and "search_knowledge" not in observations:
                return self._react_action(
                    "需要检索心理健康知识库补充依据。",
                    "search_knowledge",
                    {"query": current, "max_results": 2},
                )
            return self._react_final_for_agent(input_text, prompt, "证据观察已足够，可以形成综合评估。")

        if self._has_consult_signal(current):
            if "recommend_lifestyle" in prompt and "recommend_lifestyle" not in observations:
                return self._react_action(
                    "需要先生成可执行的生活支持建议。",
                    "recommend_lifestyle",
                    {"user_input": current},
                )
            if "search_history" in prompt and "search_history" not in observations:
                return self._react_action(
                    "需要查看当前会话上下文，避免忽略已有信息。",
                    "search_history",
                    {"query": current, "limit": 3},
                )
            return self._react_final_for_agent(input_text, prompt, "支持建议和上下文观察已足够。")

        return self._react_final_for_agent(input_text, prompt, "无需调用额外能力，可以直接回答。")

    def _react_action(self, thought: str, name: str, input_data: dict) -> str:
        return json.dumps(
            {
                "thought": thought,
                "action": {
                    "name": name,
                    "input": input_data,
                },
            },
            ensure_ascii=False,
        )

    def _react_final_for_agent(self, input_text: str, prompt: str, thought: str) -> str:
        if "Agent：diagnostic" in prompt or "校园心理风险与模式分析 Agent" in prompt:
            final = self._diagnostic_agent_answer(input_text)
        elif "Agent：research" in prompt or "校园心理健康知识与证据支持 Agent" in prompt:
            final = self._research_agent_answer(input_text)
        else:
            final = self._consultation_agent_answer(input_text)
        return json.dumps({"thought": thought, "final": final}, ensure_ascii=False)

    def _react_observations(self, prompt: str) -> str:
        marker = "已有 Observation："
        if marker not in prompt:
            return ""
        return prompt.split(marker, 1)[1]

    def _consultation_agent_answer(self, input_text: str) -> str:
        focus = self._focus_from(input_text)
        return (
            "【回答】\n"
            f"我理解你现在想处理的是：{focus}。先不用急着给自己下结论，"
            "可以先把当下最明显的触发场景、持续时间和对睡眠/学习/关系的影响说清楚。"
            "如果已经出现伤害自己或他人的冲动，请优先联系身边可信任的人、辅导员、学校心理中心或当地紧急救助。\n\n"
            "【核心建议】\n"
            "1. 先记录最困扰你的具体场景，以及它出现的频率和持续时间。\n"
            "2. 做一个短暂稳定动作：放慢呼吸、离开刺激源，或联系一个现实中能回应你的人。\n"
            "3. 如果困扰持续两周以上，或明显影响睡眠、上课、饮食，建议尽快联系学校心理中心。\n\n"
            "【免责声明】\n"
            "以上信息仅供心理健康支持和科普参考，不能替代专业心理咨询、医学诊断或治疗。如有疑虑或风险，请及时联系专业人员。"
        )

    def _diagnostic_agent_answer(self, input_text: str) -> str:
        current = self._current_question(input_text).lower()
        if self._has_high_risk_signal(current):
            risk_level = "高"
            urgency = "需要尽快联系现实支持，若有即时危险请立即寻求紧急帮助"
        elif self._contains_any(current, "持续", "两周", "越来越", "严重"):
            risk_level = "中"
            urgency = "建议尽快联系学校心理中心或辅导员进一步评估"
        else:
            risk_level = "低"
            urgency = "可先观察和自我支持，同时留意变化"
        return (
            "【风险评估】\n"
            f"风险等级：{risk_level}\n"
            f"紧急程度：{urgency}\n\n"
            "【模式分析】\n"
            "主要困扰类别：压力、焦虑或睡眠相关困扰线索。\n"
            "线索关联性：当前表达显示情绪紧绷可能与睡眠、学习压力或人际支持不足相互影响。\n\n"
            "【可能影响因素】\n"
            "1. 压力累积\n"
            "   - 支持线索：用户表达出持续困扰或焦虑相关内容。\n"
            "   - 不确定点：压力来源、持续时间和功能影响仍需确认。\n"
            "2. 支持系统不足\n"
            "   - 支持线索：当前问题需要现实支持和进一步沟通。\n"
            "   - 不确定点：身边是否有可联系的人尚不明确。\n\n"
            "【建议进一步了解】\n"
            "- 这种状态持续了多久，是否越来越严重？\n"
            "- 是否影响睡眠、上课、饮食或人际关系？\n"
            "- 身边是否有可以马上联系的同学、家人、辅导员或老师？\n\n"
            "【推理过程】\n"
            "先根据危险信号和持续加重线索判断风险，再结合情绪、睡眠、学习和支持系统线索整理可能模式；当前只提供风险与模式分析，不做诊断结论。"
        )

    def _research_agent_answer(self, input_text: str) -> str:
        current = self._current_question(input_text).lower()
        if self._has_high_risk_signal(current):
            keywords = "校园心理危机干预，现实支持，紧急求助，安全计划"
            conclusion = "存在安全风险线索时，优先级最高的是现实陪伴、降低即时危险和连接专业/紧急资源。"
            recommendation = "先确认用户是否安全，并引导联系身边可信任的人、辅导员、学校心理中心或当地紧急救助。"
        elif self._contains_any(current, "指南", "证据", "研究", "资源", "干预原则", "循证"):
            keywords = "校园心理支持，心理健康资源，危机干预原则，循证支持"
            conclusion = "权威资料普遍支持早期识别、稳定化支持、现实资源连接和持续随访。"
            recommendation = "结合学校心理中心流程、辅导员支持、校园支持资源和可信任同伴/家人资源制定下一步求助安排。"
        else:
            keywords = "学生压力，情绪困扰，睡眠支持，校园心理咨询"
            conclusion = "对于常见压力和情绪困扰，稳定作息、社会支持、问题拆解和及时求助是较一致的支持方向。"
            recommendation = "先提供通俗、可执行的自我支持建议，并在持续加重或影响功能时建议联系学校心理中心。"
        return (
            "【资料检索结果】\n"
            f"关键词：{keywords}\n"
            "找到相关资料：3 条\n\n"
            "【证据摘要】\n"
            "1. 高校心理健康教育与咨询服务原则（校园心理支持资源，长期有效）\n"
            "   - 核心发现：学校心理中心、辅导员和现实支持网络通常是学生心理困扰的重要求助入口。\n"
            "   - 证据强度：中\n"
            "   - 支持建议：鼓励用户在困扰持续、加重或影响学习生活时尽快连接校园支持。\n"
            "2. 心理危机干预基本原则（公共心理健康资源，长期有效）\n"
            "   - 核心发现：出现自伤、伤人或失控风险时，应优先确保安全、减少危险物接触并寻求即时帮助。\n"
            "   - 证据强度：强\n"
            "   - 支持建议：高风险内容需要明确提示现实陪伴、学校心理中心和紧急救助。\n"
            "3. 学生压力与睡眠支持建议（心理健康科普资料，长期有效）\n"
            "   - 核心发现：压力、睡眠和社交支持会相互影响，短期稳定化技巧可作为初步支持。\n"
            "   - 证据强度：中\n"
            "   - 支持建议：建议使用呼吸放松、规律作息、任务拆解和现实沟通等低风险方法。\n\n"
            "【综合评估】\n"
            "- 证据强度：中\n"
            f"- 主要结论：{conclusion}\n"
            "- 局限性：当前未接入真实检索和个体评估，只能提供通用心理健康知识与支持原则。\n"
            f"- 建议：{recommendation}\n\n"
            "【适用边界】\n"
            "以上资料适用于校园心理健康支持、科普和求助引导；不能替代专业心理咨询、医学诊断或治疗。若用户存在即时安全风险，应优先联系现实支持或紧急救助。"
        )

    def _consult_answer(self, input_text: str) -> str:
        focus = self._focus_from(input_text)
        return (
            f"我能感觉到这件事已经占用了你不少精力。先不用急着把它归结成“我是不是不行”，我们可以先把它拆小一点看：{focus}。\n\n"
            "你现在可以先做几件很具体的小事：\n"
            "1. 用一两句话写下最困扰你的触发点，尽量区分“发生了什么”和“我脑子里正在担心什么”。\n"
            "2. 给身体一个短暂停顿：慢慢呼气 6 秒、吸气 4 秒，重复 3 轮，先把紧绷感降一点。\n"
            "3. 今天只选一个能完成的小动作，比如给老师/同学发一条确认信息、洗个热水澡、或提前 20 分钟放下手机。\n"
            "4. 如果这种状态已经持续两周以上，或明显影响上课、睡眠、饮食，建议尽快联系学校心理中心或辅导员。\n\n"
            "我们可以继续从最具体的地方开始。这个困扰最明显是在什么时候出现的？"
        )

    def _focus_from(self, input_text: str) -> str:
        normalized = input_text.lower()
        if self._contains_any(normalized, "低落", "抑郁", "难过", "没动力", "想哭", "压抑"):
            return "你现在的低落感值得被认真看见，而不是被简单劝成“想开点”"
        if self._contains_any(normalized, "睡不着", "失眠", "睡眠", "insomnia"):
            return "你提到的睡眠问题可能正在放大白天的疲惫和焦虑"
        if self._contains_any(normalized, "考试", "考研", "学习", "挂科", "作业", "论文"):
            return "你面对的学习或考试压力需要被拆成可处理的任务，而不是一次性压在心里"
        if self._contains_any(normalized, "分手", "恋爱", "亲密关系", "关系", "室友", "朋友", "社交"):
            return "关系里的不确定和消耗很容易让人反复想、反复内耗"
        if self._contains_any(normalized, "焦虑", "紧张", "害怕", "恐惧", "压力", "烦躁"):
            return "你身体和脑子都像是在持续警觉，所以会很累"
        return "先抓住一个最让你难受的场景，比泛泛地处理全部情绪更容易开始"

    def _last_user_message(self, messages: list[AiMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return messages[-1].content if messages else ""

    def _current_input(self, input_text: str) -> str:
        marker = "当前输入："
        index = input_text.rfind(marker)
        if index < 0:
            return input_text
        return input_text[index + len(marker) :].strip()

    def _current_question(self, input_text: str) -> str:
        marker = "用户问题："
        index = input_text.rfind(marker)
        if index < 0:
            return self._current_input(input_text)
        current = input_text[index + len(marker) :].strip()
        for stop_marker in ("\n\n已有 Observation：", "\n\n能力调用观察结果："):
            stop_index = current.find(stop_marker)
            if stop_index >= 0:
                current = current[:stop_index].strip()
        return current

    def _has_high_risk_signal(self, text: str) -> bool:
        return any(word in text for word in HIGH_RISK_WORDS)

    def _has_consult_signal(self, text: str) -> bool:
        return any(word in text for word in CONSULT_WORDS)

    def _contains_any(self, text: str, *words: str) -> bool:
        return any(word in text for word in words)

    def _should_use_diagnostic_swarm(self, text: str) -> bool:
        if self._has_high_risk_signal(text):
            return True
        functional_impairment = self._contains_any(
            text,
            "无法上课",
            "不能上课",
            "上不了课",
            "无法学习",
            "不能学习",
            "无法睡觉",
            "完全睡不着",
            "吃不下饭",
            "无法吃饭",
            "不能出门",
            "无法出门",
            "影响上课",
            "影响学习",
            "影响生活",
        )
        worsening = self._contains_any(text, "越来越", "明显加重", "加重", "急剧", "撑不住")
        asks_risk = self._contains_any(text, "严重吗", "风险", "需要就医", "要不要去", "安全吗", "危险吗")
        if functional_impairment or (worsening and asks_risk):
            return True
        distress_groups = (
            ("低落", "抑郁", "难过", "没动力", "想哭", "压抑"),
            ("焦虑", "紧张", "害怕", "恐惧", "烦躁"),
            ("睡不着", "失眠", "睡眠"),
            ("考试", "学习", "挂科", "论文", "作业"),
            ("关系", "室友", "朋友", "分手", "社交"),
            ("胸闷", "心慌", "头痛", "胃痛", "发抖", "出汗"),
        )
        matched_groups = sum(1 for group in distress_groups if self._contains_any(text, *group))
        return matched_groups >= 3 and asks_risk


class OllamaAiClient(AiClient):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60)

    def _payload(self, messages: list[AiMessage], stream: bool) -> dict:
        return {
            "model": self.settings.ollama_model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": stream,
            "options": {
                "temperature": self.settings.ai_temperature,
                "num_predict": self.settings.ai_max_tokens,
                "top_p": 0.85,
                "repeat_penalty": 1.12,
            },
        }

    async def complete(self, messages: list[AiMessage]) -> str:
        response = await self.client.post("/api/chat", json=self._payload(messages, stream=False))
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def stream(self, messages: list[AiMessage]) -> AsyncIterator[str]:
        async with self.client.stream("POST", "/api/chat", json=self._payload(messages, stream=True)) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = httpx.Response(200, content=line).json()
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


class OpenAiClient(AiClient):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    async def complete(self, messages: list[AiMessage]) -> str:
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            temperature=self.settings.ai_temperature,
            max_tokens=self.settings.ai_max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[AiMessage]) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            temperature=self.settings.ai_temperature,
            max_tokens=self.settings.ai_max_tokens,
            stream=True,
        )
        async for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token


def create_ai_client(settings: Settings) -> AiClient:
    if settings.ai_provider == "ollama":
        return OllamaAiClient(settings)
    if settings.ai_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("AI_PROVIDER=openai requires OPENAI_API_KEY.")
        return OpenAiClient(settings)
    return HeuristicAiClient()
