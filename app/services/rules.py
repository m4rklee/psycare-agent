import re

from app.models.enums import IntentType
from app.services.ai import AiClient, AiMessage
from app.services.prompts import intent_prompt

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

GENERAL_TASK_WORDS = (
    "java",
    "python",
    "javascript",
    "代码",
    "编程",
    "程序",
    "算法",
    "数据库",
    "spring",
    "maven",
    "前端",
    "后端",
    "项目",
    "接口",
    "bug",
    "报错",
    "作业",
    "论文",
    "翻译",
    "总结",
    "解释",
    "怎么写",
    "如何",
    "是什么",
    "为什么",
    "给我",
    "帮我",
    "推荐",
    "查询",
    "天气",
    "路线",
)


def has_high_risk_signal(text: str) -> bool:
    return any(word in text for word in HIGH_RISK_WORDS)


def has_consult_signal(text: str) -> bool:
    return any(word in text for word in CONSULT_WORDS)


def sanitize(text: str | None) -> str:
    if not text or not text.strip():
        return ""
    value = text
    value = re.sub(r"1[3-9]\d{9}", "[手机号]", value)
    value = re.sub(r"(?i)(学号|student\s*id)[:：\s]*[A-Za-z0-9_-]{6,20}", r"\1:[学号]", value)
    value = re.sub(r"(?i)(身份证|id\s*card)[:：\s]*[0-9xX]{15,18}", r"\1:[证件号]", value)
    value = re.sub(r"我叫[\u4e00-\u9fa5]{2,4}", "我叫[姓名]", value)
    value = re.sub(r"我是[\u4e00-\u9fa5]{2,4}", "我是[姓名]", value)
    return value


class IntentClassifier:
    def __init__(self, ai_client: AiClient) -> None:
        self.ai_client = ai_client

    async def classify(self, user_input: str, history: list[AiMessage]) -> IntentType:
        normalized = user_input.lower()
        if has_high_risk_signal(normalized):
            return IntentType.RISK
        if self._is_general_task(normalized):
            return IntentType.CHAT
        try:
            label = (await self.ai_client.complete(intent_prompt(history, user_input))).strip().upper()
            if "RISK" in label:
                return IntentType.RISK
            if "CONSULT" in label:
                return IntentType.CONSULT
            if "CHAT" in label:
                return IntentType.CHAT
        except Exception:
            pass
        if has_consult_signal(normalized) or self._has_recent_consult_context(history):
            return IntentType.CONSULT
        return IntentType.CHAT

    def _is_general_task(self, text: str) -> bool:
        return not has_consult_signal(text) and any(word in text for word in GENERAL_TASK_WORDS)

    def _has_recent_consult_context(self, history: list[AiMessage]) -> bool:
        return any(has_consult_signal(message.content.lower()) for message in history[-6:])
