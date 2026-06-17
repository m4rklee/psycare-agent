metadata = {
    "name": "disease_code",
    "description": "Map mental-health support topics to non-diagnostic internal category codes.",
}


async def run(context, input):
    from skills._shared import HIGH_RISK_TERMS, contains_any, first_text

    text = first_text(input or {}, "user_input", "query", "topic")
    if contains_any(text, HIGH_RISK_TERMS):
        code = "PSY-SAFETY"
        category = "安全风险与危机支持"
    elif contains_any(text, ("睡不着", "失眠", "焦虑", "压力", "紧张", "考试", "学习")):
        code = "PSY-STRESS-SLEEP"
        category = "压力、焦虑与睡眠支持"
    elif contains_any(text, ("低落", "抑郁", "没动力", "难过", "想哭", "绝望")):
        code = "PSY-MOOD"
        category = "情绪低落与动力支持"
    elif contains_any(text, ("室友", "朋友", "分手", "恋爱", "社交", "关系", "孤独")):
        code = "PSY-RELATIONSHIP"
        category = "人际关系与支持系统"
    else:
        code = "PSY-GENERAL"
        category = "一般心理健康支持"
    answer = (
        "【心理健康内部分类】\n"
        f"分类码：{code}\n"
        f"类别：{category}\n\n"
        "说明：该分类仅用于内部任务分流和支持建议组织，不是医学诊断或疾病编码。"
    )
    return {
        "status": "success",
        "skill": "disease_code",
        "answer": answer,
        "code": code,
        "category": category,
        "coding_system": "campus_mental_health_internal",
        "diagnostic": False,
        "source": "rule_engine",
    }
