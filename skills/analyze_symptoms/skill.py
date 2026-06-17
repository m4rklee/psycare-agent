metadata = {
    "name": "analyze_symptoms",
    "description": "Analyze emotional, cognitive, physical, behavioral and support-system patterns.",
}


async def run(context, input):
    from skills._shared import contains_any, first_text

    text = first_text(input or {}, "user_input", "query", "description")
    categories = []
    if contains_any(text, ("焦虑", "紧张", "害怕", "恐惧", "烦躁", "压力")):
        categories.append({"id": "emotion_anxiety", "name": "情绪紧张/焦虑线索"})
    if contains_any(text, ("低落", "难过", "没动力", "绝望", "想哭", "抑郁")):
        categories.append({"id": "mood_low", "name": "低落/动力下降线索"})
    if contains_any(text, ("睡不着", "失眠", "早醒", "困", "疲惫", "吃不下", "胸闷", "心慌")):
        categories.append({"id": "physical_sleep", "name": "睡眠或躯体反应线索"})
    if contains_any(text, ("逃课", "不想上课", "拖延", "回避", "效率", "挂科", "学习")):
        categories.append({"id": "function_study", "name": "学习/功能受影响线索"})
    if contains_any(text, ("室友", "朋友", "分手", "恋爱", "社交", "孤独", "关系")):
        categories.append({"id": "relationship", "name": "人际关系线索"})
    if not categories:
        categories.append({"id": "general_distress", "name": "一般心理困扰线索"})

    patterns = [item["name"] for item in categories]
    questions = [
        "这种状态持续了多久，最近是否有加重？",
        "它对睡眠、上课、饮食或人际关系影响到什么程度？",
        "身边是否有可以联系的同学、家人、老师、辅导员或学校心理中心？",
    ]
    answer = (
        "【心理困扰模式分析】\n"
        f"用户描述：{text}\n\n"
        "识别到的模式：\n"
        + "\n".join(f"- {pattern}" for pattern in patterns)
        + "\n\n建议进一步了解：\n"
        + "\n".join(f"- {question}" for question in questions)
        + "\n\n边界：以上只做线索整理，不输出诊断结论。"
    )
    return {
        "status": "success",
        "skill": "analyze_symptoms",
        "answer": answer,
        "patterns": patterns,
        "categories": categories,
        "follow_up_questions": questions,
        "source": "rule_engine",
    }
