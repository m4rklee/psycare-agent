metadata = {
    "name": "recommend_lifestyle",
    "description": "Recommend practical lifestyle and campus-support steps for mental-health concerns.",
}


async def run(context, input):
    from skills._shared import contains_any, first_text

    text = first_text(input or {}, "user_input", "query", "topic")
    suggestions = [
        "把今天最困扰的一件事写成一句话，区分事实、想法和感受。",
        "做 3 轮慢呼吸或短暂离开刺激源，先降低身体紧绷感。",
        "联系一个现实中可信任的人，简单说明“我最近有点撑不住，想有人陪我聊一下”。",
    ]
    categories = ["stabilization", "social_support"]
    if contains_any(text, ("睡不着", "失眠", "睡眠", "困", "早醒")):
        categories.append("sleep")
        suggestions.extend(
            [
                "睡前 30 分钟减少屏幕刺激，把担心的事情写到纸上，留到第二天处理。",
                "如果连续多日明显失眠并影响上课，建议联系学校心理中心或校医院咨询。",
            ]
        )
    if contains_any(text, ("考试", "学习", "论文", "作业", "挂科", "压力")):
        categories.append("study_stress")
        suggestions.append("把学习任务拆成 25 分钟内可完成的小块，先做最容易启动的一步。")
    if contains_any(text, ("室友", "朋友", "分手", "恋爱", "关系", "社交")):
        categories.append("relationship")
        suggestions.append("先用低冲突表达描述自己的感受和需求，避免在情绪最高点做重大决定。")
    answer = (
        "【生活方式与校园支持建议】\n"
        + "\n".join(f"{index}. {item}" for index, item in enumerate(suggestions, start=1))
        + "\n\n边界：这些建议用于低风险心理支持；若有自伤、伤人或即时危险，请优先寻求现实支持或紧急帮助。"
    )
    return {
        "status": "success",
        "skill": "recommend_lifestyle",
        "answer": answer,
        "query": text,
        "categories": categories,
        "suggestions": suggestions,
        "source": "rule_engine",
    }
