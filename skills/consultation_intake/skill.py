metadata = {
    "name": "consultation_intake",
    "description": "Prepare initial consultation intake questions for a student support conversation.",
}


async def run(context, input):
    user_input = str(input.get("user_input") or "")
    focus = "general"
    if any(word in user_input for word in ("焦虑", "压力", "失眠", "睡不着")):
        focus = "stress_and_sleep"
    if any(word in user_input for word in ("不想活", "自杀", "自残", "想死")):
        focus = "safety"
    return {
        "focus": focus,
        "questions": [
            "这种状态最明显是在什么时候出现的？",
            "它已经持续多久，并且对睡眠、上课或饮食有什么影响？",
        ],
    }
