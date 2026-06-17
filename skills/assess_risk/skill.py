metadata = {
    "name": "assess_risk",
    "description": "Assess campus mental-health risk signals and urgency boundaries.",
}


async def run(context, input):
    from skills._shared import HIGH_RISK_TERMS, LOW_SUPPORT_TERMS, MEDIUM_RISK_TERMS, contains_any, first_text

    text = first_text(input or {}, "user_input", "query", "description")
    reasons = []
    if contains_any(text, HIGH_RISK_TERMS):
        risk_level = "emergency"
        urgency = "需要立即确认安全并连接现实支持"
        recommendation = "请立刻联系现实支持资源，例如身边可信任的人、辅导员、学校心理中心；如有即时危险，联系当地紧急救助。"
        reasons.append("表达中包含自伤、伤人或生命安全相关高风险线索。")
    elif contains_any(text, MEDIUM_RISK_TERMS):
        risk_level = "medium"
        urgency = "建议尽快联系学校心理中心或辅导员进一步评估"
        recommendation = "建议记录持续时间、功能影响和触发因素，并尽快预约学校心理中心或联系辅导员。"
        reasons.append("表达中包含持续、加重、睡眠/学习受影响或强烈无助等线索。")
    elif contains_any(text, LOW_SUPPORT_TERMS):
        risk_level = "low"
        urgency = "可先自我支持并观察变化"
        recommendation = "可以先使用稳定作息、呼吸放松、任务拆解和现实沟通；若持续加重再寻求专业支持。"
        reasons.append("表达中包含常见压力、焦虑、睡眠或人际困扰线索。")
    else:
        risk_level = "low"
        urgency = "暂无明显高风险线索"
        recommendation = "继续温和询问具体情境、持续时间和支持系统。"
        reasons.append("当前输入未识别到明显危险信号。")

    answer = (
        "【心理风险线索评估】\n"
        f"风险等级：{risk_level}\n"
        f"紧急程度：{urgency}\n\n"
        "识别依据：\n"
        + "\n".join(f"- {reason}" for reason in reasons)
        + f"\n\n建议：{recommendation}\n\n"
        "边界：本结果仅用于心理健康支持和内部辅助，不替代正式风险评估。"
    )
    return {
        "status": "success",
        "skill": "assess_risk",
        "answer": answer,
        "risk_level": risk_level,
        "urgency": urgency,
        "reasons": reasons,
        "recommendation": recommendation,
        "source": "rule_engine",
    }
