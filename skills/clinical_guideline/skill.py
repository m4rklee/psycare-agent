metadata = {
    "name": "clinical_guideline",
    "description": "Search campus mental-health support principles and crisis-intervention guidance.",
}


async def run(context, input):
    from skills._shared import first_text, format_knowledge_results, max_results, retrieve_knowledge

    input_data = input or {}
    query = first_text(input_data, "query", "user_input", "topic")
    guideline_query = f"{query} 校园心理支持 危机干预 原则 专业共识"
    top_k = max_results(input_data, default=3)
    results, unavailable = await retrieve_knowledge(context or {}, guideline_query, top_k)
    if unavailable is not None:
        unavailable["answer"] = "校园心理支持原则检索暂不可用：缺少 db/knowledge_service 或检索失败。"
        unavailable["skill"] = "clinical_guideline"
        return unavailable
    return {
        "status": "success",
        "skill": "clinical_guideline",
        "answer": format_knowledge_results("【校园心理支持原则检索】", guideline_query, results),
        "query": query,
        "total_found": len(results),
        "results": results,
        "source": "knowledge_service",
        "guideline_type": "campus_mental_health_support",
    }
