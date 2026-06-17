metadata = {
    "name": "search_knowledge",
    "description": "Search campus mental-health knowledge with the project KnowledgeService.",
}


async def run(context, input):
    from skills._shared import first_text, format_knowledge_results, max_results, retrieve_knowledge

    input_data = input or {}
    query = first_text(input_data, "query", "user_input", "topic")
    top_k = max_results(input_data, default=5)
    results, unavailable = await retrieve_knowledge(context or {}, query, top_k)
    if unavailable is not None:
        unavailable["answer"] = "知识库检索暂不可用：缺少 db/knowledge_service 或检索失败。"
        unavailable["skill"] = "search_knowledge"
        return unavailable
    return {
        "status": "success",
        "skill": "search_knowledge",
        "answer": format_knowledge_results("【校园心理健康知识库检索】", query, results),
        "query": query,
        "total_found": len(results),
        "results": results,
        "source": "knowledge_service",
    }
