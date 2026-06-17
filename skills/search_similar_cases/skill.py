metadata = {
    "name": "search_similar_cases",
    "description": "Search similar summaries or historical cases from context long_memory.",
}


async def run(context, input):
    from skills._shared import first_text, max_results, rank_memory_items

    context_data = context or {}
    input_data = input or {}
    query = first_text(input_data, "query", "user_input", default="")
    limit = max_results(input_data, default=3)
    long_memory = list(context_data.get("long_memory") or [])
    matches = rank_memory_items(query, long_memory, limit)
    if not matches:
        answer = "【相似历史摘要】\n未找到匹配的长期记忆摘要。"
    else:
        lines = ["【相似历史摘要】", ""]
        for index, item in enumerate(matches, start=1):
            lines.append(f"【摘要 {index}】匹配分：{item.get('score', 0)}")
            lines.append(str(item.get("content") or ""))
            lines.append("")
        answer = "\n".join(lines).strip()
    return {
        "status": "success",
        "skill": "search_similar_cases",
        "answer": answer,
        "query": query,
        "total_found": len(matches),
        "results": matches,
        "source": "context.long_memory",
    }
