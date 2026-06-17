metadata = {
    "name": "search_history",
    "description": "Search current-session history from context.",
}


async def run(context, input):
    from skills._shared import first_text, max_results, rank_memory_items

    context_data = context or {}
    input_data = input or {}
    query = first_text(input_data, "query", "user_input", default="")
    limit = max_results(input_data, default=10)
    history = list(context_data.get("history") or [])
    matches = rank_memory_items(query, history, limit)
    if not matches:
        answer = "【当前会话历史】\n未找到匹配的历史消息。"
    else:
        lines = ["【当前会话历史】", ""]
        for item in matches:
            role = item.get("role") or "message"
            lines.append(f"- {role}: {item.get('content', '')}")
        answer = "\n".join(lines)
    return {
        "status": "success",
        "skill": "search_history",
        "answer": answer,
        "query": query,
        "total_messages": len(history),
        "total_found": len(matches),
        "results": matches,
        "session_id": context_data.get("session_id"),
        "source": "context.history",
    }
