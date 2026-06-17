metadata = {
    "name": "deep_research",
    "description": "Synthesize campus mental-health evidence from multiple KnowledgeService queries.",
}


async def run(context, input):
    from skills._shared import first_text, format_knowledge_results, max_results, retrieve_knowledge

    input_data = input or {}
    query = first_text(input_data, "query", "user_input", "topic")
    top_k = max_results(input_data, default=3)
    queries = [
        query,
        f"{query} 校园心理支持 资源",
        f"{query} 危机干预 风险边界 循证建议",
    ]
    all_results = []
    unavailable_error = None
    for item in queries:
        results, unavailable = await retrieve_knowledge(context or {}, item, top_k)
        if unavailable is not None:
            unavailable_error = unavailable
            break
        all_results.extend(results)
    if unavailable_error is not None:
        unavailable_error["answer"] = "深度研究暂不可用：缺少 db/knowledge_service 或检索失败。"
        unavailable_error["skill"] = "deep_research"
        return unavailable_error

    best_by_key = {}
    for item in all_results:
        key = item.get("chunk_id") or f"{item.get('source')}:{item.get('content')}"
        if key not in best_by_key or item.get("score", 0.0) > best_by_key[key].get("score", 0.0):
            best_by_key[key] = item
    results = sorted(best_by_key.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
    findings = [str(item.get("content") or "")[:180] for item in results]
    confidence = "medium" if results else "low"
    evidence_strength = "中" if results else "弱"
    return {
        "status": "success",
        "skill": "deep_research",
        "answer": (
            format_knowledge_results("【校园心理健康深度研究】", query, results)
            + "\n\n【综合结论】\n"
            + ("已找到可用于内部分析的知识片段，建议结合学生当前风险线索和校园资源边界使用。" if results else "未找到足够资料，仅能提供通用支持原则。")
            + "\n\n【适用边界】\n本结果用于心理健康支持和内部 agent 辅助，不替代专业心理咨询、医学诊断或治疗。"
        ),
        "query": query,
        "queries": queries,
        "total_found": len(results),
        "results": results,
        "findings": findings,
        "confidence": confidence,
        "evidence_strength": evidence_strength,
        "source": "knowledge_service",
    }
