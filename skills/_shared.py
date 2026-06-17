from __future__ import annotations

from typing import Any


HIGH_RISK_TERMS = (
    "不想活",
    "活不下去",
    "撑不下去",
    "自杀",
    "自残",
    "轻生",
    "结束生命",
    "伤害自己",
    "伤人",
    "想死",
    "suicide",
    "kill myself",
    "self harm",
    "hurt myself",
)

MEDIUM_RISK_TERMS = (
    "持续",
    "两周",
    "一周",
    "越来越",
    "加重",
    "严重",
    "影响上课",
    "影响学习",
    "睡不着",
    "失眠",
    "吃不下",
    "崩溃",
    "绝望",
    "无助",
)

LOW_SUPPORT_TERMS = (
    "焦虑",
    "压力",
    "紧张",
    "低落",
    "难过",
    "烦躁",
    "孤独",
    "内耗",
    "人际",
    "室友",
    "考试",
    "学习",
    "睡眠",
)


def get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def first_text(input_data: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = input_data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def max_results(input_data: dict[str, Any], default: int = 3) -> int:
    try:
        value = int(input_data.get("max_results") or input_data.get("limit") or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, 10))


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(term.lower() in normalized for term in terms)


def split_terms(text: str) -> list[str]:
    separators = " \n\t,，。！？、；：,.!?;:"
    terms: list[str] = []
    current = []
    for char in text.lower():
        if char in separators:
            if current:
                terms.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        terms.append("".join(current))
    return [term for term in terms if len(term) >= 2]


def text_from_message(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", message) or "")


def role_from_message(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def score_text(query: str, text: str) -> int:
    terms = split_terms(query)
    normalized = text.lower()
    if not terms:
        return 1 if query and query.lower() in normalized else 0
    return sum(1 for term in terms if term in normalized)


def normalize_result(item: Any) -> dict[str, Any]:
    return {
        "chunk_id": get_value(item, "chunk_id"),
        "source": str(get_value(item, "source", "knowledge")),
        "content": str(get_value(item, "content", "")),
        "score": float(get_value(item, "score", 0.0) or 0.0),
    }


async def retrieve_knowledge(
    context: dict[str, Any],
    query: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    db = context.get("db")
    knowledge_service = context.get("knowledge_service")
    if db is None or knowledge_service is None:
        return [], {
            "status": "unavailable",
            "error": "KnowledgeService requires context['db'] and context['knowledge_service'].",
            "query": query,
            "total_found": 0,
            "results": [],
            "source": "knowledge_service",
        }
    try:
        raw_results = await knowledge_service.retrieve(db, query, top_k)
    except Exception as exc:
        return [], {
            "status": "unavailable",
            "error": str(exc),
            "query": query,
            "total_found": 0,
            "results": [],
            "source": "knowledge_service",
        }
    return [normalize_result(item) for item in raw_results], None


def format_knowledge_results(title: str, query: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f"{title}\n\n查询：{query}\n未找到可用资料。"
    lines = [title, "", f"查询：{query}", f"找到资料：{len(results)} 条", ""]
    for index, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        score_text_value = f"；相关度：{score:.2f}" if score else ""
        lines.append(f"【资料 {index}】来源：{item.get('source', 'knowledge')}{score_text_value}")
        lines.append(str(item.get("content") or ""))
        lines.append("")
    return "\n".join(lines).strip()


def rank_memory_items(query: str, items: list[Any], limit: int) -> list[dict[str, Any]]:
    ranked = []
    for index, item in enumerate(items):
        content = text_from_message(item)
        score = score_text(query, content)
        if score > 0 or not query:
            ranked.append(
                {
                    "index": index,
                    "role": role_from_message(item),
                    "content": content,
                    "score": score,
                }
            )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]
