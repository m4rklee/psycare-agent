import re


DEFAULT_PSYCHOLOGY_DISCLAIMER = "以上信息仅供心理健康支持和科普参考，不能替代专业心理咨询、医学诊断或治疗。如有疑虑或风险，请及时联系专业人员。"
SECTION_RE = re.compile(r"【([^】]+)】")


def extract_section(text: str, title: str) -> str:
    marker = f"【{title}】"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    next_marker = text.find("【", start)
    end = next_marker if next_marker >= 0 else len(text)
    return text[start:end].strip()


def extract_numbered_suggestions(text: str, title: str = "核心建议", limit: int = 5) -> list[str]:
    section = extract_section(text, title)
    if not section:
        return []
    suggestions = re.findall(r"(?:^|\n)\s*\d+[.、]\s*([^\n]+)", section)
    return [suggestion.strip() for suggestion in suggestions if suggestion.strip()][:limit]


def extract_disclaimer(text: str, default: str = DEFAULT_PSYCHOLOGY_DISCLAIMER) -> str:
    section = extract_section(text, "免责声明")
    return section.strip() if section.strip() else default


def scan_risk_level(text: str) -> str:
    if "风险等级" not in text:
        return "unknown"
    match = re.search(r"风险等级\s*[:：]\s*([^\n]+)", text)
    target = match.group(1) if match else text
    if any(marker in target for marker in ("紧急", "高", "HIGH", "high")):
        return "high"
    if any(marker in target for marker in ("中", "MEDIUM", "medium")):
        return "medium"
    if any(marker in target for marker in ("低", "LOW", "low")):
        return "low"
    return "unknown"


def scan_evidence_level(text: str) -> str:
    if "A级" in text or "A 级" in text:
        return "A"
    if "B级" in text or "B 级" in text:
        return "B"
    if "C级" in text or "C 级" in text:
        return "C"
    return "unknown"


def count_literature_references(text: str) -> int:
    return text.count("文献") + text.count("资料")
