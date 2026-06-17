import pytest

from app.models.enums import IntentType
from app.services.ai import HeuristicAiClient
from app.services.rules import IntentClassifier, has_high_risk_signal, sanitize


def test_high_risk_signal_detects_self_harm_language() -> None:
    assert has_high_risk_signal("我真的不想活了")


def test_sanitize_removes_sensitive_identifiers() -> None:
    text = sanitize("我叫张三，手机号是13812345678，学号: 20240001")
    assert "[手机号]" in text
    assert "学号:[学号]" in text
    assert "我叫[姓名]" in text


@pytest.mark.asyncio
async def test_general_task_routes_to_chat() -> None:
    classifier = IntentClassifier(HeuristicAiClient())
    assert await classifier.classify("Python 后端是什么", []) == IntentType.CHAT


@pytest.mark.asyncio
async def test_consult_signal_routes_to_consult() -> None:
    classifier = IntentClassifier(HeuristicAiClient())
    assert await classifier.classify("我最近压力很大睡不着", []) == IntentType.CONSULT
