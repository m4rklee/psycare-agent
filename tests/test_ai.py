import json

import pytest

from app.services.ai import AiMessage, HeuristicAiClient
from app.services.prompts import (
    answer_system_prompt,
    intent_prompt,
    psychology_prompt,
    rag_plan_prompt,
    rag_review_prompt,
)
from app.models.enums import IntentType, RiskLevel


@pytest.mark.asyncio
async def test_mock_ai_classifies_intent_prompt() -> None:
    client = HeuristicAiClient()
    assert await client.complete(intent_prompt([], "我最近压力很大睡不着")) == "CONSULT"
    assert await client.complete(intent_prompt([], "我不想活了")) == "RISK"
    assert await client.complete(intent_prompt([], "Python 后端是什么")) == "CHAT"


@pytest.mark.asyncio
async def test_mock_ai_returns_psychology_json() -> None:
    client = HeuristicAiClient()
    payload = json.loads(await client.complete(psychology_prompt([], "我最近压力很大睡不着")))
    assert payload["emotion"] == "ANXIETY"
    assert payload["risk"] == "LOW"
    assert payload["emotionScore"] == 2.2


@pytest.mark.asyncio
async def test_mock_ai_returns_agentic_rag_json() -> None:
    client = HeuristicAiClient()
    plan = json.loads(await client.complete(rag_plan_prompt([], "我最近很焦虑")))
    assert len(plan["queries"]) == 3
    assert "校园心理" in plan["reason"]

    review = json.loads(await client.complete(rag_review_prompt("我最近很焦虑", "校园心理中心支持")))
    assert review["sufficient"] is True
    assert review["followUpQueries"] == []


@pytest.mark.asyncio
async def test_mock_ai_streams_contextual_answer() -> None:
    client = HeuristicAiClient()
    chunks = [
        token
        async for token in client.stream(
            [
                AiMessage("system", "你需要优先基于下方 Agentic RAG 计划回答。\n检索知识：校园心理中心。"),
                AiMessage("user", "我最近焦虑"),
            ]
        )
    ]
    text = "".join(chunks)
    assert "具体的小事" in text
    assert "什么时候出现" in text


def test_chat_prompt_keeps_java_constraints() -> None:
    prompt = answer_system_prompt(IntentType.CHAT, RiskLevel.LOW, "", "student").content
    assert "不要把普通聊天强行引导成心理咨询" in prompt
    assert "普通问候用 1 句回答" in prompt
    assert "不要自己续写用户问题" in prompt
    assert "学生显示名：student" in prompt


def test_consult_prompt_keeps_java_rag_and_multimodal_constraints() -> None:
    prompt = answer_system_prompt(IntentType.CONSULT, RiskLevel.LOW, "Agentic RAG：未检索到可用知识。", "student").content
    assert "如果上下文中出现【多模态分析记忆】或【多模态后台分析】" in prompt
    assert "如果复核认为知识不足或检索知识不足" in prompt
    assert "默认用 2-4 个短段落或要点回答" in prompt
