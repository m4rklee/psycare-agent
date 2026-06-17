from app.core.config import Settings
from app.schemas.api import AgentStatusResponse


def test_agent_status_response_keeps_java_fields() -> None:
    payload = AgentStatusResponse(
        provider="mock",
        model="heuristic-local",
        realModelEnabled=False,
        chromaEnabled=False,
        ragTopK=4,
        note="当前为本地 mock 演示模式，不会调用大模型。",
        knowledgeMode="local",
        mcpExcelMode="local",
        mcpEmailMode="log",
    ).model_dump()
    assert {"provider", "model", "realModelEnabled", "chromaEnabled", "ragTopK", "note"} <= set(payload)
