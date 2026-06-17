from functools import lru_cache

from app.agent.infrastructure.runtime import AgentRuntime
from app.core.config import Settings, get_settings
from app.services.ai import AiClient, create_ai_client
from app.services.assessment import PsychologicalAssessmentService
from app.services.knowledge import AgenticRagService, KnowledgeService
from app.services.memory import ShortTermMemoryService
from app.services.multimodal import (
    MediaPipeClient,
    MultimodalFusionService,
    MultimodalInputService,
    WhisperClient,
)
from app.services.rules import IntentClassifier
from app.services.tools import AlertNotifier, ExcelReportWriter, ToolOrchestrationService


@lru_cache
def get_ai_client() -> AiClient:
    return create_ai_client(get_settings())


@lru_cache
def get_short_memory() -> ShortTermMemoryService:
    return ShortTermMemoryService(get_settings())


@lru_cache
def get_assessment_service() -> PsychologicalAssessmentService:
    return PsychologicalAssessmentService(get_ai_client())


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService(get_settings())


@lru_cache
def get_agentic_rag_service() -> AgenticRagService:
    return AgenticRagService(get_settings(), get_ai_client(), get_knowledge_service())


@lru_cache
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier(get_ai_client())


@lru_cache
def get_tool_service() -> ToolOrchestrationService:
    settings = get_settings()
    return ToolOrchestrationService(settings, ExcelReportWriter(settings), AlertNotifier(settings))


@lru_cache
def get_agent_runtime() -> AgentRuntime:
    settings = get_settings()
    return AgentRuntime(
        settings,
        get_ai_client(),
        knowledge_service=get_knowledge_service(),
        tool_service=get_tool_service(),
    )


@lru_cache
def get_multimodal_service() -> MultimodalInputService:
    settings = get_settings()
    return MultimodalInputService(
        WhisperClient(settings),
        MediaPipeClient(settings),
        get_assessment_service(),
        MultimodalFusionService(settings),
    )


def settings() -> Settings:
    return get_settings()
