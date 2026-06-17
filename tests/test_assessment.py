import pytest

from app.models.enums import EmotionLabel, RiskLevel
from app.services.ai import AiClient, AiMessage
from app.services.assessment import PsychologicalAssessmentService


class BrokenAiClient(AiClient):
    async def complete(self, messages: list[AiMessage]) -> str:
        raise RuntimeError("model unavailable")

    async def stream(self, messages: list[AiMessage]):
        if False:
            yield ""


@pytest.mark.asyncio
async def test_high_risk_rule_overrides_model() -> None:
    service = PsychologicalAssessmentService(BrokenAiClient())
    assessment = await service.assess("我不想活了")
    assert assessment.emotion == EmotionLabel.HIGH_RISK
    assert assessment.risk == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_assessment_falls_back_to_keywords() -> None:
    service = PsychologicalAssessmentService(BrokenAiClient())
    assessment = await service.assess("最近压力很大，总是睡不着")
    assert assessment.emotion == EmotionLabel.ANXIETY
    assert assessment.risk == RiskLevel.LOW
