import json
from dataclasses import dataclass

from app.models.enums import EmotionLabel, RiskLevel
from app.services.ai import AiClient, AiMessage
from app.services.prompts import psychology_prompt
from app.services.rules import has_high_risk_signal


@dataclass(frozen=True)
class PsychologyAssessment:
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    confidence: float
    summary: str


class PsychologicalAssessmentService:
    def __init__(self, ai_client: AiClient) -> None:
        self.ai_client = ai_client

    async def assess(self, user_input: str, history: list[AiMessage] | None = None) -> PsychologyAssessment:
        history = history or []
        if has_high_risk_signal(user_input.lower()):
            return PsychologyAssessment(
                EmotionLabel.HIGH_RISK,
                4.0,
                RiskLevel.HIGH,
                0.95,
                "Explicit high-risk signal detected.",
            )
        try:
            raw = await self.ai_client.complete(psychology_prompt(history, user_input))
            return self._normalize(self._parse_json(raw))
        except Exception:
            return self._heuristic(user_input)

    def _parse_json(self, raw: str) -> PsychologyAssessment:
        start = raw.find("{")
        end = raw.rfind("}")
        text = raw[start : end + 1] if start >= 0 and end > start else raw
        data = json.loads(text)
        emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
        score = float(data.get("emotionScore", self._score_for_emotion(emotion)))
        risk = RiskLevel(data.get("risk", self._risk_from_score(score).value).upper())
        confidence = float(data.get("confidence", 0.75))
        summary = str(data.get("summary", "Model assessment."))
        return PsychologyAssessment(emotion, score, risk, confidence, summary)

    def _normalize(self, assessment: PsychologyAssessment) -> PsychologyAssessment:
        score_risk = self._risk_from_score(assessment.emotion_score)
        risk = assessment.risk if self._risk_rank(assessment.risk) > self._risk_rank(score_risk) else score_risk
        if assessment.emotion == EmotionLabel.HIGH_RISK:
            risk = RiskLevel.HIGH
        return PsychologyAssessment(
            assessment.emotion,
            assessment.emotion_score,
            risk,
            max(0.0, min(1.0, assessment.confidence)),
            assessment.summary,
        )

    def _heuristic(self, user_input: str) -> PsychologyAssessment:
        normalized = user_input.lower()
        if any(word in normalized for word in ("抑郁", "低落", "压抑", "崩溃", "难过", "depress", "hopeless")):
            return PsychologyAssessment(EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, 0.75, "Low mood keywords detected.")
        if any(word in normalized for word in ("焦虑", "压力", "睡不着", "失眠", "anxious", "stress", "insomnia")):
            return PsychologyAssessment(EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, 0.72, "Anxiety or pressure keywords detected.")
        return PsychologyAssessment(EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, 0.66, "No obvious risk signal.")

    def _risk_from_score(self, score: float) -> RiskLevel:
        if score >= 4.0:
            return RiskLevel.HIGH
        if score >= 3.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _risk_rank(self, risk: RiskLevel) -> int:
        return {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}[risk]

    def _score_for_emotion(self, emotion: EmotionLabel) -> float:
        return {
            EmotionLabel.NORMAL: 0.0,
            EmotionLabel.ANXIETY: 2.0,
            EmotionLabel.DEPRESSED: 3.0,
            EmotionLabel.HIGH_RISK: 4.0,
        }[emotion]
