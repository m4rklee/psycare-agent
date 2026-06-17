from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import EmotionLabel, IntentType, MessageRole, RiskLevel, ToolStatus


class ApiMessage(BaseModel):
    message: str


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class ChatRequest(BaseModel):
    sessionId: Optional[str] = None
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        return _not_blank(value)


class ChatStreamEvent(BaseModel):
    type: str
    sessionId: Optional[str] = None
    content: str = ""
    displayText: Optional[str] = None
    agentRunId: Optional[str] = None
    agentDispatchedAgents: Optional[list[str]] = None
    agentTimeoutOccurred: Optional[bool] = None
    agentDecompositionError: Optional[str] = None
    visualEmotion: Optional[str] = None
    visualRiskLevel: Optional[str] = None
    visualConfidence: Optional[float] = None
    visualEvidence: Optional[str] = None
    intent: Optional[IntentType] = None
    riskLevel: Optional[RiskLevel] = None


class KnowledgeIngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=180)
    content: str = Field(min_length=1)

    @field_validator("source", "content")
    @classmethod
    def field_not_blank(cls, value: str) -> str:
        return _not_blank(value)


class KnowledgeIngestResponse(BaseModel):
    source: str
    chunks: int


class ProfileAuthority(BaseModel):
    authority: str


class ProfileResponse(BaseModel):
    id: int
    username: str
    displayName: str
    roles: list[ProfileAuthority]


class AgentStatusResponse(BaseModel):
    provider: str
    model: str
    realModelEnabled: bool
    chromaEnabled: bool
    ragTopK: int
    note: str
    knowledgeMode: str
    mcpExcelMode: str
    mcpEmailMode: str


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    userId: int
    username: str
    sessionId: Optional[str]
    intent: IntentType
    emotion: EmotionLabel
    emotionScore: float
    riskLevel: RiskLevel
    confidence: float
    summary: Optional[str]
    emotionTags: Optional[str]
    excelStatus: ToolStatus
    emailStatus: ToolStatus
    createdAt: datetime


class ExcelRecordResponse(BaseModel):
    reportId: int
    userId: int
    username: str
    sessionId: Optional[str]
    intent: IntentType
    emotion: EmotionLabel
    emotionScore: float
    riskLevel: RiskLevel
    confidence: float
    summary: Optional[str]
    emotionTags: Optional[str]
    content: str
    excelStatus: ToolStatus
    createdAt: datetime


class AlertRecordResponse(BaseModel):
    id: int
    reportId: int
    userId: int
    username: str
    sessionId: Optional[str]
    riskLevel: RiskLevel
    summary: Optional[str]
    recipient: str
    status: ToolStatus
    errorMessage: Optional[str]
    attempts: int
    createdAt: datetime
    updatedAt: datetime


class ConversationMessageResponse(BaseModel):
    id: int
    role: MessageRole
    content: str
    createdAt: datetime


class ChatSessionSummaryResponse(BaseModel):
    sessionId: str
    title: str
    createdAt: datetime
    updatedAt: datetime


class ConversationResponse(BaseModel):
    sessionId: str
    title: str
    userId: int
    username: str
    displayName: str
    createdAt: datetime
    updatedAt: datetime
    messages: list[ConversationMessageResponse]
