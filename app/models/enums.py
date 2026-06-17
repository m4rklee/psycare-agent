from enum import StrEnum


class EmotionLabel(StrEnum):
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"


class IntentType(StrEnum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


class MessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ToolStatus(StrEnum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
