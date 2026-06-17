from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import EmotionLabel, IntentType, MessageRole, RiskLevel, ToolStatus


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    roles: Mapped[list["UserAccountRole"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class UserAccountRole(Base):
    __tablename__ = "user_account_roles"

    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(80), primary_key=True)

    user: Mapped[UserAccount] = relationship(back_populates="roles")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[UserAccount] = relationship(lazy="selectin")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), nullable=False)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship(lazy="selectin")
    user: Mapped[UserAccount] = relationship(lazy="selectin")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(180), nullable=False)
    source_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PsychologicalReport(Base):
    __tablename__ = "psychological_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), nullable=False)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("chat_sessions.id"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[IntentType] = mapped_column(Enum(IntentType), nullable=False)
    emotion: Mapped[EmotionLabel] = mapped_column(Enum(EmotionLabel), nullable=False)
    emotion_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(String(500))
    emotion_tags: Mapped[Optional[str]] = mapped_column(Text)
    excel_status: Mapped[ToolStatus] = mapped_column(
        Enum(ToolStatus), nullable=False, default=ToolStatus.PENDING
    )
    email_status: Mapped[ToolStatus] = mapped_column(
        Enum(ToolStatus), nullable=False, default=ToolStatus.SKIPPED
    )
    tool_error: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[UserAccount] = relationship(lazy="selectin")
    session: Mapped[Optional[ChatSession]] = relationship(lazy="selectin")


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("psychological_reports.id"), nullable=False)
    recipient: Mapped[str] = mapped_column(String(240), nullable=False)
    status: Mapped[ToolStatus] = mapped_column(Enum(ToolStatus), nullable=False, default=ToolStatus.PENDING)
    error_message: Mapped[Optional[str]] = mapped_column(String(500))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    report: Mapped[PsychologicalReport] = relationship(lazy="selectin")


class AgentTaskRecord(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (
        Index("ix_agent_tasks_run_id_status", "run_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(80), nullable=False)
    task_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="PENDING")
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_json: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(String(500))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship(lazy="selectin")


class AgentMemorySummary(Base):
    __tablename__ = "agent_memory_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship(lazy="selectin")
