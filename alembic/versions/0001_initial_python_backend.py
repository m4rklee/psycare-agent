"""initial python backend schema

Revision ID: 0001_initial_python_backend
Revises:
Create Date: 2026-06-08 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_python_backend"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=80), nullable=False),
        sa.Column("password", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=180), nullable=False),
        sa.Column("source_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "user_account_roles",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=80), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"]),
        sa.PrimaryKeyConstraint("user_id", "role"),
    )
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.Enum("USER", "ASSISTANT", "SYSTEM", name="messagerole"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "psychological_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("intent", sa.Enum("CHAT", "CONSULT", "RISK", name="intenttype"), nullable=False),
        sa.Column("emotion", sa.Enum("NORMAL", "ANXIETY", "DEPRESSED", "HIGH_RISK", name="emotionlabel"), nullable=False),
        sa.Column("emotion_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.Enum("LOW", "MEDIUM", "HIGH", name="risklevel"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("summary", sa.String(length=500), nullable=True),
        sa.Column("emotion_tags", sa.Text(), nullable=True),
        sa.Column("excel_status", sa.Enum("PENDING", "SUCCESS", "FAILED", "SKIPPED", name="toolstatus"), nullable=False),
        sa.Column("email_status", sa.Enum("PENDING", "SUCCESS", "FAILED", "SKIPPED", name="toolstatus"), nullable=False),
        sa.Column("tool_error", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["user_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "alert_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("report_id", sa.Integer(), nullable=False),
        sa.Column("recipient", sa.String(length=240), nullable=False),
        sa.Column("status", sa.Enum("PENDING", "SUCCESS", "FAILED", "SKIPPED", name="toolstatus"), nullable=False),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["report_id"], ["psychological_reports.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("alert_records")
    op.drop_table("psychological_reports")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("user_account_roles")
    op.drop_table("knowledge_chunks")
    op.drop_table("user_accounts")
    for enum_name in ("toolstatus", "risklevel", "emotionlabel", "intenttype", "messagerole"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
