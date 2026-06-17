"""add agent framework state tables

Revision ID: 0002_agent_framework
Revises: 0001_initial_python_backend
Create Date: 2026-06-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_agent_framework"
down_revision: str | None = "0001_initial_python_backend"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(length=80), nullable=False),
        sa.Column("task_type", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("input_json", sa.Text(), nullable=False),
        sa.Column("output_json", sa.Text(), nullable=True),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_tasks_session_status", "agent_tasks", ["session_id", "status"])
    op.create_index("ix_agent_tasks_agent_name", "agent_tasks", ["agent_name"])

    op.create_table(
        "agent_memory_summaries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "content_hash", name="uq_agent_memory_session_hash"),
    )
    op.create_index(
        "ix_agent_memory_summaries_session_id",
        "agent_memory_summaries",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_memory_summaries_session_id", table_name="agent_memory_summaries")
    op.drop_table("agent_memory_summaries")
    op.drop_index("ix_agent_tasks_agent_name", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_session_status", table_name="agent_tasks")
    op.drop_table("agent_tasks")
