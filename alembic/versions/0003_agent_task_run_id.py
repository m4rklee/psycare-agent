"""add run id to agent tasks

Revision ID: 0003_agent_task_run_id
Revises: 0002_agent_framework
Create Date: 2026-06-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_agent_task_run_id"
down_revision: str | None = "0002_agent_framework"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_tasks",
        sa.Column("run_id", sa.String(length=36), nullable=False, server_default="legacy"),
    )
    op.create_index("ix_agent_tasks_run_id_status", "agent_tasks", ["run_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_agent_tasks_run_id_status", table_name="agent_tasks")
    op.drop_column("agent_tasks", "run_id")
