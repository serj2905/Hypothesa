"""Persist topic-cycle quality and time-to-insight.

Revision ID: 20260710_0004
Revises: 20260710_0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260710_0004"
down_revision = "20260710_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "topic_runs",
        sa.Column("run_id", sa.Uuid(), primary_key=True),
        sa.Column("survey_id", sa.String(length=100), nullable=False),
        sa.Column("questionnaire_version", sa.Integer(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("data_cutoff", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_topic_runs_survey_completed",
        "topic_runs",
        ["survey_id", "completed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_topic_runs_survey_completed", table_name="topic_runs")
    op.drop_table("topic_runs")
