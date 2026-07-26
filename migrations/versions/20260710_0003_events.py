"""Append-only interview events.

Revision ID: 20260710_0003
Revises: 20260710_0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260710_0003"
down_revision = "20260710_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_events",
        sa.Column("event_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "interview_id",
            sa.Uuid(),
            sa.ForeignKey("interview_sessions.interview_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("question_id", sa.Integer()),
        sa.Column("details", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_interview_events_interview_time",
        "interview_events",
        ["interview_id", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_interview_events_interview_time", table_name="interview_events")
    op.drop_table("interview_events")
