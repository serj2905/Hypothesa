"""Reliable interview sessions and completed interviews.

Revision ID: 20260710_0001
Revises: None
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260710_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interview_sessions",
        sa.Column("interview_id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("survey_id", sa.String(length=100), nullable=False),
        sa.Column("survey_version", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(length=20), nullable=False),
        sa.Column("session", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
        sa.Column("summary_attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column("processing_started_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_interview_sessions_user_status",
        "interview_sessions",
        ["user_id", "status"],
    )
    op.create_index(
        "ix_interview_sessions_status_updated",
        "interview_sessions",
        ["status", "updated_at"],
    )
    op.create_index(
        "uq_active_session_per_user_survey",
        "interview_sessions",
        ["user_id", "survey_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "completed_interviews",
        sa.Column("interview_id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("survey_id", sa.String(length=100), nullable=False),
        sa.Column("survey_version", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(length=20), nullable=False),
        sa.Column("age", sa.Integer()),
        sa.Column("city", sa.Text()),
        sa.Column("open_answers", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("faithful", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_completed_survey_created",
        "completed_interviews",
        ["survey_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_completed_survey_created", table_name="completed_interviews")
    op.drop_table("completed_interviews")
    op.drop_index("uq_active_session_per_user_survey", table_name="interview_sessions")
    op.drop_index("ix_interview_sessions_status_updated", table_name="interview_sessions")
    op.drop_index("ix_interview_sessions_user_status", table_name="interview_sessions")
    op.drop_table("interview_sessions")
