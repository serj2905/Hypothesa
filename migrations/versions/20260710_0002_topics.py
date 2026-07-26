"""Topic registry, assignments and questionnaire versions.

Revision ID: 20260710_0002
Revises: 20260710_0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260710_0002"
down_revision = "20260710_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "topics",
        sa.Column("topic_id", sa.Uuid(), primary_key=True),
        sa.Column("survey_id", sa.String(length=100), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("keywords", postgresql.JSONB(), nullable=False),
        sa.Column("centroid", postgresql.JSONB(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("mention_count", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_topics_survey_active_rating", "topics", ["survey_id", "active", "rating"]
    )
    op.create_table(
        "topic_assignments",
        sa.Column("document_id", sa.Uuid(), primary_key=True),
        sa.Column("survey_id", sa.String(length=100), nullable=False),
        sa.Column("interview_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("topic_id", sa.Uuid(), sa.ForeignKey("topics.topic_id")),
        sa.Column("probability", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_topic_assignments_survey_topic",
        "topic_assignments",
        ["survey_id", "topic_id"],
    )
    op.create_table(
        "questionnaire_versions",
        sa.Column("survey_id", sa.String(length=100), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("questions", postgresql.JSONB(), nullable=False),
        sa.Column("source_topic_ids", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("questionnaire_versions")
    op.drop_index("ix_topic_assignments_survey_topic", table_name="topic_assignments")
    op.drop_table("topic_assignments")
    op.drop_index("ix_topics_survey_active_rating", table_name="topics")
    op.drop_table("topics")
