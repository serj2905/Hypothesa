"""Track the completed-record cutoff used by automatic topic refresh.

Revision ID: 20260710_0005
Revises: 20260710_0004
"""

import sqlalchemy as sa
from alembic import op

revision = "20260710_0005"
down_revision = "20260710_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "topic_runs",
        sa.Column("data_cutoff_created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_column("topic_runs", "data_cutoff_created_at")
