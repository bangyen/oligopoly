"""Add params column to runs table

Revision ID: 003
Revises: 002
Create Date: 2026-02-25 18:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable params JSON column to runs table."""
    op.add_column(
        "runs",
        sa.Column("params", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Remove params column from runs table."""
    op.drop_column("runs", "params")
