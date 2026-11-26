"""Baseline: the run table as Milestone 8 created it.

Revision ID: 0001
Revises:
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

M8_STATUSES = ("running", "awaiting_review", "completed", "compile_failed", "failed")


def upgrade() -> None:
    op.create_table(
        "run",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Enum(*M8_STATUSES, name="runstatus"), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("company", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("fit_score", sa.Integer(), nullable=True),
        sa.Column("run_dir", sa.String(), nullable=False),
        sa.Column("base_cv_tex", sa.String(), nullable=False),
        sa.Column("job_json", sa.String(), nullable=False),
        sa.Column("analysis_json", sa.String(), nullable=False),
        sa.Column("edits_json", sa.String(), nullable=True),
        sa.Column("validation_json", sa.String(), nullable=True),
        sa.Column("review_json", sa.String(), nullable=True),
        sa.Column("tailored_tex", sa.String(), nullable=True),
        sa.Column("tailored_pdf", sa.LargeBinary(), nullable=True),
        sa.Column("report_md", sa.String(), nullable=True),
        sa.Column("original_pages", sa.Integer(), nullable=True),
        sa.Column("tailored_pages", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_thread_id", "run", ["thread_id"], unique=True)
    op.create_index("ix_run_created_at", "run", ["created_at"])


def downgrade() -> None:
    op.drop_table("run")
