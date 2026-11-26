"""Service layer (M9A): runs get a row at submit, before the job is known.

- the job columns become optional (filled in by start_run)
- new: step (progress), facts_md and custom_prompt (inputs), status "queued"

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_STATUSES = ("running", "awaiting_review", "completed", "compile_failed", "failed")
NEW_STATUSES = ("queued", *OLD_STATUSES)
NOW_OPTIONAL = ("run_dir", "job_json", "analysis_json")


def upgrade() -> None:
    # SQLite can't change a column's NOT NULL in place: batch mode rebuilds the table.
    with op.batch_alter_table("run") as batch:
        for name in NOW_OPTIONAL:
            batch.alter_column(name, existing_type=sa.String(), nullable=True)
        batch.alter_column(
            "status",
            existing_type=sa.Enum(*OLD_STATUSES, name="runstatus"),
            type_=sa.Enum(*NEW_STATUSES, name="runstatus"),
            existing_nullable=False,
        )
        batch.add_column(sa.Column("step", sa.String(), nullable=True))
        # server_default fills the column for rows that already exist.
        batch.add_column(sa.Column("facts_md", sa.String(), nullable=False, server_default=""))
        batch.add_column(sa.Column("custom_prompt", sa.String(), nullable=False, server_default=""))


def downgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.drop_column("custom_prompt")
        batch.drop_column("facts_md")
        batch.drop_column("step")
        batch.alter_column(
            "status",
            existing_type=sa.Enum(*NEW_STATUSES, name="runstatus"),
            type_=sa.Enum(*OLD_STATUSES, name="runstatus"),
            existing_nullable=False,
        )
        for name in NOW_OPTIONAL:
            batch.alter_column(name, existing_type=sa.String(), nullable=False)
