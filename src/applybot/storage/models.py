"""The run history table. SQLModel classes are Pydantic models that are also SQL tables."""

from datetime import datetime
from enum import StrEnum

from sqlmodel import Field, SQLModel


class RunStatus(StrEnum):
    running = "running"  # generating and validating edits
    awaiting_review = "awaiting_review"  # paused at the review step (resumable)
    completed = "completed"  # approved, compiled (or compiling was off) and reported
    compile_failed = "compile_failed"  # approved, but the tailored CV didn't compile
    failed = "failed"  # stopped by an error; see `error`


class Run(SQLModel, table=True):
    """One tailoring run: what went in, what came out, and where it got to.

    Times are timezone-aware and stored in UTC; convert with .astimezone() to show them.
    JSON columns hold the same text as the run folder's files, so a run can be shown
    or exported without the folder.
    """

    id: int | None = Field(default=None, primary_key=True)
    thread_id: str = Field(unique=True, index=True)  # the checkpointer's key for this run
    created_at: datetime = Field(index=True)
    updated_at: datetime
    status: RunStatus = RunStatus.running
    error: str | None = None

    company: str | None = None
    title: str | None = None
    fit_score: int | None = None
    run_dir: str

    # Inputs
    base_cv_tex: str
    job_json: str
    analysis_json: str

    # Outputs, filled in when the run finishes
    edits_json: str | None = None
    validation_json: str | None = None
    review_json: str | None = None
    tailored_tex: str | None = None
    tailored_pdf: bytes | None = None
    report_md: str | None = None
    original_pages: int | None = None
    tailored_pages: int | None = None


class RunSummary(SQLModel):
    """What a list of runs needs: no CV, report or PDF. (No `table=True`: not a table,
    just the shape of a query result.)"""

    id: int
    thread_id: str
    created_at: datetime
    status: RunStatus
    company: str | None
    title: str | None
    fit_score: int | None


# The Run columns a summary is read from, in the order of RunSummary's fields.
SUMMARY_COLUMNS = tuple(getattr(Run, name) for name in RunSummary.model_fields)
