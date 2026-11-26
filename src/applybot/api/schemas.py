"""What the API sends back. Requests reuse the app's own models (e.g. ReviewDecision)."""

from datetime import datetime

from pydantic import BaseModel

from applybot.storage.models import Run, RunStatus


class RunCreated(BaseModel):
    id: int


class Accepted(BaseModel):
    """The request was queued; poll GET /runs/{id} for progress."""

    id: int
    status: RunStatus


class RunDetail(BaseModel):
    """One run's progress and results, without the stored files themselves."""

    id: int
    status: RunStatus
    step: str | None
    error: str | None
    company: str | None
    title: str | None
    fit_score: int | None
    created_at: datetime
    updated_at: datetime
    original_pages: int | None
    tailored_pages: int | None
    has_pdf: bool
    has_report: bool

    @classmethod
    def of(cls, run: Run) -> "RunDetail":
        assert run.id is not None
        return cls(
            id=run.id,
            status=run.status,
            step=run.step,
            error=run.error,
            company=run.company,
            title=run.title,
            fit_score=run.fit_score,
            created_at=run.created_at,
            updated_at=run.updated_at,
            original_pages=run.original_pages,
            tailored_pages=run.tailored_pages,
            has_pdf=run.tailored_pdf is not None,
            has_report=run.report_md is not None,
        )
