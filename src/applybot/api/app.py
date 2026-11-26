"""The HTTP API: a thin layer that turns requests into RunManager calls.

Every endpoint is a plain `def`: FastAPI runs those in a thread pool, which suits code
that blocks on SQLite. The tailoring itself happens in the RunManager's worker thread,
so no request waits for the LLM; clients poll GET /runs/{id} instead.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from applybot.agent.state import ReviewDecision
from applybot.api.schemas import Accepted, RunCreated, RunDetail
from applybot.latex.parser import LatexParseError, parse
from applybot.review.diff import ReviewRequest
from applybot.service.runs import RunInputs, RunManager, RunNotFoundError, RunStateError
from applybot.storage.models import RunStatus, RunSummary
from applybot.tools.fetch_job import FetchError, file_to_text

OptionalText = Annotated[str | None, Form()]


def create_app(manager: RunManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        manager.close()  # on shutdown: let the running step finish, close the files

    app = FastAPI(title="ApplyBot", lifespan=lifespan)

    # Errors from the service become HTTP status codes in one place.
    @app.exception_handler(RunNotFoundError)
    def not_found(request: Request, error: RunNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_404_NOT_FOUND)

    @app.exception_handler(RunStateError)
    def conflict(request: Request, error: RunStateError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_409_CONFLICT)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(
        cv: Annotated[UploadFile, File(description="The LaTeX CV (main.tex).")],
        facts: Annotated[str, Form()] = "",
        custom_prompt: Annotated[str, Form()] = "",
        job_url: OptionalText = None,
        job_text: OptionalText = None,
        job_file: Annotated[UploadFile | None, File()] = None,
    ) -> RunCreated:
        """Start tailoring; the run continues in the background."""
        if sum(x is not None for x in (job_url, job_text, job_file)) != 1:
            raise bad_request("Give exactly one of job_url, job_text or job_file.")
        # Only web addresses: a file path here would let a caller read the server's files.
        if job_url is not None and not job_url.startswith(("http://", "https://")):
            raise bad_request("job_url must start with http:// or https://.")
        try:
            cv_tex = cv.file.read().decode("utf-8")
            parse(cv_tex)  # reject a CV that can't be parsed now, not minutes later
            if job_file is not None:
                job_text = file_to_text(job_file.filename or "job", job_file.file.read())
        except UnicodeDecodeError as error:
            raise bad_request("The CV must be a UTF-8 text file.") from error
        except (LatexParseError, FetchError) as error:
            raise bad_request(str(error)) from error

        inputs = RunInputs(
            cv_tex=cv_tex,
            facts=facts,
            custom_prompt=custom_prompt,
            job_source=job_url,
            job_text=job_text,
        )
        return RunCreated(id=manager.submit(inputs))

    @app.get("/runs")
    def list_runs(limit: int = 50) -> list[RunSummary]:
        return manager.store.list_runs(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: int) -> RunDetail:
        return RunDetail.of(manager.get(run_id))

    @app.get("/runs/{run_id}/review")
    def get_review(run_id: int) -> ReviewRequest:
        return manager.review(run_id)

    @app.post("/runs/{run_id}/decision", status_code=status.HTTP_202_ACCEPTED)
    def decide(run_id: int, decision: ReviewDecision) -> Accepted:
        """Approve, reject (with a comment) or send hand-edited text."""
        manager.decide(run_id, decision)
        return Accepted(id=run_id, status=RunStatus.queued)

    @app.get("/runs/{run_id}/pdf", response_class=Response)
    def get_pdf(run_id: int) -> Response:
        pdf = manager.get(run_id).tailored_pdf
        if pdf is None:
            raise not_ready(run_id, "PDF")
        return Response(pdf, media_type="application/pdf")

    @app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
    def get_report(run_id: int) -> PlainTextResponse:
        report = manager.get(run_id).report_md
        if report is None:
            raise not_ready(run_id, "report")
        return PlainTextResponse(report, media_type="text/markdown")

    return app


def bad_request(message: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, message)


def not_ready(run_id: int, what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} has no {what} yet.")
