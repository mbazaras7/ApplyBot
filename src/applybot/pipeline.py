"""Run the tailoring graph end to end, handing each review pause to a reviewer."""

import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel

from applybot.agent.graph import (
    RECURSION_LIMIT,
    build_graph,
    sqlite_checkpointer,
    start_timestamp,
)
from applybot.agent.state import ReviewDecision, ReviewRound, TailorState
from applybot.latex.compiler import CompileResult
from applybot.llm.client import LLMClient
from applybot.review.diff import ReviewRequest
from applybot.storage.history import HistoryStore
from applybot.storage.models import RunStatus
from applybot.storage.paths import CHECKPOINT_DB, HISTORY_DB, data_dir
from applybot.tools.analyse import JobAnalysis
from applybot.tools.cv_index import Embedder
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import EditPlan
from applybot.tools.validate import ValidatedPlan

Reviewer = Callable[[ReviewRequest], ReviewDecision]


class RunResult(BaseModel):
    run_id: int | None = None  # the run's row in the history database
    run_dir: Path
    job: JobPosting
    analysis: JobAnalysis
    validation: ValidatedPlan
    review_history: list[ReviewRound]
    original: CompileResult | None
    tailored: CompileResult | None

    @property
    def plan(self) -> EditPlan:
        return self.validation.plan

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"


def run_tailor(
    cv_path: Path,
    llm: LLMClient,
    reviewer: Reviewer,
    *,
    job_source: str | None = None,
    job_text: str | None = None,
    facts_path: Path | None = None,
    out_root: Path = Path("runs"),
    now: datetime | None = None,
    compile_pdf: bool = True,
    embedder: Embedder | None = None,
    data_folder: Path | None = None,
) -> RunResult:
    """Tailor the CV to a job given as a source (URL/file/-) or as text.

    The graph pauses at every review. Each pause is handed to `reviewer`, and its answer
    resumes the graph, until the reviewer approves and the CV is compiled.

    The run is recorded in the history database, and the graph's state is saved to the
    checkpoint database after every step, both in `data_folder` (default: the user's
    data folder).
    """
    if (job_source is None) == (job_text is None):
        raise ValueError("Pass exactly one of job_source or job_text.")

    # The checkpointer keeps one saved state per thread; each run gets its own thread.
    thread_id = str(uuid.uuid4())
    initial: TailorState = {
        "thread_id": thread_id,
        "cv_path": str(cv_path),
        "job_source": job_source,
        "job_text": job_text,
        "facts_path": str(facts_path) if facts_path else None,
        "out_root": str(out_root),
        "started": start_timestamp(now),
    }
    folder = data_folder or data_dir()
    store = HistoryStore(folder / HISTORY_DB)
    try:
        with sqlite_checkpointer(folder / CHECKPOINT_DB) as checkpointer:
            graph = build_graph(llm, compile_pdf, embedder, store=store, checkpointer=checkpointer)
            return drive(graph, store, reviewer, initial)
    finally:
        store.close()


def drive(graph: Any, store: HistoryStore, reviewer: Reviewer, initial: TailorState) -> RunResult:
    """Invoke the graph, answer each review pause, and collect the result."""
    config = {
        "configurable": {"thread_id": initial["thread_id"]},
        "recursion_limit": RECURSION_LIMIT,
    }
    try:
        state = graph.invoke(initial, config)
        while "__interrupt__" in state:  # paused at human_review
            request = state["__interrupt__"][0].value
            state = graph.invoke(Command(resume=reviewer(request)), config)
    except Exception as error:
        # The last saved state says whether a history row was created before the error.
        run_id = graph.get_state(config).values.get("run_id")
        if run_id is not None:
            store.update_run(run_id, status=RunStatus.failed, error=str(error) or repr(error))
        raise

    validation = ValidatedPlan(
        plan=state["plan"],
        attempts=state["attempts"],
        dropped=state.get("dropped", []),
        warnings=[p for p in state["problems"] if p.severity == "warning"],
    )
    return RunResult(
        run_id=state.get("run_id"),
        run_dir=Path(state["run_dir"]),
        job=state["job"],
        analysis=state["analysis"],
        validation=validation,
        review_history=state.get("review_history", []),
        original=state.get("original"),
        tailored=state.get("tailored"),
    )
