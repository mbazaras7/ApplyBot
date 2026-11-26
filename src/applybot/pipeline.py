"""Run the tailoring graph end to end, handing each review pause to a reviewer."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel

from applybot.agent.graph import build_graph, sqlite_checkpointer
from applybot.agent.state import ReviewDecision, ReviewRound
from applybot.latex.compiler import CompileResult
from applybot.llm.client import LLMClient
from applybot.review.diff import ReviewRequest
from applybot.service.runs import RunInputs, advance, record_failure, submit_run
from applybot.storage.history import HistoryStore
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
    custom_prompt: str = "",
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
    data folder). The API runs the same steps in the background (service/runs.py).
    """
    inputs = RunInputs(
        cv_tex=cv_path.read_bytes().decode("utf-8"),
        facts=facts_path.read_text(encoding="utf-8") if facts_path else "",
        custom_prompt=custom_prompt,
        job_source=job_source,
        job_text=job_text,
    )
    if (job_source is None) == (job_text is None):
        raise ValueError("Pass exactly one of job_source or job_text.")
    folder = data_folder or data_dir()
    store = HistoryStore(folder / HISTORY_DB)
    try:
        run_id, initial = submit_run(store, inputs, out_root, now)
        thread_id = initial["thread_id"]
        try:
            with sqlite_checkpointer(folder / CHECKPOINT_DB) as checkpointer:
                graph = build_graph(
                    llm, compile_pdf, embedder, store=store, checkpointer=checkpointer
                )
                progress = advance(graph, store, run_id, thread_id, initial)
                while progress.review is not None:
                    decision = reviewer(progress.review)
                    progress = advance(graph, store, run_id, thread_id, Command(resume=decision))
        except BaseException as error:  # incl. Ctrl+C: a run left "running" would be wrong
            record_failure(store, run_id, error)
            raise
    finally:
        store.close()
    return result_of(run_id, progress.values)


def result_of(run_id: int, state: dict[str, Any]) -> RunResult:
    validation = ValidatedPlan(
        plan=state["plan"],
        attempts=state["attempts"],
        dropped=state.get("dropped", []),
        warnings=[p for p in state["problems"] if p.severity == "warning"],
    )
    return RunResult(
        run_id=run_id,
        run_dir=Path(state["run_dir"]),
        job=state["job"],
        analysis=state["analysis"],
        validation=validation,
        review_history=state.get("review_history", []),
        original=state.get("original"),
        tailored=state.get("tailored"),
    )
