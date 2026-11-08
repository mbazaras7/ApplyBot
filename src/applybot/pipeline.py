"""Run the tailoring graph end to end, handing each review pause to a reviewer."""

import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from langgraph.types import Command
from pydantic import BaseModel

from applybot.agent.graph import RECURSION_LIMIT, build_graph, start_timestamp
from applybot.agent.state import ReviewDecision, ReviewRound, TailorState
from applybot.latex.compiler import CompileResult
from applybot.llm.client import LLMClient
from applybot.review.diff import ReviewRequest
from applybot.tools.analyse import JobAnalysis
from applybot.tools.cv_index import Embedder
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import EditPlan
from applybot.tools.validate import ValidatedPlan

Reviewer = Callable[[ReviewRequest], ReviewDecision]


class RunResult(BaseModel):
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
) -> RunResult:
    """Tailor the CV to a job given as a source (URL/file/-) or as text.

    The graph pauses at every review. Each pause is handed to `reviewer`, and its answer
    resumes the graph, until the reviewer approves and the CV is compiled.
    """
    if (job_source is None) == (job_text is None):
        raise ValueError("Pass exactly one of job_source or job_text.")

    graph = build_graph(llm, compile_pdf=compile_pdf, embedder=embedder)
    # The checkpointer keeps one saved state per thread; each run gets its own thread.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
    initial: TailorState = {
        "cv_path": str(cv_path),
        "job_source": job_source,
        "job_text": job_text,
        "facts_path": str(facts_path) if facts_path else None,
        "out_root": str(out_root),
        "started": start_timestamp(now),
    }

    state = graph.invoke(initial, config)
    while "__interrupt__" in state:  # paused at human_review
        request = state["__interrupt__"][0].value
        state = graph.invoke(Command(resume=reviewer(request)), config)

    validation = ValidatedPlan(
        plan=state["plan"],
        attempts=state["attempts"],
        dropped=state.get("dropped", []),
        warnings=[p for p in state["problems"] if p.severity == "warning"],
    )
    return RunResult(
        run_dir=Path(state["run_dir"]),
        job=state["job"],
        analysis=state["analysis"],
        validation=validation,
        review_history=state.get("review_history", []),
        original=state.get("original"),
        tailored=state.get("tailored"),
    )
