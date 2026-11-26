"""The state every node of the graph reads from and writes to."""

from typing import Literal, TypedDict

from pydantic import BaseModel

from applybot.latex.compiler import CompileResult
from applybot.latex.models import CVDocument
from applybot.review.diff import ReviewRequest
from applybot.tools.analyse import JobAnalysis
from applybot.tools.cv_index import CVIndex
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import EditPlan
from applybot.tools.validate import DroppedEdit, Problem


class ReviewDecision(BaseModel):
    """The reviewer's answer, sent back into the graph with `Command(resume=...)`."""

    action: Literal["approve", "edit", "reject"]
    plan: EditPlan | None = None  # the hand-edited plan, for "edit"
    comment: str = ""  # what should change, for "reject"


class ReviewRound(BaseModel):
    """One pass through the review step, kept for review.json."""

    action: Literal["approve", "edit", "reject"]
    comment: str = ""
    edited_blocks: list[str] = []


class TailorState(TypedDict, total=False):
    """`total=False`: every key is optional, because each node fills in its own part.

    A node returns only the keys it changed; LangGraph merges them into the state.
    """

    # Inputs, all as text: the graph never reads input files itself
    cv_tex: str
    facts: str
    custom_prompt: str  # the user's preferences, added below the generation rules
    job_source: str | None  # URL or file path, fetched by fetch_job
    job_text: str | None  # or the job's text, given directly
    out_root: str
    started: str  # timestamp for the run folder name
    thread_id: str  # the checkpointer's key for this run, stored in the history row
    run_id: int | None  # the run's row in the history database, if one is kept

    # Filled in along the way
    original_tex: str
    doc: CVDocument
    index: CVIndex
    job: JobPosting
    analysis: JobAnalysis
    run_dir: str
    original: CompileResult | None

    # Generate / validate loop
    plan: EditPlan
    problems: list[Problem]
    attempts: int  # generations since the last reset (first try + retries)
    dropped: list[DroppedEdit]
    retry_errors: str  # validation errors for the next generation, or ""
    reviewer_comment: str | None  # set when the reviewer rejected the plan

    # Review
    review_request: ReviewRequest
    review_errors: list[Problem]  # errors in hand edits; approve is blocked while non-empty
    decision: ReviewDecision
    review_history: list[ReviewRound]

    # Output
    tailored: CompileResult | None
