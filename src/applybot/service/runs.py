"""Submitting runs and moving them forward, shared by the CLI and the HTTP API.

A run's life:
    submit_run   -> history row (queued) + the graph's first input
    advance      -> runs the graph until it pauses for review or finishes
    (review)     -> advance again with Command(resume=decision)

`RunManager` does this in the background for the API: one worker thread runs the
queued work in order, and every step's state is saved in checkpoints.db, so a run
waiting for review can be picked up again after a restart.
"""

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel

from applybot.agent.graph import (
    RECURSION_LIMIT,
    STEP_LABELS,
    build_graph,
    sqlite_checkpointer,
    start_timestamp,
)
from applybot.agent.state import ReviewDecision, TailorState
from applybot.llm.client import LLMClient
from applybot.review.diff import ReviewRequest
from applybot.storage.history import HistoryStore
from applybot.storage.models import Run, RunStatus
from applybot.storage.paths import CHECKPOINT_DB, HISTORY_DB
from applybot.tools.cv_index import Embedder, FastEmbedder

log = logging.getLogger(__name__)


class RunInputs(BaseModel):
    """Everything a run starts from, as text. Exactly one of job_source / job_text."""

    cv_tex: str
    facts: str = ""
    custom_prompt: str = ""
    job_source: str | None = None  # a URL (or, from the CLI, a file path or "-")
    job_text: str | None = None


class RunNotFoundError(LookupError):
    pass


class RunStateError(RuntimeError):
    """The run isn't in a state that allows this (e.g. a decision while it's running)."""


def graph_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def submit_run(
    store: HistoryStore, inputs: RunInputs, out_root: Path, now: datetime | None = None
) -> tuple[int, TailorState]:
    """Create the run's history row (queued) and the graph's first input."""
    if (inputs.job_source is None) == (inputs.job_text is None):
        raise ValueError("Pass exactly one of job_source or job_text.")
    created = (now or datetime.now()).astimezone()  # local time, with its timezone
    thread_id = str(uuid.uuid4())  # the checkpointer keeps one saved state per thread
    run_id = store.create_run(
        Run(
            thread_id=thread_id,
            created_at=created,
            updated_at=datetime.now(UTC),
            base_cv_tex=inputs.cv_tex,
            facts_md=inputs.facts,
            custom_prompt=inputs.custom_prompt,
        )
    )
    initial: TailorState = {
        "run_id": run_id,
        "thread_id": thread_id,
        "cv_tex": inputs.cv_tex,
        "facts": inputs.facts,
        "custom_prompt": inputs.custom_prompt,
        "job_source": inputs.job_source,
        "job_text": inputs.job_text,
        "out_root": str(out_root),
        "started": start_timestamp(created.replace(tzinfo=None)),
    }
    return run_id, initial


@dataclass
class Progress:
    """Where the graph stopped: waiting for review (`review` set) or finished."""

    values: dict[str, Any]
    review: ReviewRequest | None


def advance(
    graph: Any, store: HistoryStore, run_id: int, thread_id: str, graph_input: Any
) -> Progress:
    """Run the graph from `graph_input` (the first input, or a Command resuming a review)
    until it pauses or ends, recording each step it starts as the run's progress."""
    config = graph_config(thread_id)
    store.update_run(run_id, status=RunStatus.running)
    # "tasks" streams an event when each node starts (it has an "input") and when it ends.
    for event in graph.stream(graph_input, config, stream_mode="tasks"):
        if "input" in event:
            store.update_run(run_id, step=STEP_LABELS.get(event["name"], event["name"]))
    snapshot = graph.get_state(config)
    if snapshot.interrupts:  # paused at human_review
        return Progress(snapshot.values, snapshot.interrupts[0].value)
    store.update_run(run_id, step=None)
    return Progress(snapshot.values, None)


def record_failure(store: HistoryStore, run_id: int, error: BaseException) -> None:
    """Mark a run failed, unless it was waiting for review (then it can still resume)."""
    run = store.get_run(run_id)
    if run is not None and run.status in (RunStatus.queued, RunStatus.running):
        message = str(error) or type(error).__name__
        store.update_run(run_id, status=RunStatus.failed, step=None, error=message)


class RunManager:
    """Runs tailoring jobs in the background for the API.

    One worker thread (by default) runs the work in submission order: on a small laptop
    two runs at once would compete for memory, and the embedding model is shared.
    """

    def __init__(
        self,
        folder: Path,
        llm_factory: Callable[[], LLMClient],
        *,
        out_root: Path = Path("runs"),
        compile_pdf: bool = True,
        embedder: Embedder | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.store = HistoryStore(folder / HISTORY_DB)
        self.checkpoints = folder / CHECKPOINT_DB
        self.llm_factory = llm_factory
        self.out_root = out_root
        self.compile_pdf = compile_pdf
        self.embedder = embedder or FastEmbedder()  # loads its model on first use
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        # Work queued or running in a previous process is gone: record it as failed.
        self.store.mark_interrupted()

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.store.close()

    def get(self, run_id: int) -> Run:
        run = self.store.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"No run {run_id}.")
        return run

    def submit(self, inputs: RunInputs, now: datetime | None = None) -> int:
        run_id, initial = submit_run(self.store, inputs, self.out_root, now)
        self.executor.submit(self.work, run_id, initial["thread_id"], initial)
        return run_id

    def review(self, run_id: int) -> ReviewRequest:
        """The review the run is waiting on, read from its saved state."""
        run = self.awaiting_review(run_id)
        with sqlite_checkpointer(self.checkpoints) as saver:
            snapshot = self.graph(saver).get_state(graph_config(run.thread_id))
        if not snapshot.interrupts:
            raise RunStateError(f"Run {run_id} has no review saved.")
        request: ReviewRequest = snapshot.interrupts[0].value
        return request

    def decide(self, run_id: int, decision: ReviewDecision) -> None:
        """Answer the review; the run continues in the background."""
        run = self.awaiting_review(run_id)
        if decision.action == "approve" and not self.review(run_id).can_approve:
            raise RunStateError("Can't approve while the hand edits have errors.")
        self.store.update_run(run_id, status=RunStatus.queued)
        self.executor.submit(self.work, run_id, run.thread_id, Command(resume=decision))

    def awaiting_review(self, run_id: int) -> Run:
        run = self.get(run_id)
        if run.status != RunStatus.awaiting_review:
            raise RunStateError(f"Run {run_id} isn't waiting for review (it's {run.status}).")
        return run

    def graph(self, saver: Any) -> Any:
        return build_graph(
            self.llm_factory(),
            self.compile_pdf,
            self.embedder,
            store=self.store,
            checkpointer=saver,
        )

    def work(self, run_id: int, thread_id: str, graph_input: Any) -> None:
        """One piece of background work: advance the run to its next pause or the end."""
        try:
            with sqlite_checkpointer(self.checkpoints) as saver:
                advance(self.graph(saver), self.store, run_id, thread_id, graph_input)
        except Exception as error:  # a background thread has nobody to raise to
            log.exception("Run %s failed", run_id)
            record_failure(self.store, run_id, error)
