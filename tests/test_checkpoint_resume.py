"""A run paused at review survives the program exiting: the state is saved in SQLite."""

import uuid
from pathlib import Path

from fakes import GOOD_ACME, FakeEmbedder, ScriptedLLM
from langgraph.types import Command

from applybot.agent.graph import RECURSION_LIMIT, build_graph, sqlite_checkpointer
from applybot.agent.state import ReviewDecision
from applybot.storage.history import HistoryStore
from applybot.storage.models import RunStatus

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"


def test_paused_run_resumes_in_a_new_graph(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "applybot.db")
    checkpoints = tmp_path / "checkpoints.db"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    # First "session": run until the review pause, then close everything.
    with sqlite_checkpointer(checkpoints) as saver:
        graph = build_graph(
            ScriptedLLM([[GOOD_ACME]]), False, FakeEmbedder(), store=store, checkpointer=saver
        )
        state = graph.invoke(
            {
                "thread_id": thread_id,
                "cv_path": str(SAMPLE),
                "job_source": None,
                "job_text": "Backend Engineer at Initech",
                "out_root": str(tmp_path / "runs"),
                "started": "20261118-090000",
            },
            config,
        )
    assert "__interrupt__" in state
    [run] = store.list_runs()
    assert run.status == RunStatus.awaiting_review
    assert run.thread_id == thread_id

    # Second "session": a brand-new graph and connection, found by the stored thread ID.
    # Its LLM has no plans left, so it would fail if anything before the pause re-ran.
    with sqlite_checkpointer(checkpoints) as saver:
        graph = build_graph(ScriptedLLM([]), False, FakeEmbedder(), store=store, checkpointer=saver)
        saved = graph.get_state(config)
        assert saved.next == ("human_review",)
        assert saved.values["review_request"].plan.edits[0].block_id == GOOD_ACME["block_id"]
        state = graph.invoke(Command(resume=ReviewDecision(action="approve")), config)

    assert "__interrupt__" not in state
    finished = store.get_run(run.id)
    store.close()
    assert finished is not None
    assert finished.status == RunStatus.completed
    assert finished.tailored_tex is not None
    assert GOOD_ACME["new_text"] in finished.tailored_tex
