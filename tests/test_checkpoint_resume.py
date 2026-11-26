"""A run paused at review survives the program exiting: the state is saved in SQLite."""

from pathlib import Path

from fakes import GOOD_ACME, FakeEmbedder, ScriptedLLM
from langgraph.types import Command

from applybot.agent.graph import build_graph, sqlite_checkpointer
from applybot.agent.state import ReviewDecision
from applybot.service.runs import RunInputs, advance, graph_config, submit_run
from applybot.storage.history import HistoryStore
from applybot.storage.models import RunStatus

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"


def test_paused_run_resumes_in_a_new_graph(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "applybot.db")
    checkpoints = tmp_path / "checkpoints.db"
    inputs = RunInputs(cv_tex=SAMPLE.read_text(encoding="utf-8"), job_text="Backend at Initech")
    run_id, initial = submit_run(store, inputs, tmp_path / "runs")
    thread_id = initial["thread_id"]

    # First "session": run until the review pause, then close everything.
    with sqlite_checkpointer(checkpoints) as saver:
        graph = build_graph(
            ScriptedLLM([[GOOD_ACME]]), False, FakeEmbedder(), store=store, checkpointer=saver
        )
        progress = advance(graph, store, run_id, thread_id, initial)
    assert progress.review is not None
    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.awaiting_review
    assert run.step == "Waiting for review"

    # Second "session": a brand-new graph and connection, found by the stored thread ID.
    # Its LLM has no plans left, so it would fail if anything before the pause re-ran.
    with sqlite_checkpointer(checkpoints) as saver:
        graph = build_graph(ScriptedLLM([]), False, FakeEmbedder(), store=store, checkpointer=saver)
        saved = graph.get_state(graph_config(run.thread_id))
        assert saved.next == ("human_review",)
        assert saved.values["review_request"].plan.edits[0].block_id == GOOD_ACME["block_id"]
        resume: Command[str] = Command(resume=ReviewDecision(action="approve"))
        progress = advance(graph, store, run_id, run.thread_id, resume)

    assert progress.review is None
    finished = store.get_run(run_id)
    store.close()
    assert finished is not None
    assert finished.status == RunStatus.completed
    assert finished.step is None
    assert finished.tailored_tex is not None
    assert GOOD_ACME["new_text"] in finished.tailored_tex
