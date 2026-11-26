"""RunManager: runs moving through the background worker, review over calls, restarts."""

from pathlib import Path

import pytest
from fakes import (
    GOOD_ACME,
    GOOD_GLOBEX,
    FakeEmbedder,
    InlineExecutor,
    ManualExecutor,
    ScriptedLLM,
)

from applybot.agent.state import ReviewDecision
from applybot.llm.client import LLMClient, LLMConfigError
from applybot.service.runs import RunInputs, RunManager, RunNotFoundError, RunStateError
from applybot.storage.models import RunStatus
from applybot.tools.generate_edits import Edit, EditPlan

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
INPUTS = RunInputs(cv_tex=SAMPLE.read_text(encoding="utf-8"), job_text="Backend at Initech")


def make_manager(
    tmp_path: Path, llm: LLMClient, executor: InlineExecutor | ManualExecutor | None = None
) -> RunManager:
    return RunManager(
        tmp_path / "data",
        lambda: llm,
        out_root=tmp_path / "runs",
        compile_pdf=False,
        embedder=FakeEmbedder(),
        executor=executor or InlineExecutor(),
    )


def test_run_pauses_for_review_then_completes(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME]]))
    run_id = manager.submit(INPUTS)

    run = manager.get(run_id)
    assert run.status == RunStatus.awaiting_review
    assert run.step == "Waiting for review"
    assert (run.company, run.fit_score) == ("Initech Ltd", 64)
    review = manager.review(run_id)
    assert [e.block_id for e in review.plan.edits] == [GOOD_ACME["block_id"]]
    assert review.can_approve

    manager.decide(run_id, ReviewDecision(action="approve"))
    run = manager.get(run_id)
    assert run.status == RunStatus.completed
    assert run.step is None
    assert run.tailored_tex is not None and GOOD_ACME["new_text"] in run.tailored_tex
    assert run.report_md is not None
    manager.close()


def test_reject_generates_a_new_plan(tmp_path: Path) -> None:
    llm = ScriptedLLM([[GOOD_ACME], [GOOD_GLOBEX]])
    manager = make_manager(tmp_path, llm)
    run_id = manager.submit(INPUTS)

    manager.decide(run_id, ReviewDecision(action="reject", comment="Focus on Globex."))
    assert manager.get(run_id).status == RunStatus.awaiting_review
    assert [e.block_id for e in manager.review(run_id).plan.edits] == [GOOD_GLOBEX["block_id"]]
    assert "Focus on Globex." in llm.edit_prompts[1]
    manager.close()


def test_hand_edit_with_broken_latex_blocks_approval(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME]]))
    run_id = manager.submit(INPUTS)
    broken = {**GOOD_ACME, "new_text": GOOD_ACME["new_text"] + "}"}
    manager.decide(
        run_id, ReviewDecision(action="edit", plan=EditPlan(edits=[Edit.model_validate(broken)]))
    )

    assert manager.get(run_id).status == RunStatus.awaiting_review
    assert not manager.review(run_id).can_approve
    with pytest.raises(RunStateError, match="hand edits have errors"):
        manager.decide(run_id, ReviewDecision(action="approve"))
    manager.close()


def test_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, ScriptedLLM([]))
    run_id = manager.submit(INPUTS.model_copy(update={"job_text": None, "job_source": "nope.pdf"}))
    run = manager.get(run_id)
    assert run.status == RunStatus.failed
    assert run.error is not None and "No such file" in run.error
    assert run.step is None
    manager.close()


def test_missing_configuration_fails_the_run(tmp_path: Path) -> None:
    def no_model() -> LLMClient:
        raise LLMConfigError("Set ANTHROPIC_MODEL in your environment or .env file.")

    manager = RunManager(
        tmp_path / "data", no_model, out_root=tmp_path / "runs", executor=InlineExecutor()
    )
    run = manager.get(manager.submit(INPUTS))
    assert run.status == RunStatus.failed
    assert run.error is not None and "ANTHROPIC_MODEL" in run.error
    manager.close()


def test_calls_in_the_wrong_state(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME]]))
    run_id = manager.submit(INPUTS)
    manager.decide(run_id, ReviewDecision(action="approve"))

    with pytest.raises(RunStateError, match="isn't waiting for review"):
        manager.review(run_id)
    with pytest.raises(RunStateError):
        manager.decide(run_id, ReviewDecision(action="approve"))
    with pytest.raises(RunNotFoundError):
        manager.get(999)
    manager.close()


def test_runs_wait_their_turn(tmp_path: Path) -> None:
    executor = ManualExecutor()
    manager = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME], [GOOD_GLOBEX]]), executor)
    first, second = manager.submit(INPUTS), manager.submit(INPUTS)
    assert manager.get(first).status == manager.get(second).status == RunStatus.queued

    executor.run_next()
    assert manager.get(first).status == RunStatus.awaiting_review
    assert manager.get(second).status == RunStatus.queued
    manager.close()


def test_review_survives_a_restart(tmp_path: Path) -> None:
    before = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME]]))
    run_id = before.submit(INPUTS)
    before.close()

    # A new process: new manager, same data folder. Its LLM has nothing left to say,
    # so the run must continue from the saved state rather than start again.
    after = make_manager(tmp_path, ScriptedLLM([]))
    assert after.get(run_id).status == RunStatus.awaiting_review
    assert after.review(run_id).plan.edits[0].block_id == GOOD_ACME["block_id"]
    after.decide(run_id, ReviewDecision(action="approve"))
    assert after.get(run_id).status == RunStatus.completed
    after.close()


def test_unfinished_work_is_failed_on_restart(tmp_path: Path) -> None:
    before = make_manager(tmp_path, ScriptedLLM([[GOOD_ACME]]), ManualExecutor())
    queued = before.submit(INPUTS)  # never run: the "process" closes first
    before.store.close()

    after = make_manager(tmp_path, ScriptedLLM([]))
    run = after.get(queued)
    assert run.status == RunStatus.failed
    assert run.error == "ApplyBot was closed while this run was in progress."
    after.close()
