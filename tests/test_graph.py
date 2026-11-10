"""The tailoring graph, driven with a scripted LLM and reviewer."""

import json
import uuid
from pathlib import Path
from typing import Any

from fakes import (
    BAD_METRIC,
    GOOD_ACME,
    GOOD_GLOBEX,
    FakeEmbedder,
    ScriptedLLM,
    ScriptedReviewer,
    approve,
    edit_text,
    reject,
)

from applybot.agent.graph import RECURSION_LIMIT, build_graph
from applybot.pipeline import run_tailor
from applybot.review.diff import ReviewRequest

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"


def run_to_review(llm: ScriptedLLM, out_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run the graph until it pauses for review, without a reviewer."""
    graph = build_graph(llm, compile_pdf=False, embedder=FakeEmbedder())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
    initial = {
        "cv_path": str(SAMPLE),
        "job_source": None,
        "job_text": "Backend Engineer at Initech. Python.",
        "out_root": str(out_root),
        "started": "20261108-120000",
    }
    state = graph.invoke(initial, config)
    return graph, config, state


def request_of(state: dict[str, Any]) -> ReviewRequest:
    value = state["__interrupt__"][0].value
    assert isinstance(value, ReviewRequest)
    return value


# --- Up to the review pause ----------------------------------------------------------------


def test_pauses_for_review_before_compiling(tmp_path: Path) -> None:
    graph, config, state = run_to_review(ScriptedLLM([[GOOD_ACME]]), tmp_path)

    assert graph.get_state(config).next == ("human_review",)
    run_dir = tmp_path / "20261108-120000_initech-ltd"
    assert (run_dir / "original.tex").exists()
    assert (run_dir / "job.json").exists()
    assert not (run_dir / "tailored.tex").exists()  # nothing built before approval
    request = request_of(state)
    assert [e.block_id for e in request.entries] == ["exp.acme-corp.1"]
    assert request.can_approve


def test_analysis_runs_before_edits_and_reaches_review(tmp_path: Path) -> None:
    llm = ScriptedLLM([[GOOD_ACME]])
    _, _, state = run_to_review(llm, tmp_path)

    assert len(llm.tool_loops) == 1  # analyse_and_plan ran its tool loop
    assert "<analysis>" in llm.edit_prompts[0]  # ...and generation saw the result
    assert '"kind": "gap"' in llm.edit_prompts[0]
    analysis = request_of(state).analysis
    assert analysis is not None and analysis.fit_score == 64  # shown at review
    saved = json.loads((tmp_path / "20261108-120000_initech-ltd" / "analysis.json").read_text())
    assert saved["fit_score"] == 64


def test_validation_errors_are_retried(tmp_path: Path) -> None:
    llm = ScriptedLLM([[GOOD_ACME, BAD_METRIC], [GOOD_ACME]])
    _, _, state = run_to_review(llm, tmp_path)

    assert request_of(state).attempts == 2
    assert "<validation_errors>" in llm.edit_prompts[1]
    assert "exp.acme-corp.2: numbers must stay unchanged" in llm.edit_prompts[1]


def test_still_failing_edits_are_dropped(tmp_path: Path) -> None:
    attempt = [GOOD_ACME, BAD_METRIC]
    _, _, state = run_to_review(ScriptedLLM([attempt, attempt, attempt]), tmp_path)

    request = request_of(state)
    assert request.attempts == 3
    assert [e.block_id for e in request.entries] == ["exp.acme-corp.1"]
    assert [d.edit.block_id for d in request.dropped] == ["exp.acme-corp.2"]


# --- Through review ------------------------------------------------------------------------


def run(llm: ScriptedLLM, reviewer: ScriptedReviewer, out_root: Path) -> Any:
    return run_tailor(
        SAMPLE, llm, reviewer, job_text="Backend Engineer at Initech. Python.",
        out_root=out_root, compile_pdf=False, embedder=FakeEmbedder(),
    )  # fmt: skip


def test_approve_builds_the_tailored_cv(tmp_path: Path) -> None:
    result = run(ScriptedLLM([[GOOD_ACME]]), ScriptedReviewer([approve]), tmp_path)

    tailored = (result.run_dir / "tailored.tex").read_bytes().decode("utf-8")
    assert GOOD_ACME["new_text"] in tailored
    assert json.loads((result.run_dir / "review.json").read_text()) == [
        {"action": "approve", "comment": "", "edited_blocks": []}
    ]
    report = result.report_path.read_text(encoding="utf-8")
    assert report.startswith("# Tailoring report: Backend Engineer at Initech Ltd")
    assert "## Fit: 64/100" in report
    assert "### `exp.acme-corp.1`" in report
    assert "1. Approve" in report
    assert "PDF compilation was skipped" in report  # compile_pdf=False in these tests


def test_reject_regenerates_with_feedback(tmp_path: Path) -> None:
    llm = ScriptedLLM([[GOOD_ACME], [GOOD_GLOBEX]])
    reviewer = ScriptedReviewer([reject("Focus on testing"), approve])
    result = run(llm, reviewer, tmp_path)

    assert "<reviewer_feedback>\nFocus on testing\n</reviewer_feedback>" in llm.edit_prompts[1]
    assert reviewer.requests[1].attempts == 1  # the count restarts after a rejection
    assert [e.block_id for e in result.plan.edits] == ["exp.globex.1"]
    assert [r.action for r in result.review_history] == ["reject", "approve"]


def test_valid_hand_edit_is_rechecked_then_approved(tmp_path: Path) -> None:
    hand = "Built a \\textbf{Python} ingestion service for 2M events per day at 99.9\\% uptime."
    reviewer = ScriptedReviewer([edit_text("exp.acme-corp.1", hand), approve])
    result = run(ScriptedLLM([[GOOD_ACME]]), reviewer, tmp_path)

    second = reviewer.requests[1]
    assert second.entries[0].new == hand  # the review shows the hand-edited text
    assert second.can_approve
    assert result.plan.edits[0].new_text == hand
    assert result.review_history[0].edited_blocks == ["exp.acme-corp.1"]


def test_failing_hand_edit_goes_back_to_review(tmp_path: Path) -> None:
    bad = "Built a \\textbf{Python} ingestion service for 5M events per day at 99.9\\% uptime."
    good = "Built a \\textbf{Python} ingestion service for 2M events per day at 99.9\\% uptime."
    reviewer = ScriptedReviewer(
        [edit_text("exp.acme-corp.1", bad), edit_text("exp.acme-corp.1", good), approve]
    )
    result = run(ScriptedLLM([[GOOD_ACME]]), reviewer, tmp_path)

    shown = reviewer.requests[1]
    assert not shown.can_approve  # approve isn't offered while the hand edit fails
    assert "numbers must stay unchanged" in shown.entries[0].problems[0].message
    assert reviewer.requests[2].can_approve  # fixed on the second try
    assert result.plan.edits[0].new_text == good
