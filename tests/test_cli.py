from pathlib import Path
from typing import Any

import pytest
from fakes import ANALYSIS
from rich.prompt import Prompt
from rich.text import Text
from typer.testing import CliRunner

from applybot import cli
from applybot.agent.state import ReviewDecision, ReviewRound
from applybot.cli import app
from applybot.pipeline import RunResult
from applybot.review.diff import ReviewRequest
from applybot.review.editor import EditFileError
from applybot.tools.analyse import JobAnalysis
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import Edit, EditError, EditPlan
from applybot.tools.validate import ValidatedPlan

runner = CliRunner()

JOB = JobPosting(
    title="Backend Engineer",
    company="Initech",
    seniority=None,
    required_skills=[],
    nice_to_have=[],
    qualifications=[],
    soft_skills=[],
    responsibilities=[],
    keywords=[],
    raw_text="Backend engineer",
)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Replace the LLM and the pipeline, recording what the CLI passed in."""
    calls: list[dict[str, Any]] = []

    def run_tailor(cv_path: Path, llm: object, reviewer: object, **kwargs: Any) -> RunResult:
        calls.append({"cv_path": cv_path, "reviewer": reviewer, **kwargs})
        plan = EditPlan(edits=[Edit(block_id="exp.acme.1", new_text="x", rationale="Mirrors REST")])
        validation = ValidatedPlan(plan=plan, attempts=1, dropped=[], warnings=[])
        return RunResult(
            run_dir=tmp_path,
            job=JOB,
            analysis=JobAnalysis.model_validate(ANALYSIS),
            validation=validation,
            review_history=[ReviewRound(action="reject"), ReviewRound(action="approve")],
            original=None,
            tailored=None,
        )

    monkeypatch.setattr(cli, "AnthropicClient", lambda: object())
    monkeypatch.setattr(cli, "run_tailor", run_tailor)
    return calls


@pytest.fixture
def cv_file(tmp_path: Path) -> Path:
    cv = tmp_path / "main.tex"
    cv.write_text("\\documentclass{article}\n", encoding="utf-8")
    return cv


def test_help_lists_tailor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tailor" in result.output


def test_tailor_runs_pipeline(cv_file: Path, fake_run: list[dict[str, Any]]) -> None:
    result = runner.invoke(app, ["tailor", str(cv_file), "--job-text", "Backend engineer"])

    assert result.exit_code == 0, result.output
    assert fake_run[0]["job_text"] == "Backend engineer"
    assert fake_run[0]["job_source"] is None
    assert fake_run[0]["reviewer"] is cli.interactive_review
    assert "Initech: 1 edits approved after 2 review(s)." in result.output
    assert "Fit 64/100, 1 gap(s)." in result.output


# --- The interactive reviewer --------------------------------------------------------------

REQUEST = ReviewRequest(
    plan=EditPlan(edits=[Edit(block_id="exp.acme.1", new_text="new", rationale="r")]),
    entries=[],
    plan_problems=[],
    dropped=[],
    attempts=1,
    can_approve=True,
)


def answer(monkeypatch: pytest.MonkeyPatch, *replies: str) -> list[list[str] | None]:
    """Script Prompt.ask; returns the choices offered at each prompt."""
    offered: list[list[str] | None] = []
    queue = list(replies)

    def ask(prompt: str, choices: list[str] | None = None, **kwargs: Any) -> str:
        offered.append(choices)
        return queue.pop(0)

    monkeypatch.setattr(Prompt, "ask", ask)
    return offered


def test_review_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    offered = answer(monkeypatch, "a")
    assert cli.interactive_review(REQUEST) == ReviewDecision(action="approve")
    assert offered[0] == ["a", "e", "r"]


def test_review_prompt_shows_the_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Found in the live demo: Rich read "[a]" as a formatting tag and printed "pprove".
    prompts: list[str] = []

    def ask(prompt: str, **kwargs: Any) -> str:
        prompts.append(prompt)
        return "a"

    monkeypatch.setattr(Prompt, "ask", ask)
    cli.interactive_review(REQUEST)
    assert Text.from_markup(prompts[0]).plain == "[a]pprove, [e]dit, [r]eject"


def test_review_reject_with_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, "r", "Be more specific")
    assert cli.interactive_review(REQUEST) == ReviewDecision(
        action="reject", comment="Be more specific"
    )


def test_review_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    edited = EditPlan(edits=[Edit(block_id="exp.acme.1", new_text="mine", rationale="r")])
    monkeypatch.setattr(cli, "edit_by_hand", lambda plan: edited)
    answer(monkeypatch, "e")
    assert cli.interactive_review(REQUEST) == ReviewDecision(action="edit", plan=edited)


def test_review_hides_approve_when_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    offered = answer(monkeypatch, "r", "")
    cli.interactive_review(REQUEST.model_copy(update={"can_approve": False}))
    assert offered[0] == ["e", "r"]


def test_review_bad_edit_file_asks_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(plan: EditPlan) -> EditPlan:
        raise EditFileError("[exp.acme.1] has no text")

    monkeypatch.setattr(cli, "edit_by_hand", broken)
    answer(monkeypatch, "e", "a")
    assert cli.interactive_review(REQUEST).action == "approve"


def test_edit_by_hand_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_editor(path: Path) -> None:
        path.write_text(path.read_text(encoding="utf-8").replace("new", "edited"), "utf-8")

    monkeypatch.setattr(cli, "open_in_editor", fake_editor)
    assert cli.edit_by_hand(REQUEST.plan).edits[0].new_text == "edited"


def test_tailor_passes_facts(cv_file: Path, tmp_path: Path, fake_run: list[dict[str, Any]]) -> None:
    facts = tmp_path / "facts.md"
    facts.write_text("Also used Rust.", encoding="utf-8")

    result = runner.invoke(app, ["tailor", str(cv_file), "--job-text", "x", "--facts", str(facts)])

    assert result.exit_code == 0, result.output
    assert fake_run[0]["facts_path"] == facts


def test_tailor_passes_job_file(
    cv_file: Path, tmp_path: Path, fake_run: list[dict[str, Any]]
) -> None:
    job = tmp_path / "job.txt"
    job.write_text("Backend engineer", encoding="utf-8")

    result = runner.invoke(app, ["tailor", str(cv_file), "--job-file", str(job)])

    assert result.exit_code == 0, result.output
    assert fake_run[0]["job_source"] == str(job)


def test_tailor_reports_expected_errors(cv_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_run(*args: Any, **kwargs: Any) -> RunResult:
        raise EditError("Edit plan rejected:\nexp.nope.1: unknown block ID")

    monkeypatch.setattr(cli, "AnthropicClient", lambda: object())
    monkeypatch.setattr(cli, "run_tailor", failing_run)

    result = runner.invoke(app, ["tailor", str(cv_file), "--job-text", "x"])

    assert result.exit_code == 1
    assert "unknown block ID" in result.output


def test_tailor_rejects_missing_cv(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tailor", str(tmp_path / "nope.tex"), "--job-text", "x"])
    assert result.exit_code != 0


def test_tailor_rejects_two_job_sources(cv_file: Path) -> None:
    result = runner.invoke(
        app,
        ["tailor", str(cv_file), "--job-url", "https://example.com/job", "--job-text", "x"],
    )
    assert result.exit_code == 2


def test_tailor_rejects_no_job_source(cv_file: Path) -> None:
    result = runner.invoke(app, ["tailor", str(cv_file)])
    assert result.exit_code == 2
