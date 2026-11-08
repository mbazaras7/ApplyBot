from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from applybot import cli
from applybot.cli import app
from applybot.pipeline import RunResult
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import Edit, EditError, EditPlan

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

    def run_tailor(cv_path: Path, llm: object, **kwargs: Any) -> RunResult:
        calls.append({"cv_path": cv_path, **kwargs})
        plan = EditPlan(edits=[Edit(block_id="exp.acme.1", new_text="x", rationale="Mirrors REST")])
        return RunResult(run_dir=tmp_path, job=JOB, plan=plan, original=None, tailored=None)

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
    assert "Initech" in result.output
    assert "exp.acme.1" in result.output
    assert "Mirrors REST" in result.output


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
