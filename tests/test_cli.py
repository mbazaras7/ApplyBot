from pathlib import Path

import pytest
from typer.testing import CliRunner

from applybot.cli import app

runner = CliRunner()


@pytest.fixture
def cv_file(tmp_path: Path) -> Path:
    cv = tmp_path / "main.tex"
    cv.write_text("\\documentclass{article}\n", encoding="utf-8")
    return cv


def test_help_lists_tailor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tailor" in result.output


def test_tailor_echoes_args(cv_file: Path) -> None:
    result = runner.invoke(app, ["tailor", str(cv_file), "--job-text", "Backend engineer"])
    assert result.exit_code == 0
    assert "Backend engineer" in result.output


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
