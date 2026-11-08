import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from applybot.pipeline import run_tailor

T = TypeVar("T", bound=BaseModel)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
NOW = datetime(2026, 11, 7, 15, 30, 0)
HAS_COMPILER = bool(shutil.which("tectonic") or shutil.which("latexmk"))
REQUIRE_LATEX = os.environ.get("REQUIRE_LATEX") == "1"

JOB_DETAILS = {
    "title": "Backend Engineer",
    "company": "Initech Ltd",
    "seniority": "junior",
    "required_skills": ["Python"],
    "nice_to_have": [],
    "qualifications": [],
    "soft_skills": [],
    "responsibilities": ["Build REST APIs"],
    "keywords": ["Python", "REST"],
}
NEW_BULLET = (
    "Built a \\textbf{Python} Redis-backed ingestion service handling 2M events per day "
    "at 99.9\\% uptime."
)
EDIT_PLAN = {
    "edits": [{"block_id": "exp.acme-corp.1", "new_text": NEW_BULLET, "rationale": "REST"}]
}


class FakeLLM:
    """Answers each schema with a canned reply, keyed by the schema's class name."""

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        replies: dict[str, Any] = {"JobDetails": JOB_DETAILS, "EditPlan": EDIT_PLAN}
        return schema.model_validate(replies[schema.__name__])


def test_run_writes_artifacts(tmp_path: Path) -> None:
    result = run_tailor(
        SAMPLE,
        FakeLLM(),
        job_text="Backend Engineer at Initech",
        out_root=tmp_path,
        now=NOW,
        compile_pdf=False,
    )

    run_dir = tmp_path / "20261107-153000_initech-ltd"
    assert result.run_dir == run_dir
    original = SAMPLE.read_bytes()
    assert (run_dir / "original.tex").read_bytes() == original
    tailored = (run_dir / "tailored.tex").read_bytes().decode("utf-8")
    assert NEW_BULLET in tailored
    assert tailored.startswith(original.decode("utf-8").split("\\begin{document}")[0])
    assert json.loads((run_dir / "job.json").read_text())["company"] == "Initech Ltd"
    assert json.loads((run_dir / "edits.json").read_text()) == EDIT_PLAN
    validation = json.loads((run_dir / "validation.json").read_text())
    assert validation["attempts"] == 1
    assert validation["dropped"] == []
    assert result.original is None and result.tailored is None


def test_facts_reach_validation(tmp_path: Path) -> None:
    facts = tmp_path / "facts.md"
    facts.write_text("I also know Rust.", encoding="utf-8")
    result = run_tailor(
        SAMPLE,
        FakeLLM(),
        job_text="A job",
        facts_path=facts,
        out_root=tmp_path,
        now=NOW,
        compile_pdf=False,
    )
    assert result.validation.attempts == 1


def test_unknown_company_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(JOB_DETAILS, "company", None)
    result = run_tailor(
        SAMPLE, FakeLLM(), job_text="A job", out_root=tmp_path, now=NOW, compile_pdf=False
    )
    assert result.run_dir.name == "20261107-153000_unknown"


def test_needs_exactly_one_job_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_tailor(SAMPLE, FakeLLM(), out_root=tmp_path)


@pytest.mark.skipif(not HAS_COMPILER and not REQUIRE_LATEX, reason="no LaTeX compiler installed")
def test_run_compiles_both_versions(tmp_path: Path) -> None:
    result = run_tailor(SAMPLE, FakeLLM(), job_text="A job", out_root=tmp_path, now=NOW)

    assert result.original is not None and result.original.success
    assert result.tailored is not None
    assert result.tailored.success, result.tailored.log
    assert result.original.page_count == result.tailored.page_count == 1
    assert (result.run_dir / "tailored.pdf").exists()
