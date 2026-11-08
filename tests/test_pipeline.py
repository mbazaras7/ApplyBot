import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fakes import GOOD_ACME, JOB_DETAILS, FakeEmbedder, ScriptedLLM, ScriptedReviewer, approve

from applybot.pipeline import run_tailor

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
NOW = datetime(2026, 11, 7, 15, 30, 0)
HAS_COMPILER = bool(shutil.which("tectonic") or shutil.which("latexmk"))
REQUIRE_LATEX = os.environ.get("REQUIRE_LATEX") == "1"


def test_run_writes_artifacts(tmp_path: Path) -> None:
    result = run_tailor(
        SAMPLE,
        ScriptedLLM([[GOOD_ACME]]),
        ScriptedReviewer([approve]),
        job_text="Backend Engineer at Initech",
        out_root=tmp_path,
        now=NOW,
        compile_pdf=False,
        embedder=FakeEmbedder(),
    )

    run_dir = tmp_path / "20261107-153000_initech-ltd"
    assert result.run_dir == run_dir
    original = SAMPLE.read_bytes()
    assert (run_dir / "original.tex").read_bytes() == original
    tailored = (run_dir / "tailored.tex").read_bytes().decode("utf-8")
    assert GOOD_ACME["new_text"] in tailored
    assert tailored.startswith(original.decode("utf-8").split("\\begin{document}")[0])
    assert json.loads((run_dir / "job.json").read_text())["company"] == "Initech Ltd"
    assert json.loads((run_dir / "edits.json").read_text()) == {"edits": [GOOD_ACME]}
    validation = json.loads((run_dir / "validation.json").read_text())
    assert validation["attempts"] == 1
    assert validation["dropped"] == []
    assert json.loads((run_dir / "review.json").read_text())[0]["action"] == "approve"
    assert result.original is None and result.tailored is None


def test_facts_reach_validation(tmp_path: Path) -> None:
    facts = tmp_path / "facts.md"
    facts.write_text("I also know Rust.", encoding="utf-8")
    rust_edit = {**GOOD_ACME, "new_text": GOOD_ACME["new_text"].replace("Redis-backed", "Rust")}
    result = run_tailor(
        SAMPLE,
        ScriptedLLM([[rust_edit]]),
        ScriptedReviewer([approve]),
        job_text="A job",
        facts_path=facts,
        out_root=tmp_path,
        now=NOW,
        compile_pdf=False,
        embedder=FakeEmbedder(),
    )
    assert result.validation.attempts == 1  # "Rust" is allowed because facts.md has it


def test_unknown_company_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(JOB_DETAILS, "company", None)
    result = run_tailor(
        SAMPLE, ScriptedLLM([[GOOD_ACME]]), ScriptedReviewer([approve]), job_text="A job",
        out_root=tmp_path, now=NOW, compile_pdf=False, embedder=FakeEmbedder(),
    )  # fmt: skip
    assert result.run_dir.name == "20261107-153000_unknown"


def test_needs_exactly_one_job_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_tailor(SAMPLE, ScriptedLLM([]), ScriptedReviewer([]), out_root=tmp_path)


@pytest.mark.skipif(not HAS_COMPILER and not REQUIRE_LATEX, reason="no LaTeX compiler installed")
def test_run_compiles_both_versions(tmp_path: Path) -> None:
    result = run_tailor(
        SAMPLE, ScriptedLLM([[GOOD_ACME]]), ScriptedReviewer([approve]), job_text="A job",
        out_root=tmp_path, now=NOW, embedder=FakeEmbedder(),
    )  # fmt: skip

    assert result.original is not None and result.original.success
    assert result.tailored is not None
    assert result.tailored.success, result.tailored.log
    assert result.original.page_count == result.tailored.page_count == 1
    assert (result.run_dir / "tailored.pdf").exists()
