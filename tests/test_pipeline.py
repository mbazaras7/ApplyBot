import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fakes import GOOD_ACME, JOB_DETAILS, FakeEmbedder, ScriptedLLM, ScriptedReviewer, approve

from applybot.agent.state import ReviewDecision
from applybot.pipeline import run_tailor
from applybot.storage.history import HistoryStore
from applybot.storage.models import RunStatus
from applybot.storage.paths import data_dir

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

    # The tailored PDF is also stored in the history database.
    store = HistoryStore(data_dir() / "applybot.db")
    assert result.run_id is not None
    run = store.get_run(result.run_id)
    store.close()
    assert run is not None and run.status == RunStatus.completed
    assert run.tailored_pdf == (result.run_dir / "tailored.pdf").read_bytes()
    assert run.original_pages == run.tailored_pages == 1


def test_run_is_recorded_in_history(tmp_path: Path, data_dir: Path) -> None:
    result = run_tailor(
        SAMPLE, ScriptedLLM([[GOOD_ACME]]), ScriptedReviewer([approve]),
        job_text="Backend Engineer at Initech", out_root=tmp_path, now=NOW,
        compile_pdf=False, embedder=FakeEmbedder(),
    )  # fmt: skip
    assert result.run_id is not None
    assert (data_dir / "applybot.db").exists() and (data_dir / "checkpoints.db").exists()

    store = HistoryStore(data_dir / "applybot.db")
    run = store.get_run(result.run_id)
    store.close()
    assert run is not None
    assert run.status == RunStatus.completed
    assert run.created_at.astimezone().replace(tzinfo=None) == NOW
    assert (run.company, run.title, run.fit_score) == ("Initech Ltd", "Backend Engineer", 64)
    assert run.run_dir == str(result.run_dir)
    # The database holds the same files as the run folder.
    assert run.base_cv_tex.encode("utf-8") == SAMPLE.read_bytes()
    for column, name in [
        (run.job_json, "job.json"), (run.edits_json, "edits.json"),
        (run.validation_json, "validation.json"), (run.review_json, "review.json"),
        (run.report_md, "report.md"),
    ]:  # fmt: skip
        assert column == (result.run_dir / name).read_text(encoding="utf-8")
    assert run.tailored_tex is not None
    assert run.tailored_tex.encode("utf-8") == (result.run_dir / "tailored.tex").read_bytes()
    assert run.tailored_pdf is None  # compiling was off


def test_crash_after_the_run_starts_is_recorded(tmp_path: Path, data_dir: Path) -> None:
    def crash(request: object) -> ReviewDecision:
        raise RuntimeError("reviewer went away")

    with pytest.raises(RuntimeError):
        run_tailor(
            SAMPLE, ScriptedLLM([[GOOD_ACME]]), crash, job_text="A job",
            out_root=tmp_path, now=NOW, compile_pdf=False, embedder=FakeEmbedder(),
        )  # fmt: skip
    store = HistoryStore(data_dir / "applybot.db")
    [summary] = store.list_runs()
    run = store.get_run(summary.id)
    store.close()
    assert run is not None
    assert run.status == RunStatus.failed
    assert run.error == "reviewer went away"


def test_crash_before_the_run_starts_leaves_no_row(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_tailor(
            tmp_path / "missing.tex", ScriptedLLM([]), ScriptedReviewer([]), job_text="A job",
            out_root=tmp_path, compile_pdf=False, embedder=FakeEmbedder(),
        )  # fmt: skip
    store = HistoryStore(data_dir / "applybot.db")
    assert store.list_runs() == []
    store.close()
