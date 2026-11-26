from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import platformdirs
import pytest
from sqlalchemy import event

from applybot.storage import paths
from applybot.storage.history import HistoryStore, open_history
from applybot.storage.models import Run, RunStatus

T0 = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)


def make_run(thread_id: str, created_at: datetime = T0, **fields: Any) -> Run:
    return Run(
        thread_id=thread_id,
        created_at=created_at,
        updated_at=created_at,
        run_dir=f"runs/{thread_id}",
        base_cv_tex=r"\documentclass{article}",
        job_json="{}",
        analysis_json="{}",
        **fields,
    )


@pytest.fixture
def store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history.db")


def test_data_dir_override_is_created(data_dir: Path) -> None:
    assert not data_dir.exists()
    assert paths.data_dir() == data_dir
    assert data_dir.is_dir()


def test_data_dir_default_is_the_user_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APPLYBOT_DATA_DIR")
    calls: list[tuple[object, ...]] = []

    def user_data_dir(*args: object) -> str:
        calls.append(args)
        return str(tmp_path / "user-data")

    monkeypatch.setattr(platformdirs, "user_data_dir", user_data_dir)
    assert paths.data_dir() == tmp_path / "user-data"
    assert calls == [("ApplyBot", False)]


def test_create_and_get(store: HistoryStore) -> None:
    run_id = store.create_run(make_run("t1", company="Initech", fit_score=72))
    run = store.get_run(run_id)
    assert run is not None
    assert (run.thread_id, run.company, run.fit_score) == ("t1", "Initech", 72)
    assert run.status == RunStatus.queued
    assert run.created_at == T0  # comes back timezone-aware, in UTC
    assert store.get_run(run_id + 1) is None


def test_update_sets_fields_and_updated_at(store: HistoryStore) -> None:
    run_id = store.create_run(make_run("t1"))
    store.update_run(run_id, status=RunStatus.awaiting_review)
    run = store.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.awaiting_review
    assert run.updated_at > T0


def test_update_missing_run(store: HistoryStore) -> None:
    with pytest.raises(KeyError):
        store.update_run(99, status=RunStatus.failed)


def test_pdf_bytes_round_trip(store: HistoryStore) -> None:
    pdf = b"%PDF-1.7\n\x00\xff binary"
    run_id = store.create_run(make_run("t1"))
    store.update_run(run_id, tailored_pdf=pdf, report_md="# Report")
    run = store.get_run(run_id)
    assert run is not None
    assert run.tailored_pdf == pdf
    assert run.report_md == "# Report"


def test_list_is_newest_first(store: HistoryStore) -> None:
    for hours, thread in [(0, "old"), (2, "new"), (1, "middle")]:
        store.create_run(make_run(thread, T0 + timedelta(hours=hours)))
    assert [r.thread_id for r in store.list_runs()] == ["new", "middle", "old"]
    assert [r.thread_id for r in store.list_runs(limit=1)] == ["new"]


def test_thread_ids_are_unique(store: HistoryStore) -> None:
    store.create_run(make_run("t1"))
    with pytest.raises(Exception, match="UNIQUE"):
        store.create_run(make_run("t1"))


def test_history_survives_reopening(tmp_path: Path) -> None:
    first = HistoryStore(tmp_path / "history.db")
    first.create_run(make_run("t1"))
    first.close()
    assert [r.thread_id for r in HistoryStore(tmp_path / "history.db").list_runs()] == ["t1"]


def test_open_history_uses_the_data_folder(data_dir: Path) -> None:
    open_history().close()
    assert (data_dir / paths.HISTORY_DB).exists()


def test_list_never_reads_the_stored_files(store: HistoryStore) -> None:
    run_id = store.create_run(make_run("t1", company="Initech", fit_score=72))
    store.update_run(run_id, tailored_pdf=b"%PDF" * 1000, report_md="# Report")
    statements: list[str] = []

    @event.listens_for(store.engine, "before_cursor_execute")
    def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    [summary] = store.list_runs()
    assert (summary.id, summary.company, summary.fit_score) == (run_id, "Initech", 72)
    assert summary.status == RunStatus.queued
    assert summary.created_at == T0
    [select_sql] = statements
    for column in ("tailored_pdf", "report_md", "base_cv_tex", "tailored_tex"):
        assert column not in select_sql
