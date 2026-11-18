"""Reading and writing the run history."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, SQLModel, col, create_engine, select

from applybot.storage.models import SUMMARY_COLUMNS, Run, RunSummary
from applybot.storage.paths import HISTORY_DB, data_dir


class HistoryStore:
    """The run history in one SQLite file. Tables are created on first use."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def session(self) -> Session:
        # expire_on_commit=False keeps returned rows readable after the session closes.
        return Session(self.engine, expire_on_commit=False)

    def create_run(self, run: Run) -> int:
        with self.session() as session:
            session.add(run)
            session.commit()
            assert run.id is not None  # set by the database on commit
            return run.id

    def update_run(self, run_id: int, **fields: Any) -> None:
        """Set some columns of a run, and its updated_at."""
        with self.session() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise KeyError(f"No run {run_id}.")
            for name, value in fields.items():
                setattr(run, name, value)
            run.updated_at = datetime.now(UTC)
            session.add(run)
            session.commit()

    def get_run(self, run_id: int) -> Run | None:
        with self.session() as session:
            return session.get(Run, run_id)

    def list_runs(self, limit: int | None = None) -> list[RunSummary]:
        """Newest first. Selects only the summary columns, so listing runs never reads
        their stored CVs, reports or PDFs."""
        query = (
            select(*SUMMARY_COLUMNS)
            .order_by(col(Run.created_at).desc(), col(Run.id).desc())
            .limit(limit)
        )
        with self.session() as session:
            return [RunSummary.model_validate(row._mapping) for row in session.exec(query)]

    def close(self) -> None:
        self.engine.dispose()


def open_history() -> HistoryStore:
    """The history in the user's data folder."""
    return HistoryStore(data_dir() / HISTORY_DB)
