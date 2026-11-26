"""Reading and writing the run history."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, inspect
from sqlmodel import Session, col, create_engine, select

from applybot.storage.models import SUMMARY_COLUMNS, Run, RunStatus, RunSummary
from applybot.storage.paths import HISTORY_DB, data_dir

MIGRATIONS = Path(__file__).parent / "migrations"
BASELINE = "0001"  # the schema Milestone 8 created, before migrations existed


def alembic_config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.attributes["connection"] = connection
    return config


def migrate(engine: Engine) -> None:
    """Bring the database's tables up to date with the code, creating them if needed."""
    with engine.begin() as connection:
        tables = inspect(connection).get_table_names()
        config = alembic_config(connection)
        if "run" in tables and "alembic_version" not in tables:
            # Made by Milestone 8's create_all: record it as the baseline, then upgrade.
            command.stamp(config, BASELINE)
        command.upgrade(config, "head")


class HistoryStore:
    """The run history in one SQLite file, migrated to the latest schema when opened."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # The API reads from request threads while the worker thread writes; each
        # connection is still used by one thread at a time (the pool hands them out).
        self.engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        migrate(self.engine)

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

    def mark_interrupted(self) -> list[int]:
        """Fail runs left queued or running by a previous process that was closed.

        Runs awaiting review are left alone: their state is saved and they can resume.
        """
        unfinished = (RunStatus.queued, RunStatus.running)
        with self.session() as session:
            runs = list(session.exec(select(Run).where(col(Run.status).in_(unfinished))))
        for run in runs:
            assert run.id is not None
            self.update_run(
                run.id,
                status=RunStatus.failed,
                step=None,
                error="ApplyBot was closed while this run was in progress.",
            )
        return [run.id for run in runs if run.id is not None]

    def close(self) -> None:
        self.engine.dispose()


def open_history() -> HistoryStore:
    """The history in the user's data folder."""
    return HistoryStore(data_dir() / HISTORY_DB)
