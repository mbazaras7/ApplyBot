"""The database schema comes from Alembic migrations, and they must match the models."""

import sqlite3
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlmodel import SQLModel, create_engine

from applybot.storage.history import HistoryStore, alembic_config
from applybot.storage.models import RunStatus

# The run table exactly as Milestone 8's create_all made it (copied from a real database).
M8_SCHEMA = """
CREATE TABLE run (
    id INTEGER NOT NULL, thread_id VARCHAR NOT NULL, created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL, status VARCHAR(15) NOT NULL, error VARCHAR,
    company VARCHAR, title VARCHAR, fit_score INTEGER, run_dir VARCHAR NOT NULL,
    base_cv_tex VARCHAR NOT NULL, job_json VARCHAR NOT NULL, analysis_json VARCHAR NOT NULL,
    edits_json VARCHAR, validation_json VARCHAR, review_json VARCHAR, tailored_tex VARCHAR,
    tailored_pdf BLOB, report_md VARCHAR, original_pages INTEGER, tailored_pages INTEGER,
    PRIMARY KEY (id)
);
CREATE UNIQUE INDEX ix_run_thread_id ON run (thread_id);
CREATE INDEX ix_run_created_at ON run (created_at);
INSERT INTO run VALUES (
    1, 'thread-1', '2026-11-18 20:25:18.000000', '2026-11-18 20:26:00.000000', 'completed',
    NULL, 'Vhi', 'Graduate', 80, 'runs/x', 'cv', '{}', '{}', NULL, NULL, NULL, NULL,
    X'25504446', '# Report', 1, 1
);
"""


def version(db: Path) -> str:
    [(revision,)] = sqlite3.connect(db).execute("SELECT version_num FROM alembic_version")
    return str(revision)


def test_new_database_is_created_at_the_latest_version(tmp_path: Path) -> None:
    HistoryStore(tmp_path / "new.db").close()
    assert version(tmp_path / "new.db") == "0002"


def test_milestone_8_database_is_upgraded_and_keeps_its_runs(tmp_path: Path) -> None:
    db = tmp_path / "m8.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(M8_SCHEMA)

    store = HistoryStore(db)
    run = store.get_run(1)
    assert run is not None
    assert (run.company, run.status, run.tailored_pdf) == ("Vhi", RunStatus.completed, b"%PDF")
    assert (run.facts_md, run.custom_prompt, run.step) == ("", "", None)  # new columns
    store.update_run(1, job_json=None)  # job columns are optional now
    store.close()
    assert version(db) == "0002"


def test_migrations_match_the_models(tmp_path: Path) -> None:
    """If a model changes without a migration, this fails: write the migration."""
    engine = create_engine(f"sqlite:///{tmp_path / 'db.db'}")
    HistoryStore(tmp_path / "db.db").close()
    with engine.connect() as conn:
        differences = compare_metadata(MigrationContext.configure(conn), SQLModel.metadata)
    assert differences == []


def test_downgrade_and_upgrade_again(tmp_path: Path) -> None:
    db = tmp_path / "db.db"
    HistoryStore(db).close()
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        command.downgrade(alembic_config(conn), "0001")
    assert version(db) == "0001"
    HistoryStore(db).close()  # opening upgrades it again
    assert version(db) == "0002"
