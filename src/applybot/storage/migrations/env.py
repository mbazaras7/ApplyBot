"""Alembic's entry point.

ApplyBot runs migrations from code (`storage/history.py`), passing an open connection.
Run by hand (`uv run alembic current`, `uv run alembic revision --autogenerate -m ...`),
it connects to the history in the data folder instead.
"""

from alembic import context
from sqlalchemy import Connection
from sqlmodel import SQLModel, create_engine

import applybot.storage.models  # noqa: F401  (registers the tables on SQLModel.metadata)
from applybot.storage.paths import HISTORY_DB, data_dir


def run(connection: Connection) -> None:
    # render_as_batch: SQLite can't alter most column properties in place, so Alembic
    # copies the table into a new one with the changes ("batch mode") instead.
    context.configure(
        connection=connection, target_metadata=SQLModel.metadata, render_as_batch=True
    )
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    run(connection)
else:
    with create_engine(f"sqlite:///{data_dir() / HISTORY_DB}").connect() as conn:
        run(conn)
