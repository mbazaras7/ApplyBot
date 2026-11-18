"""Where ApplyBot keeps its databases."""

import os
from pathlib import Path

import platformdirs

HISTORY_DB = "applybot.db"  # run history (our tables)
CHECKPOINT_DB = "checkpoints.db"  # LangGraph's saved graph state (its own tables)


def data_dir() -> Path:
    r"""The per-user data folder, e.g. %LOCALAPPDATA%\ApplyBot on Windows.

    APPLYBOT_DATA_DIR overrides it: tests point it at a temporary folder, and the Docker
    image points it at the mounted /work folder so history outlives the container.
    """
    override = os.environ.get("APPLYBOT_DATA_DIR")
    path = Path(override) if override else Path(platformdirs.user_data_dir("ApplyBot", False))
    path.mkdir(parents=True, exist_ok=True)
    return path
