from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own data folder, so no test touches the real run history."""
    folder = tmp_path / "data"
    monkeypatch.setenv("APPLYBOT_DATA_DIR", str(folder))
    return folder
