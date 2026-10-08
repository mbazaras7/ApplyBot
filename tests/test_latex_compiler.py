import shutil
from pathlib import Path

import pytest

from applybot.latex.compiler import compile_tex

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"

pytestmark = pytest.mark.skipif(
    not (shutil.which("tectonic") or shutil.which("latexmk")),
    reason="no LaTeX compiler installed",
)


def test_compile_sample(tmp_path: Path) -> None:
    tex = tmp_path / "cv.tex"
    shutil.copy(SAMPLE, tex)

    result = compile_tex(tex)

    assert result.success, result.log
    assert result.pdf_path == tmp_path / "cv.pdf"
    assert result.page_count == 1


def test_compile_failure_returns_log(tmp_path: Path) -> None:
    tex = tmp_path / "broken.tex"
    tex.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\undefinedcommand\n\\end{document}\n",
        encoding="utf-8",
    )

    result = compile_tex(tex)

    assert not result.success
    assert result.page_count is None
    assert "undefinedcommand" in result.log
