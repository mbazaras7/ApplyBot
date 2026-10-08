from pathlib import Path

import pytest

from applybot.latex.parser import parse
from applybot.latex.renderer import render

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_cv.tex"
PERSONAL_CV = Path(__file__).parent.parent / "examples" / "main.tex"


def load(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def test_round_trip_sample() -> None:
    tex = load(SAMPLE)
    assert render(parse(tex)) == tex


def test_round_trip_crlf() -> None:
    tex = load(SAMPLE).replace("\r\n", "\n").replace("\n", "\r\n")
    assert render(parse(tex)) == tex


@pytest.mark.skipif(not PERSONAL_CV.exists(), reason="examples/main.tex is kept local")
def test_round_trip_personal_cv() -> None:
    tex = load(PERSONAL_CV)
    assert render(parse(tex)) == tex


def test_edit_changes_only_that_block() -> None:
    tex = load(SAMPLE)
    doc = parse(tex)
    old = doc.block("exp.globex.1").text
    new = "Wrote Go integration tests for billing, raising coverage from 40\\% to 75\\%."

    out = render(doc, {"exp.globex.1": new})

    assert out == tex.replace(old, new)
    assert out.startswith(doc.preamble)


def test_skills_edit_keeps_label() -> None:
    doc = parse(load(SAMPLE))
    out = render(doc, {"skills.languages": "SQL, TypeScript, Go, Python"})
    assert "\\textbf{Languages:} SQL, TypeScript, Go, Python \\\\" in out


def test_unknown_block_id() -> None:
    with pytest.raises(KeyError):
        render(parse(load(SAMPLE)), {"exp.nope.1": "x"})
