"""The diff view and the hand-editing file format."""

from pathlib import Path

import pytest
from fakes import ANALYSIS
from rich.console import Console

from applybot.latex.parser import parse
from applybot.review.diff import ReviewRequest, build_diff, render_review, word_diff
from applybot.review.editor import (
    HAND_EDIT_RATIONALE,
    EditFileError,
    editor_command,
    edits_to_text,
    text_to_edits,
)
from applybot.tools.analyse import JobAnalysis
from applybot.tools.generate_edits import Edit, EditPlan
from applybot.tools.validate import Problem

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
PLAN = EditPlan(
    edits=[
        Edit(block_id="exp.acme-corp.1", new_text="Built a fast service.", rationale="speed"),
        Edit(block_id="skills.languages", new_text="SQL, Python, Go, TypeScript", rationale="SQL"),
    ]
)


# --- Editor file ---------------------------------------------------------------------------


def test_round_trip() -> None:
    assert text_to_edits(edits_to_text(PLAN), PLAN) == PLAN


def test_changed_text_keeps_rationale() -> None:
    text = edits_to_text(PLAN).replace("Built a fast service.", "Built a quick service.")
    edits = text_to_edits(text, PLAN).edits
    assert edits[0].new_text == "Built a quick service."
    assert edits[0].rationale == "speed"


def test_deleting_a_section_drops_the_edit() -> None:
    text = edits_to_text(PLAN).replace("[skills.languages]\nSQL, Python, Go, TypeScript\n", "")
    assert [e.block_id for e in text_to_edits(text, PLAN).edits] == ["exp.acme-corp.1"]


def test_added_section_gets_default_rationale() -> None:
    text = edits_to_text(PLAN) + "\n[exp.globex.1]\nWrote Go tests.\n"
    added = text_to_edits(text, PLAN).edits[-1]
    assert (added.block_id, added.rationale) == ("exp.globex.1", HAND_EDIT_RATIONALE)


def test_comments_and_windows_line_endings_are_ignored() -> None:
    text = "# note\r\n[exp.acme-corp.1]\r\n# another note\r\nNew text.\r\n"
    assert text_to_edits(text, PLAN).edits[0].new_text == "New text."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("stray text\n[exp.acme-corp.1]\nx\n", "Line 1: text before the first"),
        ("[exp.acme-corp.1]\n\n", "has no text"),
    ],
)
def test_bad_files(text: str, message: str) -> None:
    with pytest.raises(EditFileError, match=message):
        text_to_edits(text, PLAN)


def test_editor_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISUAL", "code --wait")
    assert editor_command() == ["code", "--wait"]
    monkeypatch.delenv("VISUAL")
    monkeypatch.setenv("EDITOR", "nano")
    assert editor_command() == ["nano"]


# --- Diff ----------------------------------------------------------------------------------


def test_build_diff_attaches_problems() -> None:
    doc = parse(SAMPLE.read_bytes().decode("utf-8"))
    problem = Problem(block_id="exp.acme-corp.1", rule="numbers", message="numbers changed")
    entries = build_diff(doc, PLAN, [problem])

    assert entries[0].where == "Experience > Acme Corp"
    assert entries[0].old.startswith("Built a \\textbf{Python} ingestion service")
    assert entries[0].problems == [problem]
    assert entries[1].problems == []


def test_word_diff_marks_changes() -> None:
    text = word_diff("worked across teams", "collaborated across teams")
    styles = {(text.plain[span.start : span.end], str(span.style)) for span in text.spans}
    assert ("worked", "red strike") in styles
    assert ("collaborated", "green") in styles


def test_review_starts_with_fit_and_gaps() -> None:
    request = ReviewRequest(
        plan=PLAN, entries=[], plan_problems=[], dropped=[], attempts=1, can_approve=True,
        analysis=JobAnalysis.model_validate(ANALYSIS),
    )  # fmt: skip
    console = Console(record=True, width=120)
    render_review(console, request)
    output = console.export_text()

    assert "Fit 64/100" in output
    assert "Strong Python, no Kubernetes." in output
    assert "gap: Kubernetes: No container orchestration in the CV." in output
    assert output.index("Fit 64/100") < output.index("0 edits")  # fit comes first


def test_render_review_shows_everything() -> None:
    doc = parse(SAMPLE.read_bytes().decode("utf-8"))
    page_problem = Problem(block_id=None, rule="pages", message="2 pages but the original has 1")
    request = ReviewRequest(
        plan=PLAN,
        entries=build_diff(doc, PLAN, []),
        plan_problems=[page_problem],
        dropped=[],
        attempts=2,
        can_approve=False,
    )
    console = Console(record=True, width=120)
    render_review(console, request)
    output = console.export_text()

    assert "exp.acme-corp.1" in output
    assert "speed" in output  # rationale
    assert "error: 2 pages but the original has 1" in output
    assert "2 edits, generated in 2 attempt(s)." in output
