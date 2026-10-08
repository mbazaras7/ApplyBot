from pathlib import Path

import pytest

from applybot.latex.parser import (
    LatexParseError,
    parse,
    read_command_name,
    read_group,
    slugify,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"


def load(path: Path) -> str:
    # Decode bytes ourselves so Windows newline translation can't change the text.
    return path.read_bytes().decode("utf-8")


def wrap(body: str, section: str = "Experience") -> str:
    return f"\\documentclass{{article}}\n\\begin{{document}}\n\\section{{{section}}}\n{body}"


# --- read_group ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "content"),
    [
        ("{plain}", "plain"),
        ("{a \\textbf{bold} b}", "a \\textbf{bold} b"),
        ("{\\href{https://x.y}{\\underline{link}}}", "\\href{https://x.y}{\\underline{link}}"),
        ("{escaped \\{ and \\} braces}", "escaped \\{ and \\} braces"),
        ("{99.9\\% and R\\&D}", "99.9\\% and R\\&D"),
        ("{line break \\\\}", "line break \\\\"),
        ("{a % comment with }\n b}", "a % comment with }\n b"),
    ],
)
def test_read_group(text: str, content: str) -> None:
    assert read_group(text + " trailing", 0) == (content, len(text))


def test_read_group_backslash_then_brace() -> None:
    # `\\{` is a line break followed by a real opening brace, so depth must go up.
    text = "{a \\\\{b} c}"
    assert read_group(text, 0) == ("a \\\\{b} c", len(text))


def test_read_group_unbalanced() -> None:
    with pytest.raises(LatexParseError, match="line 2"):
        read_group("x\n{never \\textbf{closed}", 2)


def test_read_command_name() -> None:
    assert read_command_name("\\resumeItemListStart{", 0) == ("resumeItemListStart", 20)
    assert read_command_name("\\&x", 0) == ("&", 2)


def test_slugify() -> None:
    assert slugify("Web \\& Backend:") == "web-backend"
    assert slugify("Dublin City University") == "dublin-city-university"


# --- parse -------------------------------------------------------------------------------


def test_preamble_is_everything_before_begin_document() -> None:
    tex = load(SAMPLE)
    doc = parse(tex)
    assert doc.preamble == tex[: tex.index("\\begin{document}")]
    assert "\\newcommand{\\resumeItem}" in doc.preamble


def test_sample_block_ids_and_flags() -> None:
    doc = parse(load(SAMPLE))
    summary = [(b.id, b.kind, b.editable) for b in doc.blocks]
    assert summary == [
        ("summary.1", "summary", True),
        ("edu.springfield-university.1", "item", False),
        ("exp.acme-corp.1", "item", True),
        ("exp.acme-corp.2", "item", True),
        ("exp.globex.1", "item", True),
        ("proj.taskflow.1", "item", True),
        ("skills.languages", "skills_line", True),
        ("skills.tools-platforms", "skills_line", True),
    ]


def test_block_text_and_context() -> None:
    doc = parse(load(SAMPLE))
    bullet = doc.block("exp.acme-corp.1")
    assert bullet.text == (
        "Built a \\textbf{Python} ingestion service processing 2M events per day "
        "with 99.9\\% uptime."
    )
    assert bullet.section == "Experience"
    assert bullet.parent_heading == "Acme Corp"
    assert doc.block("proj.taskflow.1").parent_heading == "TaskFlow"


def test_skills_lines_keep_only_the_list_editable() -> None:
    doc = parse(load(SAMPLE))
    line = doc.block("skills.tools-platforms")
    assert line.text == "Docker, Redis, PostgreSQL, GitHub Actions"
    assert line.parent_heading == "Tools \\& Platforms:"
    assert doc.block("skills.languages").text == "Python, Go, TypeScript, SQL"


def test_commented_out_item_is_ignored() -> None:
    doc = parse(wrap("% \\resumeItem{old}\n\\resumeItem{new}\n"))
    assert [b.text for b in doc.blocks] == ["new"]


def test_list_start_macro_is_not_an_item() -> None:
    doc = parse(wrap("\\resumeItemListStart\n\\resumeItem{only one}\n\\resumeItemListEnd\n"))
    assert [b.text for b in doc.blocks] == ["only one"]


def test_arguments_may_span_lines() -> None:
    body = "\\resumeSubheading\n  {Initech}{City}\n  {Dev}{2020}\n\\resumeItem{x}\n"
    assert parse(wrap(body)).blocks[0].id == "exp.initech.1"


def test_unknown_section_is_not_editable() -> None:
    doc = parse(wrap("\\resumeItem{Volunteer}\n", section="Volunteering"))
    assert doc.blocks[0].id == "volunteering.1"
    assert doc.blocks[0].editable is False


def test_duplicate_ids_are_rejected() -> None:
    body = "\\section{Skills}\n\\resumeItem{\\textbf{A:} x \\\\ \\textbf{A:} y}\n"
    with pytest.raises(LatexParseError, match="Duplicate block ID 'skills.a'"):
        parse(wrap(body))


def test_missing_begin_document() -> None:
    with pytest.raises(LatexParseError):
        parse("\\resumeItem{x}")
