"""report.md: content and Markdown safety."""

from pathlib import Path

from fakes import ANALYSIS

from applybot.agent.state import ReviewRound
from applybot.latex.compiler import CompileResult
from applybot.latex.parser import parse
from applybot.tools.analyse import JobAnalysis
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import Edit, EditPlan
from applybot.tools.report import build_report, introduced_words
from applybot.tools.validate import DroppedEdit, Problem, ValidatedPlan

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
DOC = parse(SAMPLE.read_bytes().decode("utf-8"))
JOB = JobPosting(
    title="Backend Engineer", company="Initech", seniority="junior", required_skills=["Python"],
    nice_to_have=[], qualifications=[], soft_skills=[], responsibilities=[], keywords=[],
    raw_text="Backend Engineer at Initech.",
)  # fmt: skip
NEW_ACME = (
    "Built a \\textbf{Python} ingestion service in a fast-paced team processing 2M events "
    "per day with 99.9\\% uptime."
)
EDIT = Edit(block_id="exp.acme-corp.1", new_text=NEW_ACME, rationale="Targets Python services")
VALIDATION = ValidatedPlan(
    plan=EditPlan(edits=[EDIT]),
    attempts=2,
    dropped=[
        DroppedEdit(
            edit=Edit(block_id="exp.acme-corp.2", new_text="x", rationale="r"),
            reasons=["numbers must stay unchanged (added 50; removed 35)"],
        )
    ],
    warnings=[
        Problem(
            block_id="exp.acme-corp.1", rule="noop", message="barely changes", severity="warning"
        )
    ],
)
OK = CompileResult(success=True, log="", page_count=1)


def report(**changes: object) -> str:
    arguments: dict[str, object] = {
        "job": JOB,
        "analysis": JobAnalysis.model_validate(ANALYSIS),
        "doc": DOC,
        "validation": VALIDATION,
        "review_history": [ReviewRound(action="reject", comment="More Python"),
                           ReviewRound(action="edit", edited_blocks=["exp.acme-corp.1"]),
                           ReviewRound(action="approve")],
        "original": OK,
        "tailored": OK,
    } | changes  # fmt: skip
    return build_report(**arguments)  # type: ignore[arg-type]


def test_sections_in_order() -> None:
    text = report()
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == [
        "# Tailoring report: Backend Engineer at Initech",
        "## Fit: 64/100",
        "## Changes (1)",
        "### `exp.acme-corp.1`",
        "## Validation",
        "## Review",
        "## Output",
    ]


def test_fit_and_findings() -> None:
    text = report()
    assert "Strong Python, no Kubernetes." in text
    assert "| Python | Strength | `exp.acme-corp.1` | Built a Python ingestion service. |" in text
    assert "| Kubernetes | Gap | none | No container orchestration in the CV. |" in text
    assert "**Gaps (1):** Kubernetes" in text


def test_change_shows_before_after_and_new_words() -> None:
    text = report()
    assert "*Why:* Targets Python services" in text
    assert "- **Before:** Built a Python ingestion service processing 2M events" in text
    assert "- **After:** Built a Python ingestion service in a fast-paced team" in text
    # Embellishments stand out ("fast-paced" is split at the hyphen, like everywhere else).
    assert "- **Words introduced:** fast, paced, team" in text
    assert "\\textbf" not in text  # LaTeX stripped for readability


def test_validation_and_review() -> None:
    text = report()
    assert "- Generated in 2 attempt(s)." in text
    assert "  - `exp.acme-corp.2`: numbers must stay unchanged (added 50; removed 35)" in text
    assert "  - `exp.acme-corp.1`: barely changes" in text
    assert '1. Reject ("More Python")' in text
    assert "2. Edit: edited `exp.acme-corp.1`" in text
    assert "3. Approve" in text


def test_output_success_and_failure() -> None:
    assert "Compiled `tailored.pdf` successfully. Pages: 1 -> 1." in report()
    failed = CompileResult(success=False, log="! Undefined control sequence.\nl.3")
    text = report(tailored=failed)
    assert "**The tailored CV failed to compile:** Undefined control sequence." in text
    assert "Automatic repair isn't built yet" in text
    assert "compilation was skipped" in report(tailored=None)


def test_no_edits() -> None:
    validation = VALIDATION.model_copy(update={"plan": EditPlan(edits=[]), "warnings": []})
    assert "No edits were applied" in report(validation=validation)


def test_table_cells_escape_pipes() -> None:
    analysis = dict(ANALYSIS)
    analysis["findings"] = [{**ANALYSIS["findings"][0], "requirement": "Python | Go"}]
    assert "| Python \\| Go |" in report(analysis=JobAnalysis.model_validate(analysis))


def test_introduced_words_ignore_other_forms() -> None:
    assert introduced_words("Tested and deployed services", "Testing and deploying services") == []
    assert introduced_words("Wrote tests", "Wrote integration tests") == ["integration"]
