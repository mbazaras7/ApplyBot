"""report.md: a human-readable record of one tailoring run."""

from applybot.agent.state import ReviewRound
from applybot.latex.compiler import CompileResult
from applybot.latex.models import CVDocument
from applybot.tools.analyse import JobAnalysis
from applybot.tools.cv_index import plain
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import Edit
from applybot.tools.validate import FILLER_WORDS, ValidatedPlan, first_latex_error, stem, words

KIND_LABELS = {"strength": "Strength", "rewrite_target": "Rewrite target", "gap": "Gap"}


def cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def introduced_words(old: str, new: str) -> list[str]:
    """Words an edit brought in that the original block didn't have (in any form).

    Listed in the report so additions like "fast-paced" stand out at a glance.
    """
    old_stems = {stem(word) for word in words(old)}
    found = [w for w in words(new) if stem(w) not in old_stems and len(w) > 3]
    return sorted({w for w in found if w not in FILLER_WORDS})


def fit_section(analysis: JobAnalysis) -> list[str]:
    lines = [f"## Fit: {analysis.fit_score}/100", "", analysis.fit_reasoning, ""]
    lines += ["| Requirement | Finding | Evidence | Note |", "|---|---|---|---|"]
    for finding in analysis.findings:
        evidence = ", ".join(f"`{e}`" for e in finding.evidence) or "none"
        lines.append(
            f"| {cell(finding.requirement)} | {KIND_LABELS[finding.kind]} | {evidence} "
            f"| {cell(finding.note)} |"
        )
    gaps = analysis.gaps
    lines += ["", f"**Gaps ({len(gaps)}):** " + ("; ".join(g.requirement for g in gaps) or "none")]
    return lines


def change_section(doc: CVDocument, edits: list[Edit]) -> list[str]:
    lines = [f"## Changes ({len(edits)})"]
    if not edits:
        return [*lines, "", "No edits were applied: the tailored CV is the original."]
    blocks = {block.id: block for block in doc.blocks}
    for edit in edits:
        old = blocks[edit.block_id].text if edit.block_id in blocks else ""
        new_words = introduced_words(old, edit.new_text)
        lines += [
            "",
            f"### `{edit.block_id}`",
            "",
            f"*Why:* {edit.rationale}",
            "",
            f"- **Before:** {plain(old)}",
            f"- **After:** {plain(edit.new_text)}",
            f"- **Words introduced:** {', '.join(new_words) if new_words else 'none'}",
        ]
    return lines


def validation_section(validation: ValidatedPlan) -> list[str]:
    lines = ["## Validation", "", f"- Generated in {validation.attempts} attempt(s)."]
    if validation.dropped:
        lines.append(f"- Dropped {len(validation.dropped)} edit(s) that kept failing:")
        for dropped in validation.dropped:
            lines.append(f"  - `{dropped.edit.block_id}`: {'; '.join(dropped.reasons)}")
    else:
        lines.append("- No edits dropped.")
    if validation.warnings:
        lines.append(f"- {len(validation.warnings)} warning(s):")
        lines += [f"  - `{w.block_id or 'whole plan'}`: {w.message}" for w in validation.warnings]
    else:
        lines.append("- No warnings.")
    return lines


def review_section(history: list[ReviewRound]) -> list[str]:
    lines = ["## Review", ""]
    for number, round_ in enumerate(history, start=1):
        detail = ""
        if round_.edited_blocks:
            detail = f": edited {', '.join(f'`{b}`' for b in round_.edited_blocks)}"
        if round_.comment:
            detail += f' ("{round_.comment}")'
        lines.append(f"{number}. {round_.action.capitalize()}{detail}")
    return lines


def output_section(original: CompileResult | None, tailored: CompileResult | None) -> list[str]:
    lines = ["## Output", ""]
    if tailored is None:
        return [*lines, "PDF compilation was skipped for this run."]
    if not tailored.success:
        error = first_latex_error(tailored.log)
        return [
            *lines,
            f"**The tailored CV failed to compile:** {error}",
            "",
            "See `tailored.log`. Automatic repair isn't built yet, so fix the edit by hand "
            "and run again.",
        ]
    pages = (
        f"{original.page_count} -> {tailored.page_count}" if original else str(tailored.page_count)
    )
    return [*lines, f"Compiled `tailored.pdf` successfully. Pages: {pages}."]


def build_report(
    *,
    job: JobPosting,
    analysis: JobAnalysis,
    doc: CVDocument,
    validation: ValidatedPlan,
    review_history: list[ReviewRound],
    original: CompileResult | None,
    tailored: CompileResult | None,
) -> str:
    title = f"{job.title or 'Unknown role'} at {job.company or 'unknown company'}"
    sections = [
        [f"# Tailoring report: {title}"],
        fit_section(analysis),
        change_section(doc, validation.plan.edits),
        validation_section(validation),
        review_section(review_history),
        output_section(original, tailored),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
