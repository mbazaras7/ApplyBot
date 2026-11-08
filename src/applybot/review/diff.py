"""What the reviewer sees: each proposed edit as a word-level diff, with its problems."""

import difflib
import re

from pydantic import BaseModel
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from applybot.latex.models import CVDocument
from applybot.tools.generate_edits import EditPlan
from applybot.tools.validate import DroppedEdit, Problem


class DiffEntry(BaseModel):
    block_id: str
    where: str
    old: str
    new: str
    rationale: str
    problems: list[Problem]


class ReviewRequest(BaseModel):
    """Everything the review step shows, sent out of the graph by `interrupt()`."""

    plan: EditPlan
    entries: list[DiffEntry]
    plan_problems: list[Problem]  # problems not tied to one block (pages, compile)
    dropped: list[DroppedEdit]
    attempts: int
    can_approve: bool  # False while hand edits still fail validation


def build_diff(doc: CVDocument, plan: EditPlan, problems: list[Problem]) -> list[DiffEntry]:
    blocks = {block.id: block for block in doc.blocks}
    entries = []
    for edit in plan.edits:
        block = blocks.get(edit.block_id)
        where = block.section if block else "unknown block"
        if block and block.parent_heading:
            where += f" > {block.parent_heading}"
        entries.append(
            DiffEntry(
                block_id=edit.block_id,
                where=where,
                old=block.text if block else "",
                new=edit.new_text,
                rationale=edit.rationale,
                problems=[p for p in problems if p.block_id == edit.block_id],
            )
        )
    return entries


def word_diff(old: str, new: str) -> Text:
    """Removed words in red (struck through), added words in green."""
    old_words, new_words = re.findall(r"\S+|\s+", old), re.findall(r"\S+|\s+", new)
    text = Text()
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            text.append("".join(old_words[i1:i2]))
            continue
        if op in ("delete", "replace"):
            text.append("".join(old_words[i1:i2]), style="red strike")
        if op in ("insert", "replace"):
            text.append("".join(new_words[j1:j2]), style="green")
    return text


def problem_line(problem: Problem) -> Text:
    style = "red" if problem.severity == "error" else "yellow"
    return Text(f"{problem.severity}: {problem.message}", style=style)


def render_review(console: Console, request: ReviewRequest) -> None:
    for entry in request.entries:
        body: list[Text] = [word_diff(entry.old, entry.new), Text(entry.rationale, style="dim")]
        body += [problem_line(p) for p in entry.problems]
        console.print(
            Panel(Group(*body), title=f"{entry.block_id}  ({entry.where})", title_align="left")
        )
    for problem in request.plan_problems:
        console.print(problem_line(problem))
    for dropped in request.dropped:
        console.print(f"[red]Dropped[/red] {dropped.edit.block_id}: {'; '.join(dropped.reasons)}")
    console.print(f"{len(request.entries)} edits, generated in {request.attempts} attempt(s).")
