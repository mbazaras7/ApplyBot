"""Deterministic checks on an EditPlan, and the generate -> validate -> retry loop.

Every check is plain code: the same plan always gets the same verdict, each rule is
unit-tested, and the error messages are exact enough to send back to the model.
"""

import difflib
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from applybot.latex.compiler import CompileResult, compile_tex
from applybot.latex.models import Block, CVDocument
from applybot.latex.parser import LatexParseError, parse
from applybot.latex.renderer import render
from applybot.llm.client import LLMClient
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import Edit, EditPlan, generate_edits

MAX_RETRIES = 2
LENGTH_TOLERANCE = 0.20
# An edit is "small" if it changes at most this many words, or keeps this share of them.
NOOP_MAX_CHANGED_WORDS = 3
NOOP_KEPT_SHARE = 0.85
SUFFIXES = ("ing", "ed", "es", "s", "e")  # "collaborating" and "collaborate" -> "collaborat"
# Words too common to count as "job wording" when deciding whether a small edit has a purpose.
FILLER_WORDS = {
    "the", "and", "with", "within", "for", "from", "into", "onto", "using", "via", "across",
    "while", "also", "which", "that", "this", "their", "your", "our", "its", "part", "took",
    "take", "making", "made",
}  # fmt: skip
LATEX_SPECIALS = "&%#_$"
# A word, keeping the symbols tech names use: Node.js, C#, C++, CI/CD, S3 (but not a
# trailing full stop, and hyphens split "Docker-based" into "Docker" and "based").
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./]*[A-Za-z0-9+#]|[A-Za-z0-9]")
NUMBERISH = re.compile(r"[\d.,]+[KkMm]?")  # 2M, 99.9 (the numbers check covers these)
# Commands that create CV structure. Inside a block's text they'd add bullets or sections
# the parser can't see, so new text may never contain them.
STRUCTURAL_COMMANDS = {
    "resumeItem",
    "resumeSubheading",
    "resumeProjectHeading",
    "resumeItemListStart",
    "resumeItemListEnd",
    "resumeSubHeadingListStart",
    "resumeSubHeadingListEnd",
    "section",
    "item",
    "begin",
    "end",
    "input",
    "include",
    "newcommand",
    "renewcommand",
    "def",
}

CompileFn = Callable[[str], CompileResult]


class Problem(BaseModel):
    """One thing wrong with a plan. `block_id` is None for whole-plan problems."""

    block_id: str | None
    rule: str
    message: str
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        return f"{self.block_id or 'whole plan'}: {self.message}"


class ValidationContext(BaseModel):
    """What the checks compare against: the CV, extra true facts, the job, page count."""

    doc: CVDocument
    facts: str = ""
    job_text: str = ""
    original_pages: int | None = None


class DroppedEdit(BaseModel):
    edit: Edit
    reasons: list[str]


class ValidatedPlan(BaseModel):
    plan: EditPlan
    attempts: int
    dropped: list[DroppedEdit]
    warnings: list[Problem]


# --- Helpers -----------------------------------------------------------------------------


def brace_problems(text: str) -> list[str]:
    """Report unbalanced braces, ignoring escaped ones (\\{ \\})."""
    depth = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return ["has a '}' with no matching '{'"]
        i += 1
    return ["has an unclosed '{'"] if depth > 0 else []


def unescaped(text: str, char: str) -> int:
    return len(re.findall(rf"(?<!\\){re.escape(char)}", text))


def numbers_in(text: str) -> Counter[str]:
    return Counter(re.findall(r"\d+(?:[.,]\d+)*", text))


def split_items(text: str) -> list[str]:
    """Split a skills line on top-level commas: `AWS (Bedrock, S3), Azure` -> 2 items."""
    items, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def compile_text(tex: str) -> CompileResult:
    """Compile LaTeX source in a throwaway folder (used to check the page count)."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "candidate.tex"
        path.write_bytes(tex.encode("utf-8"))
        return compile_tex(path)


def first_latex_error(log: str) -> str:
    for line in log.splitlines():
        if line.startswith("!"):
            return line.lstrip("! ").strip()
    return "see the compile log"


# --- Per-edit checks ---------------------------------------------------------------------
# Each takes the edit and the original block it replaces, and returns its problems.


def check_latex(edit: Edit, block: Block) -> list[Problem]:
    problems = [
        Problem(block_id=edit.block_id, rule="braces", message=f"new text {issue}")
        for issue in brace_problems(edit.new_text)
    ]
    for char in LATEX_SPECIALS:
        if unescaped(edit.new_text, char) > unescaped(block.text, char):
            problems.append(
                Problem(
                    block_id=edit.block_id,
                    rule="escaping",
                    message=f"unescaped '{char}' (write it as '\\{char}')",
                )
            )
    for command in sorted(set(re.findall(r"\\([A-Za-z]+)", edit.new_text)) & STRUCTURAL_COMMANDS):
        problems.append(
            Problem(
                block_id=edit.block_id,
                rule="structure",
                message=f"must not contain \\{command} (only the text of the block may change)",
            )
        )
    return problems


def check_numbers(edit: Edit, block: Block) -> list[Problem]:
    old, new = numbers_in(block.text), numbers_in(edit.new_text)
    if old == new:
        return []
    added = sorted((new - old).elements())
    removed = sorted((old - new).elements())
    details = []
    if added:
        details.append(f"added {', '.join(added)}")
    if removed:
        details.append(f"removed {', '.join(removed)}")
    message = f"numbers must stay unchanged ({'; '.join(details)})"
    return [Problem(block_id=edit.block_id, rule="numbers", message=message)]


def check_length(edit: Edit, block: Block) -> list[Problem]:
    if block.kind == "skills_line" or not block.text:
        return []
    change = (len(edit.new_text) - len(block.text)) / len(block.text)
    if abs(change) <= LENGTH_TOLERANCE:
        return []
    message = (
        f"length changed by {change:+.0%} ({len(block.text)} -> {len(edit.new_text)} "
        f"characters); keep it within {LENGTH_TOLERANCE:.0%}"
    )
    return [Problem(block_id=edit.block_id, rule="length", message=message)]


def check_skills_reorder(edit: Edit, block: Block) -> list[Problem]:
    if block.kind != "skills_line":
        return []
    old, new = Counter(split_items(block.text)), Counter(split_items(edit.new_text))
    if old == new:
        return []
    details = []
    if added := sorted((new - old).elements()):
        details.append(f"added {', '.join(added)}")
    if removed := sorted((old - new).elements()):
        details.append(f"removed {', '.join(removed)}")
    message = f"skills lines may only be reordered ({'; '.join(details)})"
    return [Problem(block_id=edit.block_id, rule="skills_reorder", message=message)]


def plain_text(latex: str) -> str:
    """LaTeX to plain words: `\\textbf{Python} R\\&D` -> ` Python  R&D`."""
    text = re.sub(r"\\[A-Za-z]+\*?", " ", latex)  # drop command names, keep their text
    text = re.sub(r"\\(.)", r"\1", text)  # \& -> &, \% -> %
    return text.replace("{", " ").replace("}", " ")


def tokens(latex: str) -> list[tuple[str, bool]]:
    """Each token, and whether it starts a sentence (where capitals mean nothing)."""
    plain = plain_text(latex)
    found = []
    for match in TOKEN.finditer(plain):
        before = plain[: match.start()].rstrip()
        found.append((match.group(), not before or before[-1] in ".!?:;"))
    return found


def token_forms(token: str) -> set[str]:
    """Lowercase forms a token can be recognised by: `Agile/Scrum` -> itself, agile, scrum."""
    low = token.lower()
    parts = [p for p in re.split(r"[/.]", low) if p]
    return {low, stem(low), *parts, *(stem(p) for p in parts)}


def is_known(token: str, known: set[str]) -> bool:
    low = token.lower()
    if low in known or stem(low) in known:
        return True
    parts = [p for p in re.split(r"[/.]", low) if p]
    return bool(parts) and all(p in known or stem(p) in known for p in parts)


def looks_technical(token: str) -> bool:
    """`Node.js`, `C#`, `CI/CD`, `S3`, `C++`, and camelCase like `PostgreSQL`/`DevOps`."""
    return bool(re.search(r"[0-9+#./]", token) or re.search(r"[a-z][A-Z]", token))


def check_new_skill_tokens(edit: Edit, block: Block, ctx: ValidationContext) -> list[Problem]:
    """Reject skills, tools or technologies that appear nowhere in the CV or facts.md.

    Code can't know what a skill is, so this flags tokens that are new to the CV *and*
    look like a name: technical-looking (`Node.js`, `C#`, `DevOps`), capitalised in the
    middle of a sentence (`on Kubernetes`), or a term the job capitalises (catches
    `Kubernetes` even at the start of a sentence, where every word is capitalised).
    Adjacent flagged tokens are reported together, so `Spring Boot` is one problem.
    """
    known: set[str] = set()
    for text in [b.text for b in ctx.doc.blocks] + [ctx.facts]:
        for token, _ in tokens(text):
            known |= token_forms(token)
    # Names the job capitalises mid-sentence ("experience with Kubernetes"), not words that
    # are only capitalised because they start a sentence or a heading ("Delivery").
    job_names = {
        token.lower()
        for token, starts_sentence in tokens(ctx.job_text)
        if token[0].isupper() and not starts_sentence
    }

    flagged: list[int] = []
    found = tokens(edit.new_text)
    for i, (token, starts_sentence) in enumerate(found):
        if is_known(token, known) or NUMBERISH.fullmatch(token) or token.lower() in {"i", "a"}:
            continue
        mid_sentence_capital = token[0].isupper() and not starts_sentence
        job_name = token[0].isupper() and token.lower() in job_names
        if looks_technical(token) or mid_sentence_capital or job_name:
            flagged.append(i)

    phrases: list[list[str]] = []
    for i in flagged:
        if phrases and i - 1 in flagged:
            phrases[-1].append(found[i][0])
        else:
            phrases.append([found[i][0]])
    return [
        Problem(
            block_id=edit.block_id,
            rule="new_skill",
            message=f"'{' '.join(phrase)}' is not in the CV or facts.md",
        )
        for phrase in phrases
    ]


def words(text: str) -> list[str]:
    """Lowercase words with LaTeX commands removed: `\\textbf{Python}` -> ["python"]."""
    return re.findall(r"[a-z][a-z0-9+#./-]*", re.sub(r"\\[A-Za-z]+", " ", text).lower())


def stem(word: str) -> str:
    """Crude word root so "collaborating", "collaborate" and "collaborated" match."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def check_noop(edit: Edit, block: Block, ctx: ValidationContext) -> list[Problem]:
    """Warn about small edits whose new words don't come from the job description.

    A small change towards the job's own vocabulary ("working" -> "collaborating" for a
    posting asking to "collaborate") has a purpose; a synonym swap ("Participated in" ->
    "Took part in") doesn't.
    """
    if block.kind == "skills_line":
        return []  # reordering is the only allowed change, so it never adds words
    old, new = words(block.text), words(edit.new_text)
    kept = sum(m.size for m in difflib.SequenceMatcher(None, old, new).get_matching_blocks())
    changed = max(len(old), len(new)) - kept
    if changed > NOOP_MAX_CHANGED_WORDS and kept / max(len(old), len(new), 1) < NOOP_KEPT_SHARE:
        return []  # a substantial rewrite

    old_stems = {stem(word) for word in old}
    introduced = sorted(
        {w for w in new if stem(w) not in old_stems and len(w) > 3 and w not in FILLER_WORDS}
    )
    job_stems = {stem(word) for word in words(ctx.job_text)}
    if any(stem(word) in job_stems for word in introduced):
        return []  # small, but it brings in the job's wording

    added = f" ({', '.join(introduced)})" if introduced else ""
    message = f"barely changes the text and adds no job wording{added}"
    return [Problem(block_id=edit.block_id, rule="noop", message=message, severity="warning")]


# --- Whole-plan checks -------------------------------------------------------------------


def check_ids(ctx: ValidationContext, plan: EditPlan) -> list[Problem]:
    editable = {block.id: block.editable for block in ctx.doc.blocks}
    problems = []
    seen: set[str] = set()
    for edit in plan.edits:
        if edit.block_id not in editable:
            problems.append(Problem(block_id=edit.block_id, rule="ids", message="unknown block ID"))
        elif not editable[edit.block_id]:
            problems.append(
                Problem(block_id=edit.block_id, rule="ids", message="block is read-only")
            )
        if edit.block_id in seen:
            problems.append(
                Problem(block_id=edit.block_id, rule="ids", message="edited more than once")
            )
        seen.add(edit.block_id)
    return problems


def check_reparse(ctx: ValidationContext, plan: EditPlan, tailored: str) -> list[Problem]:
    """Parse the tailored LaTeX again: everything not edited must be byte-identical."""
    try:
        reparsed = parse(tailored)
    except LatexParseError as error:
        return [
            Problem(block_id=None, rule="reparse", message=f"tailored CV doesn't parse: {error}")
        ]
    if reparsed.preamble != ctx.doc.preamble:
        return [Problem(block_id=None, rule="reparse", message="preamble changed")]
    expected = {block.id: block.text for block in ctx.doc.blocks}
    expected.update({edit.block_id: edit.new_text for edit in plan.edits})
    actual = {block.id: block.text for block in reparsed.blocks}
    problems = []
    for block_id in sorted(set(expected) | set(actual)):
        if expected.get(block_id) != actual.get(block_id):
            edited = block_id in {edit.block_id for edit in plan.edits}
            message = (
                "new text changes the CV's structure (e.g. a stray brace or line break)"
                if edited
                else "a block that wasn't edited changed"
            )
            problems.append(
                Problem(block_id=block_id if edited else None, rule="reparse", message=message)
            )
    return problems


def check_compile(ctx: ValidationContext, tailored: str, compile_fn: CompileFn) -> list[Problem]:
    result = compile_fn(tailored)
    if not result.success:
        message = f"tailored CV doesn't compile: {first_latex_error(result.log)}"
        return [Problem(block_id=None, rule="compile", message=message)]
    pages, limit = result.page_count, ctx.original_pages
    if pages is not None and limit is not None and pages > limit:
        message = f"tailored CV has {pages} pages but the original has {limit}; shorten edits"
        return [Problem(block_id=None, rule="pages", message=message)]
    return []


# --- Running the checks ------------------------------------------------------------------


def validate(
    ctx: ValidationContext, plan: EditPlan, compile_fn: CompileFn | None = None
) -> list[Problem]:
    """Run every check and return all problems (errors and warnings)."""
    problems = check_ids(ctx, plan)
    blocks = {block.id: block for block in ctx.doc.blocks}
    bad_ids = {p.block_id for p in problems}
    for edit in plan.edits:
        if edit.block_id in bad_ids:
            continue
        block = blocks[edit.block_id]
        problems += check_latex(edit, block)
        problems += check_numbers(edit, block)
        problems += check_length(edit, block)
        problems += check_skills_reorder(edit, block)
        problems += check_new_skill_tokens(edit, block, ctx)
        problems += check_noop(edit, block, ctx)

    if any(p.severity == "error" for p in problems):
        return problems  # whole-plan checks need a clean plan to render
    tailored = render(ctx.doc, {edit.block_id: edit.new_text for edit in plan.edits})
    problems += check_reparse(ctx, plan, tailored)
    if compile_fn is not None and not any(p.severity == "error" for p in problems):
        problems += check_compile(ctx, tailored, compile_fn)
    return problems


def errors_of(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


def format_problems(problems: list[Problem]) -> str:
    return "\n".join(f"- {problem}" for problem in problems)


def generate_valid_plan(
    ctx: ValidationContext,
    job: JobPosting,
    llm: LLMClient,
    compile_fn: CompileFn | None = None,
    max_retries: int = MAX_RETRIES,
) -> ValidatedPlan:
    """Generate, validate, and retry with the errors; then drop edits that still fail."""
    plan = generate_edits(ctx.doc, job, llm)
    problems = validate(ctx, plan, compile_fn)
    attempts = 1
    while errors_of(problems) and attempts <= max_retries:
        plan = generate_edits(
            ctx.doc, job, llm, previous_plan=plan, errors=format_problems(errors_of(problems))
        )
        problems = validate(ctx, plan, compile_fn)
        attempts += 1

    plan, dropped, problems = drop_failing_edits(ctx, plan, problems, compile_fn)
    warnings = [p for p in problems if p.severity == "warning"]
    return ValidatedPlan(plan=plan, attempts=attempts, dropped=dropped, warnings=warnings)


def drop_failing_edits(
    ctx: ValidationContext,
    plan: EditPlan,
    problems: list[Problem],
    compile_fn: CompileFn | None,
) -> tuple[EditPlan, list[DroppedEdit], list[Problem]]:
    """Remove edits until the plan passes.

    Edits named in an error are dropped first. A whole-plan error (page overflow,
    compile failure) has no single culprit, so the edit that grew the most goes next.
    """
    blocks = {block.id: block for block in ctx.doc.blocks}
    dropped: list[DroppedEdit] = []
    edits = list(plan.edits)
    while errors_of(problems) and edits:
        errors = errors_of(problems)
        named = {p.block_id for p in errors if p.block_id is not None}
        if named:
            for edit in [e for e in edits if e.block_id in named]:
                reasons = [p.message for p in errors if p.block_id == edit.block_id]
                dropped.append(DroppedEdit(edit=edit, reasons=reasons))
            edits = [e for e in edits if e.block_id not in named]
        else:

            def growth(edit: Edit) -> int:
                return len(edit.new_text) - len(blocks[edit.block_id].text)

            biggest = max(edits, key=growth)
            dropped.append(DroppedEdit(edit=biggest, reasons=[p.message for p in errors]))
            edits.remove(biggest)
        plan = EditPlan(edits=edits)
        problems = validate(ctx, plan, compile_fn)
    return plan, dropped, problems
