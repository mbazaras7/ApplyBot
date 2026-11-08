"""Validator tests, including adversarial edits a model might produce."""

from pathlib import Path

import pytest

from applybot.latex.compiler import CompileResult
from applybot.latex.models import Block
from applybot.latex.parser import parse
from applybot.latex.renderer import render
from applybot.tools.generate_edits import Edit, EditPlan
from applybot.tools.validate import (
    CompileFn,
    Problem,
    ValidationContext,
    brace_problems,
    check_noop,
    check_reparse,
    drop_failing_edits,
    errors_of,
    split_items,
    stem,
    validate,
    words,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
ACME_1 = "exp.acme-corp.1"
# Built a \textbf{Python} ingestion service processing 2M events per day with 99.9\% uptime.
ACME_2 = "exp.acme-corp.2"
# Reduced API latency by 35\% by adding Redis caching to the hottest \texttt{GET} endpoints.


@pytest.fixture
def ctx() -> ValidationContext:
    return ValidationContext(doc=parse(SAMPLE.read_bytes().decode("utf-8")), original_pages=1)


def check(ctx: ValidationContext, block_id: str, new_text: str) -> list[Problem]:
    return validate(
        ctx, EditPlan(edits=[Edit(block_id=block_id, new_text=new_text, rationale="r")])
    )


def rules(problems: list[Problem]) -> set[str]:
    return {p.rule for p in problems}


GOOD_ACME_1 = (
    "Built a \\textbf{Python} Redis-backed ingestion service handling 2M events daily "
    "with 99.9\\% uptime."
)


def test_good_edit_passes(ctx: ValidationContext) -> None:
    assert errors_of(check(ctx, ACME_1, GOOD_ACME_1)) == []


# --- IDs -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block_id", "message"),
    [("exp.nope.1", "unknown block ID"), ("edu.springfield-university.1", "block is read-only")],
)
def test_bad_ids(ctx: ValidationContext, block_id: str, message: str) -> None:
    problems = check(ctx, block_id, "x")
    assert [p.message for p in problems] == [message]


def test_duplicate_edits(ctx: ValidationContext) -> None:
    edit = Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r")
    problems = validate(ctx, EditPlan(edits=[edit, edit]))
    assert "edited more than once" in {p.message for p in problems}


# --- Adversarial: broken LaTeX -------------------------------------------------------------


@pytest.mark.parametrize(
    "new_text",
    [
        "Built a \\textbf{Python ingestion service processing 2M events per day, 99.9\\% up.",
        "Built a Python} ingestion service processing 2M events per day with 99.9\\% uptime.",
    ],
)
def test_broken_braces(ctx: ValidationContext, new_text: str) -> None:
    assert "braces" in rules(check(ctx, ACME_1, new_text))


def test_escaped_braces_are_fine() -> None:
    assert brace_problems("a \\{ literal \\} brace and \\textbf{bold}") == []


@pytest.mark.parametrize(("char", "text"), [("&", "R&D"), ("%", "99.9%"), ("_", "snake_case")])
def test_unescaped_specials(ctx: ValidationContext, char: str, text: str) -> None:
    new_text = f"Built a \\textbf{{Python}} {text} service processing 2M events per day, 99.9 up."
    problems = [p for p in check(ctx, ACME_1, new_text) if p.rule == "escaping"]
    assert any(f"'{char}'" in p.message for p in problems)


# --- Numbers and length --------------------------------------------------------------------


def test_changed_metric(ctx: ValidationContext) -> None:
    new_text = (
        "Reduced API latency by 50\\% by adding Redis caching to the hottest "
        "\\texttt{GET} endpoints."
    )
    problems = [p for p in check(ctx, ACME_2, new_text) if p.rule == "numbers"]
    assert problems[0].message == "numbers must stay unchanged (added 50; removed 35)"


def test_too_long(ctx: ValidationContext) -> None:
    new_text = GOOD_ACME_1 + " Also mentored two junior engineers on testing practices weekly."
    assert "length" in rules(check(ctx, ACME_1, new_text))


# --- Skills lines --------------------------------------------------------------------------


def test_skills_reorder_is_fine(ctx: ValidationContext) -> None:
    assert errors_of(check(ctx, "skills.languages", "SQL, Go, Python, TypeScript")) == []


def test_skills_item_added(ctx: ValidationContext) -> None:
    problems = check(ctx, "skills.languages", "Python, Go, TypeScript, SQL, Rust")
    assert [p.message for p in problems if p.rule == "skills_reorder"] == [
        "skills lines may only be reordered (added Rust)"
    ]


def test_split_items_respects_brackets() -> None:
    assert split_items("AWS (Bedrock, S3), Azure, Docker") == [
        "AWS (Bedrock, S3)",
        "Azure",
        "Docker",
    ]


# --- Adversarial: invented skills ----------------------------------------------------------


def test_invented_kubernetes_is_rejected(ctx: ValidationContext) -> None:
    new_text = (
        "Built a \\textbf{Python} ingestion service on Kubernetes processing 2M events per "
        "day with 99.9\\% uptime."
    )
    problems = [p for p in check(ctx, ACME_1, new_text) if p.rule == "new_skill"]
    assert len(problems) == 1
    assert "Kubernetes" in problems[0].message


def test_known_skills_are_allowed(ctx: ValidationContext) -> None:
    # Redis and Docker are in the sample CV; Rust is in facts.md.
    ctx.facts = "I also build command-line tools in Rust."
    new_text = (
        "Built a \\textbf{Python} and Rust ingestion service on Docker processing 2M events "
        "per day at 99.9\\% uptime."
    )
    assert [p for p in check(ctx, ACME_1, new_text) if p.rule == "new_skill"] == []


def skill_problems(ctx: ValidationContext, new_text: str) -> list[str]:
    return [p.message for p in check(ctx, ACME_1, new_text) if p.rule == "new_skill"]


@pytest.mark.parametrize(
    ("new_text", "expected"),
    [
        # Technical-looking names are flagged wherever they appear.
        ("Built a Node.js service processing 2M events per day.", "Node.js"),
        ("Built a C# service processing 2M events per day.", "C#"),
        ("Built a DevOps pipeline processing 2M events per day.", "DevOps"),
        # Adjacent capitalised words are one name.
        ("Built a Spring Boot service processing 2M events per day.", "Spring Boot"),
    ],
)
def test_invented_names(ctx: ValidationContext, new_text: str, expected: str) -> None:
    assert skill_problems(ctx, new_text) == [f"'{expected}' is not in the CV or facts.md"]


def test_sentence_start_capitals_are_ordinary_words(ctx: ValidationContext) -> None:
    # "Designed" is new to the CV but just a capitalised first word.
    assert skill_problems(ctx, "Designed a Python ingestion service for 2M events per day.") == []


def test_job_terms_are_caught_even_at_sentence_start(ctx: ValidationContext) -> None:
    ctx.job_text = "Experience with Kubernetes and Terraform is required."
    new_text = "Kubernetes-hosted Python ingestion service processing 2M events per day."
    assert skill_problems(ctx, new_text) == ["'Kubernetes' is not in the CV or facts.md"]


def test_job_heading_words_stay_ordinary(ctx: ValidationContext) -> None:
    # The job capitalises "Delivery" only as a heading; in the edit it's a plain word.
    ctx.job_text = "Delivery\nYou will own end-to-end delivery of features."
    new_text = "Built a Python ingestion service with end-to-end delivery of 2M events per day."
    assert skill_problems(ctx, new_text) == []


def test_known_words_in_other_forms(ctx: ValidationContext) -> None:
    # "GitHub" is known from "GitHub Actions"; "Docker-based" splits into known "Docker";
    # case doesn't matter ("postgresql").
    new_text = "Shipped a Docker-based Python service on GitHub using postgresql, 2M events/day."
    assert skill_problems(ctx, new_text) == []


# --- No-op warning -------------------------------------------------------------------------


def test_noop_is_only_a_warning(ctx: ValidationContext) -> None:
    # Like "Participated in" -> "Took part in" from the first demo: one word swapped.
    new_text = (
        "Built a \\textbf{Python} ingestion service processing 2M events each day with "
        "99.9\\% uptime."
    )
    problems = check(ctx, ACME_1, new_text)
    assert [(p.rule, p.severity) for p in problems] == [("noop", "warning")]


TEAM_BULLET = Block(
    id="exp.x.1",
    kind="item",
    section="Experience",
    parent_heading="X",
    text="Participated in sprint planning and shipped features as part of a 6-person team, "
    "working across a React frontend and Go services.",
    editable=True,
)
JOB_TEXT = "You will collaborate with global teams to design, build, test and deploy solutions."


def noop(new_text: str, job_text: str = JOB_TEXT) -> list[str]:
    ctx = ValidationContext(doc=parse(SAMPLE.read_bytes().decode("utf-8")), job_text=job_text)
    edit = Edit(block_id=TEAM_BULLET.id, new_text=new_text, rationale="r")
    return [p.message for p in check_noop(edit, TEAM_BULLET, ctx)]


def test_small_edit_using_job_wording_is_fine() -> None:
    # "working" -> "collaborating" mirrors the job's "collaborate".
    new_text = TEAM_BULLET.text.replace("working across", "collaborating across")
    assert noop(new_text) == []


def test_synonym_swap_is_warned() -> None:
    new_text = TEAM_BULLET.text.replace("Participated in", "Took part in")
    assert noop(new_text) == ["barely changes the text and adds no job wording"]


def test_one_job_word_excuses_a_mixed_edit() -> None:
    # "Took part in" is cosmetic, but "deployed" mirrors the job's "deploy".
    new_text = TEAM_BULLET.text.replace("Participated in", "Took part in").replace(
        "shipped", "deployed"
    )
    assert noop(new_text) == []


def test_same_edit_without_matching_job_is_warned() -> None:
    new_text = TEAM_BULLET.text.replace("working across", "collaborating across")
    assert noop(new_text, job_text="Python developer for a data team.") == [
        "barely changes the text and adds no job wording (collaborating)"
    ]


def test_substantial_rewrite_is_not_a_noop() -> None:
    new_text = (
        "Planned sprints and delivered React and Go features end to end in a 6-person "
        "team, from frontend design to service rollout."
    )
    assert noop(new_text, job_text="") == []


def test_skills_reorder_is_not_a_noop(ctx: ValidationContext) -> None:
    problems = check(ctx, "skills.languages", "SQL, Go, Python, TypeScript")
    assert [p for p in problems if p.rule == "noop"] == []


def test_words_split_slashes_and_drop_full_stops() -> None:
    # Found on a real posting: "Typescript/React" hid "react" from the no-op check.
    assert words("Web Development using Typescript/React.") == [
        "web", "development", "using", "typescript", "react",
    ]  # fmt: skip
    assert words("Built with Node.js on Azure infrastructure.") == [
        "built", "with", "node.js", "on", "azure", "infrastructure",
    ]  # fmt: skip


def test_stem_matches_word_forms() -> None:
    assert stem("collaborating") == stem("collaborate") == stem("collaborated")
    assert stem("deployed") == stem("deploy")


# --- Whole-plan checks ---------------------------------------------------------------------


def fake_compile(pages: int, success: bool = True) -> CompileFn:
    def compile_fn(tex: str) -> CompileResult:
        log = "" if success else "! Undefined control sequence.\nl.42 \\resumeItme"
        return CompileResult(success=success, log=log, page_count=pages if success else None)

    return compile_fn


def test_page_overflow(ctx: ValidationContext) -> None:
    plan = EditPlan(edits=[Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r")])
    problems = errors_of(validate(ctx, plan, compile_fn=fake_compile(pages=2)))
    assert [p.message for p in problems] == [
        "tailored CV has 2 pages but the original has 1; shorten edits"
    ]


def test_compile_failure(ctx: ValidationContext) -> None:
    plan = EditPlan(edits=[Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r")])
    problems = errors_of(validate(ctx, plan, compile_fn=fake_compile(pages=1, success=False)))
    assert problems[0].message == "tailored CV doesn't compile: Undefined control sequence."


def test_smuggled_bullet_is_rejected(ctx: ValidationContext) -> None:
    # Braces balance, but it adds a nested bullet the parser would treat as plain text.
    new_text = (
        "Built a \\textbf{Python} service: 2M events per day, 99.9\\% uptime. \\resumeItem{Led QA}"
    )
    problems = [p for p in check(ctx, ACME_1, new_text) if p.rule == "structure"]
    assert [p.message for p in problems] == [
        "must not contain \\resumeItem (only the text of the block may change)"
    ]


def test_drop_named_edits_keeps_the_rest(ctx: ValidationContext) -> None:
    bad = Edit(
        block_id=ACME_2,
        new_text="Reduced API latency by 50\\% by adding Redis caching to the hottest "
        "\\texttt{GET} endpoints.",
        rationale="r",
    )
    plan = EditPlan(edits=[Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r"), bad])
    kept, dropped, problems = drop_failing_edits(ctx, plan, validate(ctx, plan), None)

    assert [e.block_id for e in kept.edits] == [ACME_1]
    assert [(d.edit.block_id, d.reasons) for d in dropped] == [
        (ACME_2, ["numbers must stay unchanged (added 50; removed 35)"])
    ]
    assert errors_of(problems) == []


def test_page_overflow_drops_the_biggest_growth(ctx: ValidationContext) -> None:
    def compile_fn(tex: str) -> CompileResult:
        # Pretend the CV only fits on one page without the longer Acme edit.
        return CompileResult(success=True, log="", page_count=2 if "Redis-backed" in tex else 1)

    globex = Edit(
        block_id="exp.globex.1",
        new_text="Wrote Go integration tests for the billing service, lifting coverage from "
        "40\\% to 75\\%.",
        rationale="r",
    )
    plan = EditPlan(edits=[Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r"), globex])
    problems = validate(ctx, plan, compile_fn)
    kept, dropped, _ = drop_failing_edits(ctx, plan, problems, compile_fn)

    assert [e.block_id for e in kept.edits] == ["exp.globex.1"]
    assert [d.edit.block_id for d in dropped] == [ACME_1]
    assert "2 pages" in dropped[0].reasons[0]


def test_reparse_backstop(ctx: ValidationContext) -> None:
    # The re-parse check is the last line of defence: feed it tampered output directly.
    plan = EditPlan(edits=[Edit(block_id=ACME_1, new_text=GOOD_ACME_1, rationale="r")])
    tailored = render(ctx.doc, {ACME_1: GOOD_ACME_1})

    assert check_reparse(ctx, plan, tailored) == []
    assert [p.message for p in check_reparse(ctx, plan, "% hi\n" + tailored)] == [
        "preamble changed"
    ]
    tampered = tailored.replace("Wrote integration tests", "Wrote unit tests")
    assert [p.message for p in check_reparse(ctx, plan, tampered)] == [
        "a block that wasn't edited changed"
    ]
