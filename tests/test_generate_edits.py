from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from applybot.latex.parser import parse
from applybot.latex.renderer import render
from applybot.llm.client import Tool
from applybot.tools.extract_job import JobPosting
from applybot.tools.generate_edits import (
    GENERATION_RULES,
    Edit,
    EditError,
    EditPlan,
    apply_edits,
    describe_cv,
    edit_plan_schema,
    generate_edits,
)

T = TypeVar("T", bound=BaseModel)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"

JOB = JobPosting(
    title="Backend Engineer",
    company="Initech",
    seniority="junior",
    required_skills=["Python"],
    nice_to_have=[],
    qualifications=[],
    soft_skills=[],
    responsibilities=["Build REST APIs"],
    keywords=["Python", "REST"],
    raw_text="Backend Engineer at Initech. Python required.",
)


class FakeLLM:
    """An LLMClient that returns a fixed answer and records the request."""

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        self.requests.append({"system": system, "user": user, "schema": schema})
        return schema.model_validate(self.answer)

    def tool_loop(
        self, *, system: str, user: str, tools: list[Tool], schema: type[T], max_steps: int
    ) -> T:
        raise AssertionError("generate_edits doesn't use the tool loop")


@pytest.fixture
def tex() -> str:
    return SAMPLE.read_bytes().decode("utf-8")


# --- schema ------------------------------------------------------------------------------


def test_schema_only_allows_editable_ids(tex: str) -> None:
    schema = edit_plan_schema(parse(tex)).model_json_schema()
    allowed = schema["$defs"]["Edit"]["properties"]["block_id"]["enum"]

    assert "exp.acme-corp.1" in allowed
    assert "skills.languages" in allowed
    assert "edu.springfield-university.1" not in allowed  # read-only


def test_schema_rejects_unknown_id(tex: str) -> None:
    schema = edit_plan_schema(parse(tex))
    with pytest.raises(ValidationError):
        schema.model_validate(
            {"edits": [{"block_id": "exp.nope.1", "new_text": "x", "rationale": "y"}]}
        )


# --- describe_cv -------------------------------------------------------------------------


def test_describe_cv(tex: str) -> None:
    text = describe_cv(parse(tex))

    assert "[exp.acme-corp.1] Experience > Acme Corp (editable): Built a \\textbf{Python}" in text
    assert "[edu.springfield-university.1] Education > Springfield University (read-only)" in text
    assert "alex@example.com" not in text  # contact header never sent
    assert "\\documentclass" not in text  # nor the preamble


# --- generate_edits ----------------------------------------------------------------------


def test_generate_edits_request(tex: str) -> None:
    edit = {"block_id": "exp.globex.1", "new_text": "New text", "rationale": "Mirrors Go"}
    llm = FakeLLM({"edits": [edit]})

    plan = generate_edits(parse(tex), JOB, llm)

    assert plan == EditPlan(edits=[Edit(**edit)])
    assert type(plan) is EditPlan  # the per-CV schema class doesn't leak out
    request = llm.requests[0]
    assert request["system"] == GENERATION_RULES
    assert "<job_requirements>" in request["user"]
    assert '"raw_text"' not in request["user"]  # raw text is sent once, in <job_posting>
    assert f"<job_posting>\n{JOB.raw_text}\n</job_posting>" in request["user"]
    assert "[exp.globex.1]" in request["user"]


# --- apply_edits -------------------------------------------------------------------------


def test_apply_edits_changes_only_those_blocks(tex: str) -> None:
    doc = parse(tex)
    old = doc.block("exp.globex.1").text
    new = "Wrote Go integration tests for billing, raising coverage from 40\\% to 75\\%."
    plan = EditPlan(edits=[Edit(block_id="exp.globex.1", new_text=new, rationale="r")])

    assert apply_edits(doc, plan) == tex.replace(old, new)


def test_no_edits_gives_original(tex: str) -> None:
    doc = parse(tex)
    assert apply_edits(doc, EditPlan(edits=[])) == render(doc) == tex


@pytest.mark.parametrize(
    ("block_ids", "problem"),
    [
        (["exp.nope.1"], "exp.nope.1: unknown block ID"),
        (["edu.springfield-university.1"], "edu.springfield-university.1: block is read-only"),
        (["exp.globex.1", "exp.globex.1"], "exp.globex.1: edited more than once"),
    ],
)
def test_bad_plans_fail_the_run(tex: str, block_ids: list[str], problem: str) -> None:
    plan = EditPlan(edits=[Edit(block_id=i, new_text="x", rationale="r") for i in block_ids])
    with pytest.raises(EditError, match=problem):
        apply_edits(parse(tex), plan)


def test_all_problems_are_reported_together(tex: str) -> None:
    plan = EditPlan(
        edits=[
            Edit(block_id="exp.nope.1", new_text="x", rationale="r"),
            Edit(block_id="edu.springfield-university.1", new_text="x", rationale="r"),
        ]
    )
    with pytest.raises(EditError) as info:
        apply_edits(parse(tex), plan)
    assert "unknown block ID" in str(info.value)
    assert "read-only" in str(info.value)
