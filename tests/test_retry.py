"""The generate -> validate -> retry -> drop loop, with a scripted fake LLM."""

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from applybot.latex.compiler import CompileResult
from applybot.latex.parser import parse
from applybot.tools.extract_job import JobPosting
from applybot.tools.validate import ValidationContext, generate_valid_plan

T = TypeVar("T", bound=BaseModel)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
JOB = JobPosting(
    title="Backend Engineer",
    company="Initech",
    seniority=None,
    required_skills=["Python"],
    nice_to_have=[],
    qualifications=[],
    soft_skills=[],
    responsibilities=[],
    keywords=["REST"],
    raw_text="Backend Engineer. Python, REST.",
)

GOOD_ACME_1 = {
    "block_id": "exp.acme-corp.1",
    "new_text": "Built a \\textbf{Python} Redis-backed ingestion service handling 2M events daily "
    "with 99.9\\% uptime.",
    "rationale": "REST",
}
GOOD_GLOBEX = {
    "block_id": "exp.globex.1",
    "new_text": "Wrote Go integration tests for the billing service, lifting coverage from "
    "40\\% to 75\\%.",
    "rationale": "testing",
}
BAD_METRIC = {
    "block_id": "exp.acme-corp.2",
    "new_text": "Reduced API latency by 50\\% by adding Redis caching to the hottest "
    "\\texttt{GET} endpoints.",
    "rationale": "performance",
}


class ScriptedLLM:
    """Returns the next scripted EditPlan for each call and records the prompts."""

    def __init__(self, plans: list[list[dict[str, Any]]]) -> None:
        self.plans = plans
        self.users: list[str] = []

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        self.users.append(user)
        return schema.model_validate({"edits": self.plans.pop(0)})


@pytest.fixture
def ctx() -> ValidationContext:
    return ValidationContext(doc=parse(SAMPLE.read_bytes().decode("utf-8")), original_pages=1)


def test_valid_first_time(ctx: ValidationContext) -> None:
    llm = ScriptedLLM([[GOOD_ACME_1]])
    result = generate_valid_plan(ctx, JOB, llm)

    assert result.attempts == 1
    assert [e.block_id for e in result.plan.edits] == ["exp.acme-corp.1"]
    assert result.dropped == []


def test_errors_are_fed_back_and_fixed(ctx: ValidationContext) -> None:
    llm = ScriptedLLM([[GOOD_ACME_1, BAD_METRIC], [GOOD_ACME_1]])
    result = generate_valid_plan(ctx, JOB, llm)

    assert result.attempts == 2
    assert result.dropped == []
    retry = llm.users[1]
    assert "<previous_attempt>" in retry
    assert "<validation_errors>" in retry
    assert "- exp.acme-corp.2: numbers must stay unchanged (added 50; removed 35)" in retry


def test_still_failing_edits_are_dropped(ctx: ValidationContext) -> None:
    attempt = [GOOD_ACME_1, BAD_METRIC]
    llm = ScriptedLLM([attempt, attempt, attempt])
    result = generate_valid_plan(ctx, JOB, llm)

    assert result.attempts == 3  # first try + 2 retries
    assert [e.block_id for e in result.plan.edits] == ["exp.acme-corp.1"]  # the good one stays
    assert [d.edit.block_id for d in result.dropped] == ["exp.acme-corp.2"]
    assert result.dropped[0].reasons == ["numbers must stay unchanged (added 50; removed 35)"]


def test_page_overflow_drops_the_biggest_growth(ctx: ValidationContext) -> None:
    def compile_fn(tex: str) -> CompileResult:
        # Pretend the CV only fits on one page without the longer Acme edit.
        pages = 2 if "Redis-backed ingestion" in tex else 1
        return CompileResult(success=True, log="", page_count=pages)

    attempt = [GOOD_ACME_1, GOOD_GLOBEX]
    llm = ScriptedLLM([attempt, attempt, attempt])
    result = generate_valid_plan(ctx, JOB, llm, compile_fn=compile_fn)

    assert [e.block_id for e in result.plan.edits] == ["exp.globex.1"]
    assert [d.edit.block_id for d in result.dropped] == ["exp.acme-corp.1"]
    assert "2 pages" in result.dropped[0].reasons[0]
