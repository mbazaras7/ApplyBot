"""analyse_and_plan: the analysis schema and how the loop is set up."""

from pathlib import Path
from typing import Any

import pytest
from fakes import ANALYSIS, FakeEmbedder, ScriptedLLM
from pydantic import ValidationError

from applybot.latex.parser import parse
from applybot.tools.analyse import (
    ANALYSIS_PROMPT,
    MAX_STEPS,
    JobAnalysis,
    SearchCV,
    analyse_and_plan,
    analysis_schema,
)
from applybot.tools.cv_index import CVIndex
from applybot.tools.extract_job import JobPosting

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
JOB = JobPosting(
    title="Backend Engineer", company="Initech", seniority="junior",
    required_skills=["Python", "Kubernetes"], nice_to_have=[], qualifications=["CS degree"],
    soft_skills=[], responsibilities=["Build services"], keywords=["Python"],
    raw_text="Backend Engineer at Initech.",
)  # fmt: skip


@pytest.fixture
def index() -> CVIndex:
    return CVIndex.build(parse(SAMPLE.read_bytes().decode("utf-8")), FakeEmbedder())


def test_analysis_and_loop_setup(index: CVIndex) -> None:
    llm = ScriptedLLM([])
    analysis = analyse_and_plan(JOB, index, FakeEmbedder(), llm)

    assert type(analysis) is JobAnalysis
    assert analysis.fit_score == 64
    assert [g.requirement for g in analysis.gaps] == ["Kubernetes"]
    loop = llm.tool_loops[0]
    assert loop["max_steps"] == MAX_STEPS
    assert [t.name for t in loop["tools"]] == ["search_cv"]
    assert "<job_requirements>" in loop["user"]
    assert "raw_text" not in loop["user"]  # the model must search, not read the CV
    assert "search_cv" in ANALYSIS_PROMPT


def test_evidence_must_be_real_ids(index: CVIndex) -> None:
    schema = analysis_schema(index)
    bad: dict[str, Any] = {
        **ANALYSIS,
        "findings": [{**ANALYSIS["findings"][0], "evidence": ["exp.nope.9"]}],
    }
    with pytest.raises(ValidationError):
        schema.model_validate(bad)
    enum = schema.model_json_schema()["$defs"]["RequirementFinding"]["properties"]["evidence"]
    assert "edu.springfield-university" in enum["items"]["enum"]  # headings are evidence too


@pytest.mark.parametrize("score", [-1, 101])
def test_fit_score_range(score: int) -> None:
    with pytest.raises(ValidationError, match="between 0 and 100"):
        JobAnalysis.model_validate({**ANALYSIS, "fit_score": score})


@pytest.mark.parametrize("k", [0, 9])
def test_search_k_range(k: int) -> None:
    with pytest.raises(ValidationError):
        SearchCV(query="Python", k=k)


def test_search_tool_returns_formatted_hits(index: CVIndex) -> None:
    llm = ScriptedLLM([])
    analyse_and_plan(JOB, index, FakeEmbedder(), llm)
    tool = llm.tool_loops[0]["tools"][0]

    result = tool.handler(SearchCV(query="billing service integration tests", k=2))

    assert result.splitlines()[0].startswith("[exp.globex.1] Experience > Globex (score ")
    assert len(result.splitlines()) == 2
