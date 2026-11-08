"""Fake LLM and reviewer for driving the graph in tests."""

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from applybot.agent.state import ReviewDecision
from applybot.review.diff import ReviewRequest

T = TypeVar("T", bound=BaseModel)

JOB_DETAILS: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": "Initech Ltd",
    "seniority": "junior",
    "required_skills": ["Python"],
    "nice_to_have": [],
    "qualifications": [],
    "soft_skills": [],
    "responsibilities": ["Build services"],
    "keywords": ["Python"],
}

# Edits to tests/fixtures/sample_cv.tex
GOOD_ACME = {
    "block_id": "exp.acme-corp.1",
    "new_text": "Built a \\textbf{Python} Redis-backed ingestion service handling 2M events "
    "daily with 99.9\\% uptime.",
    "rationale": "Python services",
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
    """Answers JobDetails with JOB_DETAILS and each EditPlan request with the next plan."""

    def __init__(self, plans: list[list[dict[str, Any]]]) -> None:
        self.plans = plans
        self.edit_prompts: list[str] = []

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        if schema.__name__ == "JobDetails":
            return schema.model_validate(JOB_DETAILS)
        self.edit_prompts.append(user)
        return schema.model_validate({"edits": self.plans.pop(0)})


class ScriptedReviewer:
    """Answers each review with the next scripted step and records what it was shown."""

    def __init__(self, steps: list[Callable[[ReviewRequest], ReviewDecision]]) -> None:
        self.steps = steps
        self.requests: list[ReviewRequest] = []

    def __call__(self, request: ReviewRequest) -> ReviewDecision:
        self.requests.append(request)
        return self.steps.pop(0)(request)


def approve(request: ReviewRequest) -> ReviewDecision:
    return ReviewDecision(action="approve")


def reject(comment: str) -> Callable[[ReviewRequest], ReviewDecision]:
    return lambda request: ReviewDecision(action="reject", comment=comment)


def edit_text(block_id: str, new_text: str) -> Callable[[ReviewRequest], ReviewDecision]:
    """Replace one edit's text, as if the reviewer changed it in the editor."""

    def step(request: ReviewRequest) -> ReviewDecision:
        edits = [
            e.model_copy(update={"new_text": new_text}) if e.block_id == block_id else e
            for e in request.plan.edits
        ]
        return ReviewDecision(action="edit", plan=request.plan.model_copy(update={"edits": edits}))

    return step
