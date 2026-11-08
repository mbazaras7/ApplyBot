"""The graph's routing functions: read the state, name the next node."""

import pytest
from conftest import needs_review_router

from applybot.agent.graph import route_after_review, route_after_validate
from applybot.agent.state import ReviewDecision, TailorState
from applybot.tools.validate import Problem

ERROR = Problem(block_id="exp.x.1", rule="numbers", message="numbers must stay unchanged")
WARNING = Problem(block_id="exp.x.1", rule="noop", message="barely changes", severity="warning")


@pytest.mark.parametrize(
    ("problems", "attempts", "expected"),
    [
        ([], 1, "render_diff"),  # passed
        ([WARNING], 1, "render_diff"),  # warnings never block
        ([ERROR], 1, "generate_edits"),  # first retry
        ([ERROR], 2, "generate_edits"),  # second retry
        ([ERROR], 3, "drop_failing"),  # out of retries
    ],
)
def test_route_after_validate(problems: list[Problem], attempts: int, expected: str) -> None:
    state: TailorState = {"problems": problems, "attempts": attempts}
    assert route_after_validate(state) == expected


@needs_review_router
@pytest.mark.parametrize(
    ("action", "expected"),
    [("approve", "compile"), ("edit", "recheck"), ("reject", "generate_edits")],
)
def test_route_after_review(action: str, expected: str) -> None:
    state: TailorState = {"decision": ReviewDecision.model_validate({"action": action})}
    assert route_after_review(state) == expected
