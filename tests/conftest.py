"""Shared test helpers."""

import pytest

from applybot.agent.graph import route_after_review
from applybot.agent.state import ReviewDecision


def review_router_ready() -> bool:
    """True once route_after_review (a TODO(human) in agent/graph.py) is implemented."""
    try:
        route_after_review({"decision": ReviewDecision(action="approve")})
    except NotImplementedError:
        return False
    return True


needs_review_router = pytest.mark.skipif(
    not review_router_ready(),
    reason="TODO(human): implement route_after_review in agent/graph.py",
)
