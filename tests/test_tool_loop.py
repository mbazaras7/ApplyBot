"""AnthropicClient.tool_loop with a fake SDK: the ReAct loop and its step cap."""

import copy
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict

from applybot.llm.client import AnthropicClient, StructuredOutputError, Tool


class Lookup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str


def reply(*calls: tuple[str, str, dict[str, Any]], text: str = "") -> Message:
    """A model turn with the given (id, tool name, input) calls."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": i, "name": n, "input": inp} for i, n, inp in calls]
    return Message.model_validate(
        {
            "id": "msg", "type": "message", "role": "assistant", "model": "test",
            "content": content, "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )  # fmt: skip


def lookup(word: str, call_id: str = "t1") -> tuple[str, str, dict[str, Any]]:
    return (call_id, "lookup", {"word": word})


def answer(summary: str, call_id: str = "f1") -> tuple[str, str, dict[str, Any]]:
    return (call_id, "record_answer", {"summary": summary})


class FakeSDK:
    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


def run(replies: list[Message], max_steps: int = 3) -> tuple[Answer, FakeSDK, list[str]]:
    looked_up: list[str] = []

    def handler(arguments: Lookup) -> str:
        looked_up.append(arguments.word)
        return f"definition of {arguments.word}"

    sdk = FakeSDK(replies)
    llm = AnthropicClient(sdk=cast(anthropic.Anthropic, sdk), model="test")
    tool = Tool(name="lookup", description="Look up a word.", input_model=Lookup, handler=handler)
    result = llm.tool_loop(
        system="Explain.", user="Explain CI/CD.", tools=[tool], schema=Answer, max_steps=max_steps
    )
    return result, sdk, looked_up


def last_user_content(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = request["messages"][-1]["content"]
    assert isinstance(content, list)
    return content


def test_tool_call_then_answer() -> None:
    result, sdk, looked_up = run([reply(lookup("CI")), reply(answer("CI means ..."))])

    assert result == Answer(summary="CI means ...")
    assert looked_up == ["CI"]
    results = last_user_content(sdk.requests[1])
    assert results == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "definition of CI",
            "is_error": False,
        }
    ]
    tools = sdk.requests[0]["tools"]
    assert [t["name"] for t in tools] == ["lookup", "record_answer"]
    assert all(t["strict"] for t in tools)
    assert sdk.requests[0]["tool_choice"] == {"type": "auto"}  # parallel calls allowed


def test_parallel_calls_count_as_one_step() -> None:
    # Two lookups in one turn = 1 step, so with max_steps=1 the budget is reached after it.
    parallel = reply(lookup("CI", "t1"), lookup("CD", "t2"))
    result, sdk, looked_up = run([parallel, reply(answer("done"))], max_steps=1)

    assert looked_up == ["CI", "CD"]
    content = last_user_content(sdk.requests[1])
    assert [c["tool_use_id"] for c in content if c["type"] == "tool_result"] == ["t1", "t2"]
    assert "You have used all 1 rounds" in content[-1]["text"]  # told to answer now


def test_cap_stops_running_tools() -> None:
    replies = [reply(lookup("a", "t1")), reply(lookup("b", "t2")), reply(answer("done"))]
    result, sdk, looked_up = run(replies, max_steps=1)

    assert looked_up == ["a"]  # the second lookup is refused, not run
    refused = last_user_content(sdk.requests[2])[0]
    assert refused["is_error"] is True
    assert "Search budget used up" in refused["content"]
    assert result.summary == "done"


def test_endless_tool_calls_give_up() -> None:
    replies = [reply(lookup("a", f"t{i}")) for i in range(5)]
    with pytest.raises(StructuredOutputError, match="kept calling tools"):
        run(replies, max_steps=1)


def test_invalid_answer_is_retried() -> None:
    bad = reply(("f1", "record_answer", {"summary": "x", "extra": 1}))
    result, sdk, _ = run([bad, reply(answer("fixed", "f2"))])

    assert result.summary == "fixed"
    error = last_user_content(sdk.requests[1])[0]
    assert error["is_error"] is True
    assert "The answer was invalid" in error["content"]


def test_prose_reply_is_nudged() -> None:
    result, sdk, _ = run([reply(text="Let me think."), reply(answer("ok"))])

    assert result.summary == "ok"
    assert "call `record_answer`" in sdk.requests[1]["messages"][-1]["content"]


def test_unknown_tool_gets_an_error_result() -> None:
    replies = [reply(("t1", "teleport", {})), reply(answer("ok"))]
    result, sdk, _ = run(replies)

    error = last_user_content(sdk.requests[1])[0]
    assert error == {
        "type": "tool_result", "tool_use_id": "t1", "content": "Unknown tool 'teleport'.",
        "is_error": True,
    }  # fmt: skip
