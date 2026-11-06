"""AnthropicClient tests with a fake SDK: no network, no API key, no cost."""

import copy
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, field_validator

from applybot.llm import client as client_module
from applybot.llm.client import AnthropicClient, LLMConfigError, StructuredOutputError


class Pet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    legs: int

    @field_validator("legs")
    @classmethod
    def not_negative(cls, legs: int) -> int:
        if legs < 0:
            raise ValueError("legs must not be negative")
        return legs


def tool_reply(data: dict[str, Any], tool_id: str = "toolu_1") -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "tool_use", "id": tool_id, "name": "record_pet", "input": data}],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def text_reply(text: str, stop_reason: str = "end_turn") -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class FakeMessages:
    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        # Copy, because the client keeps appending to the same messages list.
        self.calls.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


class FakeSDK:
    def __init__(self, replies: list[Message]) -> None:
        self.messages = FakeMessages(replies)


def make_client(replies: list[Message]) -> tuple[AnthropicClient, FakeMessages]:
    fake = FakeSDK(replies)
    llm = AnthropicClient(sdk=cast(anthropic.Anthropic, fake), model="test-model")
    return llm, fake.messages


def test_valid_first_answer() -> None:
    llm, calls = make_client([tool_reply({"name": "Rex", "legs": 4})])

    pet = llm.structured(system="Describe a pet.", user="A dog", schema=Pet)

    assert pet == Pet(name="Rex", legs=4)
    request = calls.calls[0]
    assert request["model"] == "test-model"
    assert request["tools"][0]["name"] == "record_pet"
    assert request["tools"][0]["strict"] is True
    assert request["tools"][0]["input_schema"]["additionalProperties"] is False
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_invalid_answer_is_retried_with_the_errors() -> None:
    llm, calls = make_client(
        [tool_reply({"name": "Rex", "legs": -1}), tool_reply({"name": "Rex", "legs": 4})]
    )

    pet = llm.structured(system="s", user="A dog", schema=Pet)

    assert pet.legs == 4
    assert len(calls.calls) == 2
    feedback = calls.calls[1]["messages"][-1]
    assert feedback["role"] == "user"
    result = feedback["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_1"
    assert result["is_error"] is True
    assert "legs: Value error, legs must not be negative" in result["content"]


def test_prose_reply_is_retried() -> None:
    llm, calls = make_client(
        [text_reply("Rex is a dog with four legs."), tool_reply({"name": "Rex", "legs": 4})]
    )

    assert llm.structured(system="s", user="A dog", schema=Pet).name == "Rex"
    nudge = calls.calls[1]["messages"][-1]
    assert nudge == {
        "role": "user",
        "content": "Please call the `record_pet` tool with your answer.",
    }


def test_gives_up_after_three_attempts() -> None:
    bad = {"name": "Rex", "legs": -1}
    llm, calls = make_client([tool_reply(bad), tool_reply(bad), tool_reply(bad)])

    with pytest.raises(StructuredOutputError, match="after 3 attempts"):
        llm.structured(system="s", user="A dog", schema=Pet)
    assert len(calls.calls) == 3


@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
def test_unrecoverable_stop_reasons_fail_immediately(reason: str) -> None:
    llm, calls = make_client([text_reply("", stop_reason=reason)])

    with pytest.raises(StructuredOutputError, match=reason):
        llm.structured(system="s", user="A dog", schema=Pet)
    assert len(calls.calls) == 1


def test_model_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_MODEL", "model-from-env")
    llm = AnthropicClient(sdk=cast(anthropic.Anthropic, FakeSDK([])))
    assert llm.model == "model-from-env"


def test_missing_model_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_MODEL"):
        AnthropicClient(sdk=cast(anthropic.Anthropic, FakeSDK([])))
