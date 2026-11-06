"""extract_job end to end with a mocked LLM (fake Anthropic SDK underneath)."""

from typing import Any, cast

import anthropic
from anthropic.types import Message

from applybot.llm.client import AnthropicClient
from applybot.tools.extract_job import EXTRACTION_PROMPT, JobDetails, JobPosting, extract_job

POSTING = """Backend Engineer at Initech
We need 2+ years of Python and PostgreSQL. Docker is a plus.
You will build REST APIs and review code."""


def tool_reply(data: dict[str, Any]) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "record_job_details", "input": data}
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class FakeSDK:
    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        self.requests.append(kwargs)
        return self.replies.pop(0)


GOOD = {
    "title": "Backend Engineer",
    "company": "Initech",
    "seniority": None,
    "required_skills": ["Python", "PostgreSQL"],
    "nice_to_have": ["Docker"],
    "qualifications": ["2+ years of experience"],
    "soft_skills": [],
    "responsibilities": ["Build REST APIs", "Review code"],
    "keywords": ["backend", "REST"],
}


def run(replies: list[Message]) -> tuple[JobPosting, FakeSDK]:
    fake = FakeSDK(replies)
    llm = AnthropicClient(sdk=cast(anthropic.Anthropic, fake), model="test-model")
    return extract_job(POSTING, llm), fake


def test_extracts_job_posting() -> None:
    job, fake = run([tool_reply(GOOD)])

    assert job.company == "Initech"
    assert job.required_skills == ["Python", "PostgreSQL"]
    assert job.raw_text == POSTING  # copied by our code, not by the model
    assert fake.requests[0]["system"].startswith(EXTRACTION_PROMPT)
    assert fake.requests[0]["messages"][0]["content"] == f"<job_posting>\n{POSTING}\n</job_posting>"


def test_missing_information_is_valid() -> None:
    sparse = {**GOOD, "title": None, "company": "  ", "nice_to_have": [], "keywords": []}
    job, fake = run([tool_reply(sparse)])

    assert job.title is None
    assert job.company is None  # blank string becomes None
    assert job.nice_to_have == []
    assert len(fake.requests) == 1  # no retry needed


def test_blank_skill_triggers_a_retry() -> None:
    bad = {**GOOD, "required_skills": ["Python", ""]}
    job, fake = run([tool_reply(bad), tool_reply(GOOD)])

    assert job.required_skills == ["Python", "PostgreSQL"]
    assert len(fake.requests) == 2


def test_raw_text_is_not_asked_of_the_model() -> None:
    assert "raw_text" not in JobDetails.model_json_schema()["properties"]
