"""Structured LLM calls: ask for data that fits a Pydantic model, get a validated instance.

Callers depend only on `LLMClient`. `AnthropicClient` implements it with the official
`anthropic` SDK using tool use: the schema becomes a tool's input schema, and the model
"answers" by calling that tool. Our code never runs the tool; it just reads the input.
"""

import os
import re
from typing import Protocol, TypeVar

import anthropic
from anthropic.types import MessageParam, ToolParam
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 16000


class LLMConfigError(RuntimeError):
    """Required configuration (e.g. ANTHROPIC_MODEL) is missing."""


class StructuredOutputError(RuntimeError):
    """The model did not produce valid structured output, even after retries."""


class LLMClient(Protocol):
    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of `schema` filled in by the model."""
        ...


def tool_name_for(schema: type[BaseModel]) -> str:
    """`JobDetails` -> `record_job_details`."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", schema.__name__).lower()
    return f"record_{snake}"


def format_validation_error(error: ValidationError) -> str:
    """One line per problem, e.g. `required_skills.2: String should not be empty`."""
    lines = []
    for problem in error.errors(include_url=False):
        location = ".".join(str(part) for part in problem["loc"]) or "(root)"
        lines.append(f"{location}: {problem['msg']}")
    return "\n".join(lines)


class AnthropicClient:
    """`LLMClient` backed by the Anthropic Messages API."""

    def __init__(
        self,
        sdk: anthropic.Anthropic | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        load_dotenv()  # read .env if present; real environment variables win
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL")
        if not resolved_model:
            raise LLMConfigError("Set ANTHROPIC_MODEL in your environment or .env file.")
        self.model = resolved_model
        self.sdk = sdk or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        name = tool_name_for(schema)
        tool: ToolParam = {
            "name": name,
            "description": f"Record your answer as a {schema.__name__}.",
            "input_schema": schema.model_json_schema(),
            "strict": True,  # the API guarantees the tool input matches the schema's shape
        }
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        last_problem = ""

        for _ in range(1 + self.max_retries):
            response = self.sdk.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=f"{system}\n\nAlways give your answer by calling the `{name}` tool.",
                tools=[tool],
                # Forcing the tool ("any"/"tool") is rejected by current models, so we use
                # "auto" and handle the case where the model replies in prose instead.
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=messages,
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                # Asking again won't fix these, so fail straight away.
                raise StructuredOutputError(f"Model stopped with reason '{response.stop_reason}'")

            # Keep the model's full turn so the next attempt sees what it said before.
            messages.append({"role": "assistant", "content": response.content})
            tool_use = next(
                (b for b in response.content if b.type == "tool_use" and b.name == name), None
            )

            if tool_use is None:
                last_problem = "The model replied without calling the tool."
                messages.append(
                    {"role": "user", "content": f"Please call the `{name}` tool with your answer."}
                )
                continue

            try:
                return schema.model_validate(tool_use.input)
            except ValidationError as error:
                last_problem = format_validation_error(error)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "is_error": True,
                                "content": (
                                    f"The input was invalid:\n{last_problem}\n"
                                    "Fix these problems and call the tool again."
                                ),
                            }
                        ],
                    }
                )

        attempts = 1 + self.max_retries
        raise StructuredOutputError(
            f"No valid {schema.__name__} after {attempts} attempts. Last problem:\n{last_problem}"
        )
