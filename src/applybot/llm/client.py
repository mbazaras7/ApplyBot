"""Structured LLM calls: ask for data that fits a Pydantic model, get a validated instance.

Callers depend only on `LLMClient`. `AnthropicClient` implements it with the official
`anthropic` SDK using tool use: the schema becomes a tool's input schema, and the model
"answers" by calling that tool. Our code never runs the tool; it just reads the input.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import anthropic
from anthropic.types import MessageParam, TextBlockParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 16000


class LLMConfigError(RuntimeError):
    """Required configuration (e.g. ANTHROPIC_MODEL) is missing."""


class StructuredOutputError(RuntimeError):
    """The model did not produce valid structured output, even after retries."""


@dataclass
class Tool:
    """A tool the model may call during `tool_loop`, run by our code."""

    name: str
    description: str
    input_model: type[BaseModel]  # the tool's arguments, validated with Pydantic
    handler: Callable[[Any], str]  # gets the validated arguments, returns text for the model


class LLMClient(Protocol):
    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of `schema` filled in by the model."""
        ...

    def tool_loop(
        self, *, system: str, user: str, tools: list[Tool], schema: type[T], max_steps: int
    ) -> T:
        """Let the model call `tools` for up to `max_steps` rounds, then return `schema`."""
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

    def tool_loop(
        self, *, system: str, user: str, tools: list[Tool], schema: type[T], max_steps: int
    ) -> T:
        """The ReAct loop: the model calls tools, our code runs them, repeat, then answer.

        One *step* is one round in which tools ran, however many calls the model made in
        parallel. After `max_steps` rounds our code stops running tools and tells the
        model to answer: the cap is enforced here, not left to the model.
        """
        final_name = tool_name_for(schema)
        by_name = {tool.name: tool for tool in tools}
        params: list[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
                "strict": True,
            }
            for tool in tools
        ]
        params.append(
            {
                "name": final_name,
                "description": f"Record your final answer as a {schema.__name__}.",
                "input_schema": schema.model_json_schema(),
                "strict": True,
            }
        )
        system_text = (
            f"{system}\n\nUse the tools to gather what you need, then give your final answer "
            f"by calling `{final_name}`."
        )
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        steps = mistakes = 0
        last_problem = ""

        while mistakes <= self.max_retries:
            response = self.sdk.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_text,
                tools=params,
                tool_choice={"type": "auto"},  # parallel calls allowed: many searches per round
                messages=messages,
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                raise StructuredOutputError(f"Model stopped with reason '{response.stop_reason}'")
            messages.append({"role": "assistant", "content": response.content})

            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                mistakes += 1
                last_problem = "The model replied without calling a tool."
                nudge = f"Please continue: call a tool, or call `{final_name}` with your answer."
                messages.append({"role": "user", "content": nudge})
                continue

            # Every tool call must get a tool_result, all together in one user message.
            results: list[ToolResultBlockParam | TextBlockParam] = []
            ran_tools = over_budget = False
            for call in calls:
                if call.name == final_name:
                    try:
                        return schema.model_validate(call.input)
                    except ValidationError as error:
                        mistakes += 1
                        last_problem = format_validation_error(error)
                        message = f"The answer was invalid:\n{last_problem}\nFix it and call again."
                        results.append(tool_result(call.id, message, error=True))
                elif steps >= max_steps:
                    over_budget = True
                    message = f"Search budget used up. Call `{final_name}` now with what you have."
                    results.append(tool_result(call.id, message, error=True))
                elif call.name in by_name:
                    tool = by_name[call.name]
                    try:
                        arguments = tool.input_model.model_validate(call.input)
                    except ValidationError as error:
                        message = f"Invalid arguments:\n{format_validation_error(error)}"
                        results.append(tool_result(call.id, message, error=True))
                        continue
                    results.append(tool_result(call.id, tool.handler(arguments)))
                    ran_tools = True
                else:
                    results.append(tool_result(call.id, f"Unknown tool '{call.name}'.", error=True))

            if over_budget:
                mistakes += 1
                last_problem = "The model kept calling tools after the budget was used up."
            if ran_tools:
                steps += 1
                if steps >= max_steps:
                    results.append(
                        {
                            "type": "text",
                            "text": f"You have used all {max_steps} rounds of tool calls. Give "
                            f"your final answer now by calling `{final_name}`.",
                        }
                    )
            messages.append({"role": "user", "content": results})

        raise StructuredOutputError(
            f"No valid {schema.__name__} from the tool loop. Last problem:\n{last_problem}"
        )


def tool_result(tool_use_id: str, content: str, error: bool = False) -> ToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": error,
    }
