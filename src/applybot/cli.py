"""Command-line entry point.."""

import tempfile
from pathlib import Path
from typing import Annotated

import anthropic
import typer
from rich.console import Console
from rich.prompt import Prompt

from applybot.agent.state import ReviewDecision
from applybot.latex.compiler import CompilerNotFoundError
from applybot.latex.parser import LatexParseError
from applybot.llm.client import AnthropicClient, LLMConfigError, StructuredOutputError
from applybot.pipeline import run_tailor
from applybot.review.diff import ReviewRequest, render_review
from applybot.review.editor import EditFileError, edits_to_text, open_in_editor, text_to_edits
from applybot.tools.fetch_job import FetchError
from applybot.tools.generate_edits import EditError, EditPlan

app = typer.Typer(help="Tailor a LaTeX CV to a job description.", no_args_is_help=True)
console = Console()

# Failures with a clear message for the user; anything else is a bug and shows a traceback.
EXPECTED_ERRORS = (
    FetchError,
    LLMConfigError,
    StructuredOutputError,
    EditError,
    LatexParseError,
    CompilerNotFoundError,
    anthropic.APIError,
)


def edit_by_hand(plan: EditPlan) -> EditPlan:
    """Write the edits to a text file, open the editor, and read the result back."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(edits_to_text(plan))
        path = Path(f.name)
    try:
        console.print(f"Opening {path} in your editor. Save and close it to continue.")
        open_in_editor(path)
        return text_to_edits(path.read_text(encoding="utf-8"), plan)
    finally:
        path.unlink(missing_ok=True)


def interactive_review(request: ReviewRequest) -> ReviewDecision:
    """Show the diff and ask: approve, edit or reject."""
    console.rule("Review the proposed edits")
    render_review(console, request)
    choices = {"e": "edit", "r": "reject"}
    if request.can_approve:
        choices = {"a": "approve", **choices}
    else:
        console.print("[red]Your edits have errors: edit them again or reject the plan.[/red]")
    labels = ", ".join(f"[{key}]{action[1:]}" for key, action in choices.items())

    while True:
        action = choices[Prompt.ask(labels, choices=list(choices), console=console)]
        if action == "approve":
            return ReviewDecision(action="approve")
        if action == "reject":
            comment = Prompt.ask("What should change? (optional)", default="", console=console)
            return ReviewDecision(action="reject", comment=comment)
        try:
            return ReviewDecision(action="edit", plan=edit_by_hand(request.plan))
        except EditFileError as error:
            console.print(f"[red]{error}[/red]")


@app.callback()
def main() -> None:
    """ApplyBot CLI."""
    # An explicit callback keeps `tailor` as a named subcommand even while it is the only one.


@app.command()
def tailor(
    cv_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Path to the LaTeX CV (main.tex)."),
    ],
    job_url: Annotated[str | None, typer.Option(help="URL of the job posting.")] = None,
    job_file: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="File containing the job description."),
    ] = None,
    job_text: Annotated[str | None, typer.Option(help="Job description pasted as text.")] = None,
    facts: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Optional facts.md with extra true facts."),
    ] = None,
    out_dir: Annotated[Path, typer.Option(help="Where run artifacts are saved.")] = Path("runs"),
) -> None:
    """Tailor CV_PATH to a job description and compile the result."""
    sources = [job_url, job_file, job_text]
    if sum(s is not None for s in sources) != 1:
        raise typer.BadParameter("Pass exactly one of --job-url, --job-file or --job-text.")
    job_source = job_url or (str(job_file) if job_file else None)
    console.print("Reading the job and generating edits...")
    try:
        result = run_tailor(
            cv_path,
            AnthropicClient(),
            interactive_review,
            job_source=job_source,
            job_text=job_text,
            facts_path=facts,
            out_root=out_dir,
        )
    except EXPECTED_ERRORS as error:
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from error

    job = result.job
    console.print(
        f"[bold]{job.title or 'Unknown role'}[/bold] at {job.company or 'unknown company'}: "
        f"{len(result.plan.edits)} edits approved after {len(result.review_history)} review(s)."
    )
    console.print(f"Saved to {result.run_dir}")

    original, tailored = result.original, result.tailored
    if original and tailored:
        if not tailored.success:
            console.print(
                "[red]Tailored CV failed to compile.[/red] See tailored.log in the run folder."
            )
            raise typer.Exit(1)
        console.print(f"Pages: original {original.page_count} -> tailored {tailored.page_count}")
