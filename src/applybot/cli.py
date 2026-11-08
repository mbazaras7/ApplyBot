"""Command-line entry point.."""

from pathlib import Path
from typing import Annotated

import anthropic
import typer
from rich.console import Console
from rich.table import Table

from applybot.latex.compiler import CompilerNotFoundError
from applybot.latex.parser import LatexParseError
from applybot.llm.client import AnthropicClient, LLMConfigError, StructuredOutputError
from applybot.pipeline import run_tailor
from applybot.tools.fetch_job import FetchError
from applybot.tools.generate_edits import EditError

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
    try:
        with console.status("Reading the job, generating and validating edits, compiling..."):
            result = run_tailor(
                cv_path,
                AnthropicClient(),
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
        f"[bold]{job.title or 'Unknown role'}[/bold] at {job.company or 'unknown company'}"
    )
    table = Table(title=f"{len(result.plan.edits)} edits")
    table.add_column("block")
    table.add_column("rationale")
    for edit in result.plan.edits:
        table.add_row(edit.block_id, edit.rationale)
    console.print(table)

    validation = result.validation
    console.print(f"Validation passed after {validation.attempts} attempt(s).")
    for dropped in validation.dropped:
        reasons = "; ".join(dropped.reasons)
        console.print(f"[red]Dropped[/red] {dropped.edit.block_id}: {reasons}")
    for warning in validation.warnings:
        console.print(f"[yellow]Warning[/yellow] {warning}")
    console.print(f"Saved to {result.run_dir}")

    original, tailored = result.original, result.tailored
    if original and tailored:
        if not tailored.success:
            console.print(
                "[red]Tailored CV failed to compile.[/red] See tailored.log in the run folder."
            )
            raise typer.Exit(1)
        console.print(f"Pages: original {original.page_count} -> tailored {tailored.page_count}")
