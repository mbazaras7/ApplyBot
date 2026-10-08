"""Command-line entry point.."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Tailor a LaTeX CV to a job description.", no_args_is_help=True)
console = Console()


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
    """Tailor CV_PATH to a job description (stub: prints its arguments)."""
    sources = [job_url, job_file, job_text]
    if sum(s is not None for s in sources) != 1:
        raise typer.BadParameter("Pass exactly one of --job-url, --job-file or --job-text.")

    table = Table(title="tailor arguments")
    table.add_column("argument")
    table.add_column("value")
    for name, value in [
        ("cv_path", cv_path),
        ("job_url", job_url),
        ("job_file", job_file),
        ("job_text", job_text),
        ("facts", facts),
        ("out_dir", out_dir),
    ]:
        table.add_row(name, str(value))
    console.print(table)
