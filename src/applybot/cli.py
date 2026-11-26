"""Command-line entry point.."""

import tempfile
from pathlib import Path
from typing import Annotated

import anthropic
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from applybot.agent.state import ReviewDecision
from applybot.latex.compiler import CompilerNotFoundError
from applybot.latex.parser import LatexParseError
from applybot.llm.client import AnthropicClient, LLMConfigError, StructuredOutputError
from applybot.pipeline import run_tailor
from applybot.review.diff import ReviewRequest, render_review
from applybot.review.editor import EditFileError, edits_to_text, open_in_editor, text_to_edits
from applybot.storage.history import open_history
from applybot.storage.models import Run, RunSummary
from applybot.storage.paths import HISTORY_DB, data_dir
from applybot.tools.fetch_job import FetchError
from applybot.tools.generate_edits import EditError, EditPlan

app = typer.Typer(help="Tailor a LaTeX CV to a job description.", no_args_is_help=True)
history_app = typer.Typer(help="Browse past runs.")
app.add_typer(history_app, name="history")
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
    # escape() stops Rich from reading "[a]" as a formatting tag and hiding it.
    labels = ", ".join(escape(f"[{key}]") + action[1:] for key, action in choices.items())

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
    prompt: Annotated[
        str, typer.Option(help="Your own instructions, added below the built-in rules.")
    ] = "",
    out_dir: Annotated[Path, typer.Option(help="Where run artifacts are saved.")] = Path("runs"),
) -> None:
    """Tailor CV_PATH to a job description and compile the result."""
    sources = [job_url, job_file, job_text]
    if sum(s is not None for s in sources) != 1:
        raise typer.BadParameter("Pass exactly one of --job-url, --job-file or --job-text.")
    job_source = job_url or (str(job_file) if job_file else None)
    console.print("Reading the job, searching your CV for evidence and generating edits...")
    try:
        result = run_tailor(
            cv_path,
            AnthropicClient(),
            interactive_review,
            job_source=job_source,
            job_text=job_text,
            facts_path=facts,
            custom_prompt=prompt,
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
    console.print(f"Fit {result.analysis.fit_score}/100, {len(result.analysis.gaps)} gap(s).")
    console.print(f"Saved to {result.run_dir} (summary: {result.report_path.name})")
    if result.run_id is not None:
        console.print(
            f"Recorded as run {result.run_id} (see: applybot history show {result.run_id})"
        )

    original, tailored = result.original, result.tailored
    if original and tailored:
        if not tailored.success:
            console.print(
                "[red]Tailored CV failed to compile.[/red] See tailored.log in the run folder."
            )
            raise typer.Exit(1)
        console.print(f"Pages: original {original.page_count} -> tailored {tailored.page_count}")


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 to listen on.")] = 8765,
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Where run folders are saved (default: in the data folder)."),
    ] = None,
) -> None:
    """Start the local API that the desktop app talks to."""
    # Imported here so the other commands don't pay for loading the web server.
    import uvicorn

    from applybot.api.app import create_app
    from applybot.service.runs import RunManager

    folder = data_dir()
    manager = RunManager(folder, AnthropicClient, out_root=out_dir or folder / "runs")
    console.print(f"ApplyBot API on http://127.0.0.1:{port} (docs at /docs). Ctrl+C to stop.")
    # 127.0.0.1 only: the API is for programs on this computer, never the network.
    uvicorn.run(create_app(manager), host="127.0.0.1", port=port)


def local_time(run: Run | RunSummary) -> str:
    return f"{run.created_at.astimezone():%Y-%m-%d %H:%M}"


@history_app.callback(invoke_without_command=True)
def history(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option(help="How many runs to show.")] = 20,
) -> None:
    """List past runs, newest first."""
    if ctx.invoked_subcommand is not None:  # `history show ...` runs instead
        return
    store = open_history()
    try:
        runs = store.list_runs(limit)
    finally:
        store.close()
    if not runs:
        console.print(f"No runs yet. (History: {data_dir() / HISTORY_DB})")
        return

    table = Table("ID", "Date", "Company", "Role", "Fit", "Status")
    for run in runs:
        table.add_row(
            str(run.id),
            local_time(run),
            escape(run.company or "unknown"),
            escape(run.title or "unknown"),
            "" if run.fit_score is None else str(run.fit_score),
            run.status.value.replace("_", " "),
        )
    console.print(table)


@history_app.command("show")
def history_show(run_id: Annotated[int, typer.Argument(help="The run's ID.")]) -> None:
    """Show one run: its details and report."""
    store = open_history()
    try:
        run = store.get_run(run_id)
    finally:
        store.close()
    if run is None:
        console.print(f"[red]Error:[/red] no run {run_id}.")
        raise typer.Exit(1)

    console.print(
        f"[bold]Run {run.id}[/bold]: {escape(run.title or 'Unknown role')} at "
        f"{escape(run.company or 'unknown company')}"
    )
    console.print(f"Started {local_time(run)}, status: {run.status.value.replace('_', ' ')}")
    if run.fit_score is not None:
        console.print(f"Fit {run.fit_score}/100")
    if run.error:
        console.print(f"[red]Error:[/red] {escape(run.error)}")
    pdf = f"{len(run.tailored_pdf) // 1024} KB" if run.tailored_pdf else "none"
    console.print(f"Folder: {run.run_dir}  (tailored PDF stored: {pdf})")
    if run.report_md:
        console.rule("Report")
        console.print(Markdown(run.report_md))
