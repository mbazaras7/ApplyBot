"""One tailoring run, end to end: job -> edits -> tailored .tex -> PDF."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from applybot.latex.compiler import CompileResult, compile_tex
from applybot.latex.parser import parse, slugify
from applybot.llm.client import LLMClient
from applybot.tools.extract_job import JobPosting, extract_job
from applybot.tools.fetch_job import fetch_job
from applybot.tools.generate_edits import EditPlan, apply_edits, generate_edits


class RunResult(BaseModel):
    run_dir: Path
    job: JobPosting
    plan: EditPlan
    original: CompileResult | None
    tailored: CompileResult | None


def make_run_dir(out_root: Path, company: str | None, now: datetime) -> Path:
    """`runs/20261107-153000_accenture/`."""
    name = f"{now:%Y%m%d-%H%M%S}_{slugify(company) if company else 'unknown'}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True)
    return run_dir


def write_text(path: Path, text: str) -> None:
    # Write bytes so line endings stay exactly as they are in the text.
    path.write_bytes(text.encode("utf-8"))


def run_tailor(
    cv_path: Path,
    llm: LLMClient,
    *,
    job_source: str | None = None,
    job_text: str | None = None,
    out_root: Path = Path("runs"),
    now: datetime | None = None,
    compile_pdf: bool = True,
) -> RunResult:
    """Tailor the CV at `cv_path` to a job given as a source (URL/file/-) or as text."""
    if (job_source is None) == (job_text is None):
        raise ValueError("Pass exactly one of job_source or job_text.")
    text = job_text.strip() if job_text is not None else fetch_job(str(job_source))

    job = extract_job(text, llm)
    original_tex = cv_path.read_bytes().decode("utf-8")
    doc = parse(original_tex)
    plan = generate_edits(doc, job, llm)
    tailored_tex = apply_edits(doc, plan)

    run_dir = make_run_dir(out_root, job.company, now or datetime.now())
    write_text(run_dir / "original.tex", original_tex)
    write_text(run_dir / "job.json", job.model_dump_json(indent=2))
    write_text(run_dir / "edits.json", plan.model_dump_json(indent=2))
    write_text(run_dir / "tailored.tex", tailored_tex)

    original = tailored = None
    if compile_pdf:
        original = compile_tex(run_dir / "original.tex")
        tailored = compile_tex(run_dir / "tailored.tex")
    return RunResult(run_dir=run_dir, job=job, plan=plan, original=original, tailored=tailored)
