"""One tailoring run, end to end: job -> validated edits -> tailored .tex -> PDF."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from applybot.latex.compiler import CompileResult, compile_tex
from applybot.latex.parser import parse, slugify
from applybot.llm.client import LLMClient
from applybot.tools.extract_job import JobPosting, extract_job
from applybot.tools.fetch_job import fetch_job
from applybot.tools.generate_edits import EditPlan, apply_edits
from applybot.tools.validate import (
    ValidatedPlan,
    ValidationContext,
    compile_text,
    generate_valid_plan,
)


class RunResult(BaseModel):
    run_dir: Path
    job: JobPosting
    validation: ValidatedPlan
    original: CompileResult | None
    tailored: CompileResult | None

    @property
    def plan(self) -> EditPlan:
        return self.validation.plan


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
    facts_path: Path | None = None,
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

    run_dir = make_run_dir(out_root, job.company, now or datetime.now())
    write_text(run_dir / "original.tex", original_tex)
    write_text(run_dir / "job.json", job.model_dump_json(indent=2))

    # Compile the original first: its page count is the limit the tailored CV must meet.
    original = compile_tex(run_dir / "original.tex") if compile_pdf else None
    ctx = ValidationContext(
        doc=doc,
        facts=facts_path.read_text(encoding="utf-8") if facts_path else "",
        job_text=job.raw_text,
        original_pages=original.page_count if original else None,
    )
    validation = generate_valid_plan(
        ctx, job, llm, compile_fn=compile_text if compile_pdf else None
    )
    tailored_tex = apply_edits(doc, validation.plan)

    write_text(run_dir / "edits.json", validation.plan.model_dump_json(indent=2))
    write_text(run_dir / "validation.json", validation.model_dump_json(indent=2))
    write_text(run_dir / "tailored.tex", tailored_tex)

    tailored = compile_tex(run_dir / "tailored.tex") if compile_pdf else None
    return RunResult(
        run_dir=run_dir, job=job, validation=validation, original=original, tailored=tailored
    )
