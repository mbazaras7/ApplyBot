"""Compile a .tex file to PDF with Tectonic (or latexmk as a fallback)."""

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

TIMEOUT_SECONDS = 180

# Jake's Resume uses two pdfTeX-only commands (\pdfglyphtounicode via
# \input{glyphtounicode}, and \pdfgentounicode). Tectonic runs XeTeX, where they don't
# exist. This wrapper defines no-op stand-ins and then loads the CV unchanged, so the
# CV's preamble is never edited. Under pdfLaTeX (latexmk) the stand-ins do nothing.
WRAPPER = """\
\\providecommand{{\\pdfglyphtounicode}}[2]{{}}
\\ifdefined\\pdfgentounicode\\else\\newcount\\pdfgentounicode\\fi
\\input{{{name}}}
"""


class CompilerNotFoundError(RuntimeError):
    """Neither Tectonic nor latexmk is installed."""


class CompileResult(BaseModel):
    success: bool
    log: str
    pdf_path: Path | None = None
    page_count: int | None = None


def _command(tex_name: str, out_dir: Path) -> list[str]:
    tectonic = shutil.which("tectonic")
    if tectonic:
        return [tectonic, "--keep-logs", "--outdir", str(out_dir), tex_name]
    latexmk = shutil.which("latexmk")
    if latexmk:
        return [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={out_dir}",
            tex_name,
        ]
    raise CompilerNotFoundError("Install Tectonic (preferred) or latexmk to compile PDFs.")


def compile_tex(tex_path: Path, out_dir: Path | None = None) -> CompileResult:
    """Compile `tex_path` into `out_dir/<stem>.pdf` (default: the .tex file's folder)."""
    tex_path = tex_path.resolve()
    out_dir = (out_dir or tex_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = tex_path.with_name(f"_build_{tex_path.stem}.tex")
    wrapper.write_text(WRAPPER.format(name=tex_path.name), encoding="utf-8")
    build_pdf = out_dir / f"{wrapper.stem}.pdf"
    build_log = out_dir / f"{wrapper.stem}.log"
    pdf_path = out_dir / f"{tex_path.stem}.pdf"
    for stale in (build_pdf, build_log, pdf_path):
        stale.unlink(missing_ok=True)  # never report output from an earlier run

    try:
        proc = subprocess.run(
            _command(wrapper.name, out_dir),
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, log=f"Timed out after {TIMEOUT_SECONDS}s")
    finally:
        wrapper.unlink(missing_ok=True)

    log = proc.stdout + proc.stderr
    if build_log.exists():
        log += "\n" + build_log.read_text(encoding="utf-8", errors="replace")
        build_log.replace(out_dir / f"{tex_path.stem}.log")

    if proc.returncode != 0 or not build_pdf.exists():
        return CompileResult(success=False, log=log)

    build_pdf.replace(pdf_path)
    page_count = len(PdfReader(pdf_path).pages)
    return CompileResult(success=True, log=log, pdf_path=pdf_path, page_count=page_count)
