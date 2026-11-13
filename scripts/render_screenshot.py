"""Render the review screen to assets/review-screen.svg for the README.

Uses the real `render_review` code with the fictional sample CV, so the image matches
the app without showing anyone's personal details. Re-run after changing the review UI:

    uv run python scripts/render_screenshot.py
"""

from pathlib import Path

from rich.console import Console
from rich.text import Text

from applybot.latex.parser import parse
from applybot.review.diff import ReviewRequest, build_diff, render_review
from applybot.tools.analyse import JobAnalysis
from applybot.tools.generate_edits import Edit, EditPlan
from applybot.tools.validate import Problem

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "fixtures" / "sample_cv.tex"
OUTPUT = ROOT / "assets" / "review-screen.svg"

doc = parse(SAMPLE.read_bytes().decode("utf-8"))

analysis = JobAnalysis.model_validate(
    {
        "findings": [
            {
                "requirement": "Python backend services",
                "kind": "strength",
                "evidence": ["exp.acme-corp.1"],
                "note": "Built a Python ingestion service.",
            },
            {
                "requirement": "Performance and caching",
                "kind": "rewrite_target",
                "evidence": ["exp.acme-corp.2"],
                "note": "Redis caching work, worded around latency.",
            },
            {
                "requirement": "Kubernetes",
                "kind": "gap",
                "evidence": [],
                "note": "No container orchestration experience in the CV.",
            },
        ],
        "fit_score": 78,
        "fit_reasoning": (
            "Strong match on Python services, APIs and caching, backed by a full-time role. "
            "The main gap is Kubernetes, which the CV doesn't show."
        ),
    }
)

plan = EditPlan(
    edits=[
        Edit(
            block_id="summary.1",
            new_text=(
                "Backend-focused software engineer with experience in Python, Go and "
                "TypeScript, building reliable APIs and developer tooling."
            ),
            rationale="Leads with backend and Python, the role's core requirements.",
        ),
        Edit(
            block_id="exp.acme-corp.2",
            new_text=(
                "Cut API latency by 35\\% with Redis caching on the hottest "
                "\\texttt{GET} endpoints."
            ),
            rationale="Makes the performance and caching match obvious.",
        ),
        Edit(
            block_id="skills.languages",
            new_text="Python, Go, SQL, TypeScript",
            rationale="Puts the job's languages first.",
        ),
    ]
)
problems: list[Problem] = []  # these edits pass every validator

request = ReviewRequest(
    plan=plan,
    entries=build_diff(doc, plan, problems),
    plan_problems=[],
    dropped=[],
    attempts=1,
    can_approve=True,
    analysis=analysis,
)

console = Console(record=True, width=96, force_terminal=True, color_system="truecolor")
console.rule("Review the proposed edits")
render_review(console, request)
console.print(Text("[a]pprove, [e]dit, [r]eject [a/e/r]: ", style="bold"))

OUTPUT.parent.mkdir(exist_ok=True)
console.save_svg(str(OUTPUT), title="applybot tailor")
print(f"Wrote {OUTPUT.relative_to(ROOT)}")
