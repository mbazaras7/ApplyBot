"""The tailoring workflow as a LangGraph state graph.

    START -> load_cv -> fetch_job -> extract_job -> start_run -> generate_edits -> validate
    validate --route_after_validate--> generate_edits | drop_failing | render_diff
    drop_failing -> render_diff -> human_review (interrupt: waits for the reviewer)
    human_review --route_after_review--> compile | recheck | generate_edits
    recheck -> render_diff ;  compile -> END

Nodes are plain functions: they read the state and return only the keys they change.
Routing functions only look at the state and name the next node; they never change it.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from applybot.agent.state import ReviewDecision, ReviewRound, TailorState
from applybot.latex.compiler import CompileResult, compile_tex
from applybot.latex.models import Block, CVDocument, Segment
from applybot.latex.parser import parse, slugify
from applybot.llm.client import LLMClient
from applybot.review.diff import DiffEntry, ReviewRequest, build_diff
from applybot.tools.extract_job import JobPosting, extract_job
from applybot.tools.fetch_job import fetch_job
from applybot.tools.generate_edits import Edit, EditPlan, apply_edits, generate_edits
from applybot.tools.validate import (
    MAX_RETRIES,
    DroppedEdit,
    Problem,
    ValidatedPlan,
    ValidationContext,
    compile_text,
    drop_failing_edits,
    errors_of,
    format_problems,
    validate,
)

# The checkpointer stores the state between steps. Our state holds Pydantic objects, so
# their classes are listed here to be allowed back in when a paused run is resumed.
CHECKPOINT_TYPES = (
    CVDocument, Block, Segment, JobPosting, Edit, EditPlan, Problem, DroppedEdit,
    CompileResult, ReviewDecision, ReviewRound, ReviewRequest, DiffEntry,
)  # fmt: skip
# Each review round is a few graph steps; LangGraph's default limit of 25 is too low
# for several rounds of edit / reject.
RECURSION_LIMIT = 200


def make_run_dir(out_root: Path, company: str | None, started: str) -> Path:
    """`runs/20261107-153000_accenture/`."""
    run_dir = out_root / f"{started}_{slugify(company) if company else 'unknown'}"
    run_dir.mkdir(parents=True)
    return run_dir


def write_text(path: Path, text: str) -> None:
    # Write bytes so line endings stay exactly as they are in the text.
    path.write_bytes(text.encode("utf-8"))


# --- Routing -------------------------------------------------------------------------------


def route_after_validate(
    state: TailorState,
) -> Literal["render_diff", "generate_edits", "drop_failing"]:
    """After validation: show the plan, retry with the errors, or drop what still fails."""
    if not errors_of(state["problems"]):
        return "render_diff"
    if state["attempts"] <= MAX_RETRIES:  # attempts so far: first try + retries
        return "generate_edits"
    return "drop_failing"


def route_after_review(state: TailorState) -> Literal["compile", "recheck", "generate_edits"]:
    """After the reviewer answers: decide which node runs next."""
    action = state["decision"].action
    if action == "approve":
        return "compile"
    if action == "edit":
        return "recheck"
    return "generate_edits"


# --- Nodes ---------------------------------------------------------------------------------


class Nodes:
    """The graph's steps. Held in a class so they share the LLM and the compile setting."""

    def __init__(self, llm: LLMClient, compile_pdf: bool) -> None:
        self.llm = llm
        self.compile_pdf = compile_pdf

    def context(self, state: TailorState) -> ValidationContext:
        original = state.get("original")
        return ValidationContext(
            doc=state["doc"],
            facts=state.get("facts", ""),
            job_text=state["job"].raw_text,
            original_pages=original.page_count if original else None,
        )

    def compile_fn(self) -> Any:
        return compile_text if self.compile_pdf else None

    def load_cv(self, state: TailorState) -> TailorState:
        original_tex = Path(state["cv_path"]).read_bytes().decode("utf-8")
        facts_path = state.get("facts_path")
        facts = Path(facts_path).read_text(encoding="utf-8") if facts_path else ""
        return {"original_tex": original_tex, "doc": parse(original_tex), "facts": facts}

    def fetch_job(self, state: TailorState) -> TailorState:
        job_text = state.get("job_text")
        text = job_text.strip() if job_text is not None else fetch_job(str(state["job_source"]))
        return {"job_text": text}

    def extract_job(self, state: TailorState) -> TailorState:
        return {"job": extract_job(str(state["job_text"]), self.llm)}

    def start_run(self, state: TailorState) -> TailorState:
        """Create the run folder, save the inputs, and compile the original for its page count."""
        run_dir = make_run_dir(Path(state["out_root"]), state["job"].company, state["started"])
        write_text(run_dir / "original.tex", state["original_tex"])
        write_text(run_dir / "job.json", state["job"].model_dump_json(indent=2))
        original = compile_tex(run_dir / "original.tex") if self.compile_pdf else None
        return {"run_dir": str(run_dir), "original": original, "attempts": 0}

    def generate_edits(self, state: TailorState) -> TailorState:
        comment = state.get("reviewer_comment")
        errors = state.get("retry_errors", "")
        previous = state.get("plan") if comment is not None or errors else None
        plan = generate_edits(
            state["doc"],
            state["job"],
            self.llm,
            previous_plan=previous,
            errors=errors,
            reviewer_comment=comment,
        )
        return {
            "plan": plan,
            "attempts": state.get("attempts", 0) + 1,
            "retry_errors": "",
            "reviewer_comment": None,
        }

    def validate(self, state: TailorState) -> TailorState:
        problems = validate(self.context(state), state["plan"], self.compile_fn())
        return {
            "problems": problems,
            "retry_errors": format_problems(errors_of(problems)),
            "review_errors": [],
        }

    def drop_failing(self, state: TailorState) -> TailorState:
        plan, dropped, problems = drop_failing_edits(
            self.context(state), state["plan"], state["problems"], self.compile_fn()
        )
        return {"plan": plan, "dropped": dropped, "problems": problems}

    def render_diff(self, state: TailorState) -> TailorState:
        problems = state["problems"]
        request = ReviewRequest(
            plan=state["plan"],
            entries=build_diff(state["doc"], state["plan"], problems),
            plan_problems=[p for p in problems if p.block_id is None],
            dropped=state.get("dropped", []),
            attempts=state["attempts"],
            can_approve=not state.get("review_errors"),
        )
        return {"review_request": request}

    def human_review(self, state: TailorState) -> TailorState:
        # interrupt() pauses the graph here. When it's resumed, this node runs again from
        # the top and interrupt() returns the reviewer's answer, so nothing before this
        # line may have side effects.
        decision = ReviewDecision.model_validate(interrupt(state["review_request"]))
        if decision.action == "approve" and state.get("review_errors"):
            raise ValueError("Can't approve while the hand edits fail validation.")

        edited: list[str] = []
        if decision.action == "edit" and decision.plan is not None:
            before = {e.block_id: e.new_text for e in state["plan"].edits}
            edited = [
                e.block_id for e in decision.plan.edits if before.get(e.block_id) != e.new_text
            ]
        round_ = ReviewRound(action=decision.action, comment=decision.comment, edited_blocks=edited)
        update: TailorState = {
            "decision": decision,
            "review_history": [*state.get("review_history", []), round_],
        }
        if decision.action == "reject":
            # Start a fresh round of generation, with the reviewer's comment as feedback.
            update["reviewer_comment"] = decision.comment
            update["attempts"] = 0
            update["dropped"] = []
        return update

    def recheck(self, state: TailorState) -> TailorState:
        """Validate the reviewer's hand-edited plan (no retries: a person wrote it)."""
        plan = state["decision"].plan or state["plan"]
        problems = validate(self.context(state), plan, self.compile_fn())
        return {"plan": plan, "problems": problems, "review_errors": errors_of(problems)}

    def compile(self, state: TailorState) -> TailorState:
        """Only reachable after approval: write the tailored CV and compile it."""
        run_dir = Path(state["run_dir"])
        plan = state["plan"]
        warnings = [p for p in state["problems"] if p.severity == "warning"]
        validation = ValidatedPlan(
            plan=plan,
            attempts=state["attempts"],
            dropped=state.get("dropped", []),
            warnings=warnings,
        )
        history = [r.model_dump() for r in state.get("review_history", [])]
        write_text(run_dir / "edits.json", plan.model_dump_json(indent=2))
        write_text(run_dir / "validation.json", validation.model_dump_json(indent=2))
        write_text(run_dir / "review.json", json.dumps(history, indent=2))
        write_text(run_dir / "tailored.tex", apply_edits(state["doc"], plan))
        tailored = compile_tex(run_dir / "tailored.tex") if self.compile_pdf else None
        return {"tailored": tailored}


def build_graph(llm: LLMClient, compile_pdf: bool = True) -> Any:
    nodes = Nodes(llm, compile_pdf)
    graph = StateGraph(TailorState)
    for name in (
        "load_cv", "fetch_job", "extract_job", "start_run", "generate_edits", "validate",
        "drop_failing", "render_diff", "human_review", "recheck", "compile",
    ):  # fmt: skip
        graph.add_node(name, getattr(nodes, name))

    graph.add_edge(START, "load_cv")
    graph.add_edge("load_cv", "fetch_job")
    graph.add_edge("fetch_job", "extract_job")
    graph.add_edge("extract_job", "start_run")
    graph.add_edge("start_run", "generate_edits")
    graph.add_edge("generate_edits", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, ["render_diff", "generate_edits", "drop_failing"]
    )
    graph.add_edge("drop_failing", "render_diff")
    graph.add_edge("render_diff", "human_review")
    graph.add_conditional_edges(
        "human_review", route_after_review, ["compile", "recheck", "generate_edits"]
    )
    graph.add_edge("recheck", "render_diff")
    graph.add_edge("compile", END)

    allowed = [(cls.__module__, cls.__name__) for cls in CHECKPOINT_TYPES]
    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
    return graph.compile(checkpointer=InMemorySaver(serde=serde))


def start_timestamp(now: datetime | None = None) -> str:
    return f"{now or datetime.now():%Y%m%d-%H%M%S}"
