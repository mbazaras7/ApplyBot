"""The CV index: what gets indexed, and searching by similarity."""

from pathlib import Path

import pytest
from fakes import FakeEmbedder

from applybot.latex.parser import parse
from applybot.tools.cv_index import (
    CACHE_DIR,
    MODEL_NAME,
    Chunk,
    CVIndex,
    FastEmbedder,
    SearchHit,
    chunks_of,
    format_search_results,
    plain,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"


@pytest.fixture
def index() -> CVIndex:
    return CVIndex.build(parse(SAMPLE.read_bytes().decode("utf-8")), FakeEmbedder())


def test_plain_strips_latex() -> None:
    assert (
        plain("\\textbf{TaskFlow} $|$ \\emph{TypeScript, React}") == "TaskFlow | TypeScript, React"
    )
    assert plain("99.9\\% uptime, R\\&D") == "99.9% uptime, R&D"


def test_blocks_and_headings_are_indexed() -> None:
    chunks = {c.id: c for c in chunks_of(parse(SAMPLE.read_bytes().decode("utf-8")))}

    assert chunks["exp.acme-corp.1"].kind == "block"
    assert chunks["exp.acme-corp.1"].editable
    assert not chunks["edu.springfield-university.1"].editable  # read-only blocks too
    degree = chunks["edu.springfield-university"]
    assert degree.kind == "heading"
    assert "Bachelor of Science in Computer Science" in degree.text  # not in any block
    assert "TypeScript, React, PostgreSQL" in chunks["proj.taskflow"].text  # project stack


def test_search_ranks_by_similarity(index: CVIndex) -> None:
    hits = index.search("billing service integration tests", FakeEmbedder(), k=2)

    assert len(hits) == 2
    assert hits[0].chunk.id == "exp.globex.1"
    assert hits[0].score >= hits[1].score


def test_search_respects_k(index: CVIndex) -> None:
    assert len(index.search("Python", FakeEmbedder(), k=5)) == 5


def test_index_survives_json(index: CVIndex) -> None:
    # The index lives in the graph state, which the checkpointer saves.
    assert CVIndex.model_validate_json(index.model_dump_json()) == index


# --- format_search_results -----------------------------------------------------------------

BULLET = Chunk(
    id="exp.beeline.3", kind="block", section="Experience", heading="Beeline",
    text="Shipped changes through GitLab CI/CD pipelines.", editable=True,
)  # fmt: skip
DEGREE = Chunk(
    id="edu.dcu", kind="heading", section="Education", heading=None,
    text="Dublin City University | BSc Computer Science", editable=False,
)  # fmt: skip


def test_format_search_results() -> None:
    hits = [SearchHit(chunk=BULLET, score=0.82), SearchHit(chunk=DEGREE, score=0.41)]
    assert format_search_results(hits) == (
        "[exp.beeline.3] Experience > Beeline (score 0.82): Shipped changes through GitLab "
        "CI/CD pipelines.\n"
        "[edu.dcu] Education (score 0.41): Dublin City University | BSc Computer Science"
    )


def test_format_search_results_empty() -> None:
    assert format_search_results([]) == "No matching CV content."


# --- The real model (only when already downloaded) -----------------------------------------


@pytest.mark.skipif(
    not any(CACHE_DIR.glob("*bge-small-en-v1.5*")),
    reason=f"{MODEL_NAME} not downloaded (run the CLI once to fetch it)",
)
def test_real_embeddings_find_synonyms() -> None:
    embedder = FastEmbedder()
    index = CVIndex.build(parse(SAMPLE.read_bytes().decode("utf-8")), embedder)
    # No word overlap with "GitHub Actions", but the meaning matches.
    hits = index.search("continuous integration tooling", embedder, k=3)
    assert "skills.tools-platforms" in [hit.chunk.id for hit in hits]
