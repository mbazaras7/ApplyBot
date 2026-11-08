"""Search the CV by meaning: embed every block and heading, then find the closest ones.

An embedding model turns text into a vector (a list of numbers) so that texts with
similar meaning get vectors pointing in similar directions. "deployment pipelines" ends
up close to "GitLab CI/CD", even though they share no words.
"""

import re
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel

from applybot.latex.models import CVDocument

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small (384 numbers per text), runs locally
CACHE_DIR = Path.home() / ".cache" / "applybot" / "fastembed"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """`Embedder` using fastembed. The model downloads once (~65 MB) on first use."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = CACHE_DIR) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None

    def model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # imported here: loading it is slow

            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models embed questions slightly differently from the passages they search.
        return [float(x) for x in next(iter(self.model().query_embed(text)))]


class Chunk(BaseModel):
    """One searchable piece of the CV: a block, or a heading's arguments."""

    id: str  # a block ID ("exp.beeline.2") or heading ID ("edu.dublin-city-university")
    kind: Literal["block", "heading"]
    section: str
    heading: str | None
    text: str  # plain text, LaTeX removed
    editable: bool


class SearchHit(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity: 1 = same direction, 0 = unrelated


def plain(latex: str) -> str:
    """`\\textbf{ALEXIDE} $|$ \\emph{React}` -> `ALEXIDE | React`."""
    text = re.sub(r"\\[A-Za-z]+\*?", " ", latex).replace("$", "")
    text = re.sub(r"\\(.)", r"\1", text).replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text).strip()


def chunks_of(doc: CVDocument) -> list[Chunk]:
    chunks = [
        Chunk(
            id=block.id,
            kind="block",
            section=plain(block.section),
            heading=plain(block.parent_heading) if block.parent_heading else None,
            text=plain(block.text),
            editable=block.editable,
        )
        for block in doc.blocks
    ]
    chunks += [
        Chunk(
            id=heading.id,
            kind="heading",
            section=plain(heading.section),
            heading=plain(heading.name),
            text=" | ".join(plain(arg) for arg in heading.args),
            editable=False,
        )
        for heading in doc.headings
    ]
    return chunks


class CVIndex(BaseModel):
    """The CV's chunks and their vectors, kept as plain lists so they can be saved."""

    chunks: list[Chunk]
    vectors: list[list[float]]

    @classmethod
    def build(cls, doc: CVDocument, embedder: Embedder) -> "CVIndex":
        chunks = chunks_of(doc)
        # Embed with context ("Experience > Beeline: ...") so a bullet knows where it is.
        texts = [f"{c.section} > {c.heading}: {c.text}" if c.heading else c.text for c in chunks]
        return cls(chunks=chunks, vectors=embedder.embed_documents(texts))

    def search(self, query: str, embedder: Embedder, k: int = 3) -> list[SearchHit]:
        """The `k` chunks whose meaning is closest to `query`, best first."""
        matrix = np.array(self.vectors)
        q = np.array(embedder.embed_query(query))
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        scores = matrix @ q / np.where(norms == 0, 1, norms)  # cosine similarity
        best = np.argsort(-scores)[:k]
        return [SearchHit(chunk=self.chunks[i], score=round(float(scores[i]), 3)) for i in best]


def format_search_results(hits: list[SearchHit]) -> str:
    """Turn search hits into the text the model reads as the `search_cv` tool's result.

    One line per hit, best first:

        [exp.beeline.3] Experience > Beeline (score 0.82): Participated in Agile/Scrum ...

    ` > <heading>` is left out when a chunk has no heading.
    """
    if len(hits) == 0:
        return "No matching CV content."
    lines = []
    for hit in sorted(hits, key=lambda p: p.score, reverse=True):
        where = hit.chunk.section
        if hit.chunk.heading is not None:
            where += f" > {hit.chunk.heading}"
        lines.append(f"[{hit.chunk.id}] {where} (score {hit.score}): {hit.chunk.text}")
    return "\n".join(lines)
