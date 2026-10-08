"""Turn a CVDocument back into LaTeX, optionally with new text for some blocks."""

from collections.abc import Mapping

from applybot.latex.models import CVDocument


def render(doc: CVDocument, new_texts: Mapping[str, str] | None = None) -> str:
    """Rebuild the LaTeX source.

    `new_texts` maps block IDs to replacement text. Everything else (preamble, headings,
    commands, whitespace) is copied unchanged, so with no edits the output is identical
    to the parsed input.
    """
    new_texts = new_texts or {}
    texts = {block.id: block.text for block in doc.blocks}
    unknown = set(new_texts) - set(texts)
    if unknown:
        raise KeyError(f"Unknown block IDs: {sorted(unknown)}")
    texts.update(new_texts)

    parts = [doc.preamble]
    for segment in doc.segments:
        parts.append(segment.text if segment.block_id is None else texts[segment.block_id])
    return "".join(parts)
