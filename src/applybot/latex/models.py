"""Data models for a parsed CV."""

from typing import Literal

from pydantic import BaseModel

BlockKind = Literal["summary", "item", "skills_line"]


class Block(BaseModel):
    """A piece of CV text that can be addressed (and maybe edited) by its ID."""

    id: str
    kind: BlockKind
    section: str
    parent_heading: str | None
    text: str
    editable: bool


class Segment(BaseModel):
    """One piece of the document body: either raw LaTeX or a reference to a block."""

    text: str = ""
    block_id: str | None = None


class CVDocument(BaseModel):
    """A CV split into a frozen preamble, body segments and the blocks they reference.

    Joining the preamble with every segment (raw text, or the referenced block's text)
    gives back the original file exactly.
    """

    preamble: str
    segments: list[Segment]
    blocks: list[Block]

    def block(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise KeyError(block_id)
