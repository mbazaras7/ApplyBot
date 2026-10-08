"""Parse a Jake's Resume LaTeX CV into a CVDocument.

The parser is a small hand-written scanner. It walks the text one character at a time
and counts brace depth, so it can find where a `{...}` group really ends even when the
group contains nested groups (`\\textbf{...}`) or escaped characters (`\\}`, `\\%`).
"""

import re

from applybot.latex.models import Block, BlockKind, CVDocument, Segment

BEGIN_DOCUMENT = "\\begin{document}"

# First word of a section title -> short key used in block IDs.
SECTION_KEYS = {
    "summary": "summary",
    "experience": "exp",
    "projects": "proj",
    "education": "edu",
    "skills": "skills",
}
EDITABLE_SECTIONS = {"summary", "exp", "proj", "skills"}


class LatexParseError(ValueError):
    """The LaTeX could not be parsed (e.g. unbalanced braces or duplicate block IDs)."""


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# --- Scanner -----------------------------------------------------------------------------


def skip_comment(text: str, pos: int) -> int:
    """`pos` is at an unescaped `%`. Return the position of the end of that line."""
    end = text.find("\n", pos)
    return len(text) if end == -1 else end


def skip_space(text: str, pos: int) -> int:
    """Skip whitespace and comments, returning the next meaningful position."""
    while pos < len(text):
        if text[pos] == "%":
            pos = skip_comment(text, pos)
        elif text[pos].isspace():
            pos += 1
        else:
            break
    return pos


def read_command_name(text: str, pos: int) -> tuple[str, int]:
    """`pos` is at a backslash. Return the command name and the position after it.

    `\\resumeItem` -> "resumeItem". Commands made of one symbol (`\\&`, `\\%`, `\\\\`)
    return that symbol, which is how escaped characters get consumed safely.
    """
    end = pos + 1
    while end < len(text) and text[end].isalpha():
        end += 1
    if end == pos + 1:  # control symbol such as \& or \\
        end = min(pos + 2, len(text))
    return text[pos + 1 : end], end


def read_group(text: str, pos: int) -> tuple[str, int]:
    """Read the brace group starting at `pos` (which must be `{`).

    Returns the content between the outer braces and the position just after the
    closing brace.
    """
    if pos >= len(text) or text[pos] != "{":
        raise LatexParseError(f"Expected '{{' on line {line_of(text, pos)}")
    depth = 0
    i = pos
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2  # skip the escaped character: \{ \} \% \& \\ never change depth
            continue
        if char == "%":
            i = skip_comment(text, i)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    raise LatexParseError(f"Unbalanced braces: '{{' on line {line_of(text, pos)} is never closed")


def read_args(text: str, pos: int, count: int) -> tuple[list[str], int]:
    """Read `count` brace groups, allowing whitespace/comments between them."""
    args = []
    for _ in range(count):
        content, pos = read_group(text, skip_space(text, pos))
        args.append(content)
    return args, pos


# --- Naming helpers ----------------------------------------------------------------------


def slugify(latex: str) -> str:
    """Turn LaTeX text into an ID-friendly slug: `Web \\& Backend:` -> `web-backend`."""
    text = re.sub(r"\\[A-Za-z]+\*?", " ", latex)  # drop command names, keep their text
    text = re.sub(r"\\.", " ", text)  # drop escaped characters like \&
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def section_key(title: str) -> str:
    words = slugify(title).split("-")
    for word in words:
        if word in SECTION_KEYS:
            return SECTION_KEYS[word]
    return slugify(title)


def heading_name(arg: str) -> str:
    """Name of a subheading/project from its first argument.

    `\\textbf{ALEXIDE} $|$ \\emph{React, ...}` -> `ALEXIDE`; `Beeline` -> `Beeline`.
    """
    bold = arg.find("\\textbf")
    if bold != -1:
        name, _ = read_group(arg, skip_space(arg, bold + len("\\textbf")))
        return name
    return arg.split("$|$")[0].strip()


def split_top_level_lines(content: str) -> list[tuple[int, int, int]]:
    """Split on `\\\\` line breaks that are not inside braces.

    Returns (start, end, next_start) per line: the line is content[start:end] and the
    separator (if any) is content[end:next_start].
    """
    lines = []
    start = 0
    depth = 0
    i = 0
    while i < len(content):
        char = content[i]
        if char == "\\":
            if depth == 0 and content.startswith("\\\\", i):
                lines.append((start, i, i + 2))
                start = i + 2
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        i += 1
    lines.append((start, len(content), len(content)))
    return lines


# --- Parser ------------------------------------------------------------------------------


class _Builder:
    """Collects segments and blocks while keeping block IDs unique."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.blocks: list[Block] = []
        self.counters: dict[str, int] = {}

    def raw(self, text: str) -> None:
        if text:
            self.segments.append(Segment(text=text))

    def next_index(self, prefix: str) -> int:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return self.counters[prefix]

    def add(self, block: Block, line: int) -> None:
        if any(existing.id == block.id for existing in self.blocks):
            raise LatexParseError(f"Duplicate block ID '{block.id}' on line {line}")
        self.blocks.append(block)
        self.segments.append(Segment(block_id=block.id))


def parse(tex: str) -> CVDocument:
    """Parse a CV. `render(parse(tex)) == tex` holds for any input this accepts."""
    start = tex.find(BEGIN_DOCUMENT)
    if start == -1:
        raise LatexParseError("No \\begin{document} found")
    preamble, body = tex[:start], tex[start:]

    builder = _Builder()
    section = ""
    heading: str | None = None
    raw_start = 0  # start of body text not yet emitted as a raw segment
    i = 0
    while i < len(body):
        char = body[i]
        if char == "%":
            i = skip_comment(body, i)
            continue
        if char != "\\":
            i += 1
            continue

        name, after = read_command_name(body, i)
        if name == "section":
            if body.startswith("*", after):
                after += 1
            (section,), i = read_args(body, after, 1)
            heading = None
        elif name == "resumeSubheading":
            args, i = read_args(body, after, 4)
            heading = heading_name(args[0])
        elif name == "resumeProjectHeading":
            args, i = read_args(body, after, 2)
            heading = heading_name(args[0])
        elif name == "resumeItem":
            open_brace = skip_space(body, after)
            content, end = read_group(body, open_brace)
            content_start = open_brace + 1
            builder.raw(body[raw_start:content_start])
            line = line_of(tex, start + content_start)
            if section_key(section) == "skills":
                _add_skills_lines(builder, content, section, line)
            else:
                _add_item(builder, content, section, heading, line)
            raw_start = end - 1  # the closing brace is emitted with the next raw text
            i = end
        else:
            i = after

    builder.raw(body[raw_start:])
    return CVDocument(preamble=preamble, segments=builder.segments, blocks=builder.blocks)


def _add_item(
    builder: _Builder, content: str, section: str, heading: str | None, line: int
) -> None:
    key = section_key(section)
    prefix = key if heading is None else f"{key}.{slugify(heading)}"
    kind: BlockKind = "summary" if key == "summary" else "item"
    block = Block(
        id=f"{prefix}.{builder.next_index(prefix)}",
        kind=kind,
        section=section,
        parent_heading=heading,
        text=content,
        editable=key in EDITABLE_SECTIONS,
    )
    builder.add(block, line)


def _add_skills_lines(builder: _Builder, content: str, section: str, line: int) -> None:
    """Each `\\textbf{Label:} a, b, c \\\\` line becomes its own block.

    The label, the `\\\\` separator and surrounding whitespace stay raw, so only the list
    of skills itself is editable.
    """
    for start, end, next_start in split_top_level_lines(content):
        text_start = skip_space(content[:end], start)
        text_end = len(content[:end].rstrip())
        label = None
        if content.startswith("\\textbf", text_start):
            label, text_start = read_group(
                content, skip_space(content, text_start + len("\\textbf"))
            )
            while text_start < text_end and content[text_start].isspace():
                text_start += 1

        builder.raw(content[start:text_start])
        if text_start < text_end:
            name = slugify(label) if label else str(builder.next_index("skills"))
            block = Block(
                id=f"skills.{name}",
                kind="skills_line",
                section=section,
                parent_heading=label,
                text=content[text_start:text_end],
                editable=True,
            )
            builder.add(block, line)
            builder.raw(content[text_end:next_start])
        else:
            builder.raw(content[text_start:next_start])
