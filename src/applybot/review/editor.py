"""Edit proposed edits by hand in a text editor.

The file lists each edit as a `[block_id]` line followed by its text, so the reviewer
changes only block text, never LaTeX structure.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from applybot.tools.generate_edits import Edit, EditPlan

HEADER = re.compile(r"^\[([^\]\s]+)\]\s*$")
HAND_EDIT_RATIONALE = "Edited by hand"

INSTRUCTIONS = """\
# Edit the text under each [block_id] line, then save and close the editor.
# - Delete a whole section (header and text) to drop that edit.
# - Add a section with any editable block ID to edit another block.
# - Lines starting with # are ignored. The edits are validated again afterwards.
"""


class EditFileError(ValueError):
    """The edited file couldn't be read back as edits."""


def edits_to_text(plan: EditPlan) -> str:
    sections = [f"[{edit.block_id}]\n{edit.new_text}\n" for edit in plan.edits]
    return INSTRUCTIONS + "\n" + "\n".join(sections)


def text_to_edits(text: str, previous: EditPlan) -> EditPlan:
    """Parse the edited file. Existing edits keep their rationale; new ones get a default."""
    rationales = {edit.block_id: edit.rationale for edit in previous.edits}
    sections: list[tuple[str, list[str]]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            continue
        header = HEADER.match(line)
        if header:
            sections.append((header.group(1), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip():
            raise EditFileError(f"Line {number}: text before the first [block_id] header")

    edits = []
    for block_id, lines in sections:
        new_text = "\n".join(lines).strip()
        if not new_text:
            raise EditFileError(f"[{block_id}] has no text; delete the section to drop the edit")
        rationale = rationales.get(block_id, HAND_EDIT_RATIONALE)
        edits.append(Edit(block_id=block_id, new_text=new_text, rationale=rationale))
    return EditPlan(edits=edits)


def editor_command() -> list[str]:
    """$VISUAL, then $EDITOR, then a sensible default for the platform."""
    for variable in ("VISUAL", "EDITOR"):
        if value := os.environ.get(variable):
            return shlex.split(value, posix=os.name != "nt")
    if sys.platform == "win32":
        return ["notepad"]
    return [next((e for e in ("nano", "vi") if shutil.which(e)), "vi")]


def open_in_editor(path: Path) -> None:
    """Open `path` in the user's editor and wait until it's closed."""
    subprocess.run([*editor_command(), str(path)], check=False)
