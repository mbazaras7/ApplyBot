"""Turn a job description source (URL, file or stdin) into plain text."""

import re
import sys
from pathlib import Path
from typing import TextIO

import httpx
import trafilatura
from pypdf import PdfReader

USER_AGENT = "ApplyBot/0.1 (CV tailoring CLI)"
TIMEOUT_SECONDS = 20.0


class FetchError(RuntimeError):
    """The job description could not be read."""


def html_to_text(html: str) -> str:
    """Keep the readable main content of a web page, dropping menus, footers and scripts."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise FetchError("Could not find readable text in the page.")
    return str(text)


def fetch_text(url: str, transport: httpx.BaseTransport | None = None) -> str:
    """Download a web page and return its readable text.

    Reusable for any page, not only job postings (e.g. company pages later on).
    `transport` lets tests serve fake responses without touching the network.
    """
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            transport=transport,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        raise FetchError(f"Could not download {url}: {error}") from error
    return html_to_text(response.text)


def collapse_spaces(text: str) -> str:
    """PDF text often comes out as `don't   expect   you`; squeeze runs of spaces/tabs."""
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())


def pdf_to_text(path: Path) -> str:
    pages = PdfReader(path).pages
    return collapse_spaces("\n".join(page.extract_text() or "" for page in pages))


def fetch_job(
    source: str,
    *,
    stdin: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Read a job description from `source`.

    - `http://` or `https://` -> download the page and extract its text
    - `-` -> read from standard input
    - anything else -> a file path (`.pdf`, `.html`/`.htm`, or plain text)
    """
    if source == "-":
        text = (stdin or sys.stdin).read()
    elif source.startswith(("http://", "https://")):
        text = fetch_text(source, transport=transport)
    else:
        path = Path(source)
        if not path.is_file():
            raise FetchError(f"No such file: {source}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = pdf_to_text(path)
        elif suffix in (".html", ".htm"):
            text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

    text = text.strip()
    if not text:
        raise FetchError(f"The job description from '{source}' is empty.")
    return text
