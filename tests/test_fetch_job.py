import io
from pathlib import Path

import httpx
import pytest

from applybot.tools.fetch_job import FetchError, collapse_spaces, fetch_job

PERSONAL_JOB_PDF = Path(__file__).parent / "fixtures" / "Job_description_test.pdf"

JOB_HTML = """<html><head><title>Backend Engineer - Initech</title></head><body>
<nav><a href="/">Home</a> <a href="/jobs">Jobs</a> <a href="/about">About us</a></nav>
<main><article>
<h1>Backend Engineer</h1>
<p>Initech is hiring a backend engineer to join our platform team in Springfield.
You will design, build and maintain REST APIs used by thousands of customers every day.</p>
<p>Requirements: two or more years of professional Python experience, strong PostgreSQL
skills, and experience writing automated tests. Experience with Docker is a plus.</p>
<p>You will review code, mentor interns and work closely with product managers to ship
reliable features on a regular release schedule.</p>
</article></main>
<footer>Copyright Initech. Cookie settings. Privacy policy.</footer>
<script>trackVisitor();</script>
</body></html>"""


def serve(status: int, body: str) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status, text=body))


def test_url_extracts_main_text() -> None:
    text = fetch_job("https://jobs.example.com/123", transport=serve(200, JOB_HTML))
    assert "two or more years of professional Python" in text
    assert "trackVisitor" not in text
    assert "Cookie settings" not in text


def test_url_http_error() -> None:
    with pytest.raises(FetchError, match="Could not download"):
        fetch_job("https://jobs.example.com/missing", transport=serve(404, "Not found"))


def test_url_without_readable_text() -> None:
    with pytest.raises(FetchError, match="readable text"):
        fetch_job("https://jobs.example.com/app", transport=serve(200, "<html></html>"))


def test_plain_text_file(tmp_path: Path) -> None:
    path = tmp_path / "job.txt"
    path.write_text("  Backend Engineer\nPython, PostgreSQL\n\n", encoding="utf-8")
    assert fetch_job(str(path)) == "Backend Engineer\nPython, PostgreSQL"


def test_html_file(tmp_path: Path) -> None:
    path = tmp_path / "job.html"
    path.write_text(JOB_HTML, encoding="utf-8")
    assert "strong PostgreSQL" in fetch_job(str(path))


def test_stdin() -> None:
    assert fetch_job("-", stdin=io.StringIO("Pasted job text\n")) == "Pasted job text"


def test_missing_file() -> None:
    with pytest.raises(FetchError, match="No such file"):
        fetch_job("does/not/exist.txt")


def test_empty_input() -> None:
    with pytest.raises(FetchError, match="empty"):
        fetch_job("-", stdin=io.StringIO("   \n"))


def test_collapse_spaces() -> None:
    assert collapse_spaces("don't   expect\t you  \n  to  know") == "don't expect you\nto know"


@pytest.mark.skipif(not PERSONAL_JOB_PDF.exists(), reason="job PDF fixture is kept local")
def test_pdf_file() -> None:
    text = fetch_job(str(PERSONAL_JOB_PDF))
    assert len(text) > 100
    assert "  " not in text
