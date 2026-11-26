"""The HTTP API, called through FastAPI's test client (no server, no network)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import GOOD_ACME, FakeEmbedder, InlineExecutor, ScriptedLLM
from fastapi.testclient import TestClient

from applybot.api.app import create_app
from applybot.service.runs import RunManager

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cv.tex"
JOB_PDF = Path(__file__).parent / "fixtures" / "Job_description_SEGrad_VHI.pdf"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    manager = RunManager(
        tmp_path / "data",
        lambda: ScriptedLLM([[GOOD_ACME]]),
        out_root=tmp_path / "runs",
        compile_pdf=False,
        embedder=FakeEmbedder(),
        executor=InlineExecutor(),  # the run reaches review before POST /runs returns
    )
    with TestClient(create_app(manager)) as test_client:  # `with` runs startup/shutdown
        yield test_client


def cv_upload(tex: bytes | None = None) -> dict[str, tuple[str, bytes, str]]:
    return {"cv": ("main.tex", tex if tex is not None else SAMPLE.read_bytes(), "text/plain")}


def start(client: TestClient, **form: str) -> int:
    response = client.post("/runs", files=cv_upload(), data=form or {"job_text": "Backend job"})
    assert response.status_code == 202, response.text
    run_id: int = response.json()["id"]
    return run_id


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_full_run_over_http(client: TestClient) -> None:
    run_id = start(client, job_text="Backend job", facts="I know Rust.", custom_prompt="Be brief.")

    run = client.get(f"/runs/{run_id}").json()
    assert run["status"] == "awaiting_review"
    assert run["step"] == "Waiting for review"
    assert (run["company"], run["fit_score"], run["has_pdf"]) == ("Initech Ltd", 64, False)

    review = client.get(f"/runs/{run_id}/review").json()
    assert review["plan"]["edits"][0]["block_id"] == GOOD_ACME["block_id"]
    assert review["can_approve"] is True
    assert review["analysis"]["fit_score"] == 64

    response = client.post(f"/runs/{run_id}/decision", json={"action": "approve"})
    assert response.status_code == 202
    assert response.json() == {"id": run_id, "status": "queued"}

    run = client.get(f"/runs/{run_id}").json()
    assert run["status"] == "completed"
    assert run["has_report"] is True
    report = client.get(f"/runs/{run_id}/report")
    assert report.headers["content-type"].startswith("text/markdown")
    assert "Tailoring report" in report.text
    assert client.get(f"/runs/{run_id}/pdf").status_code == 404  # compiling was off

    [summary] = client.get("/runs").json()
    assert (summary["id"], summary["status"]) == (run_id, "completed")


def test_job_file_upload(client: TestClient) -> None:
    files = {**cv_upload(), "job_file": ("job.txt", b"Backend Engineer\r\nPython", "text/plain")}
    response = client.post("/runs", files=files)
    assert response.status_code == 202
    assert client.get(f"/runs/{response.json()['id']}").json()["status"] == "awaiting_review"


@pytest.mark.skipif(not JOB_PDF.exists(), reason="fixture posting not present")
def test_job_pdf_upload(client: TestClient) -> None:
    files = {**cv_upload(), "job_file": ("vhi.pdf", JOB_PDF.read_bytes(), "application/pdf")}
    assert client.post("/runs", files=files).status_code == 202


@pytest.mark.parametrize(
    ("form", "files", "message"),
    [
        ({}, {}, "exactly one of"),
        ({"job_text": "a", "job_url": "https://x.test/job"}, {}, "exactly one of"),
        ({"job_url": "C:/Users/someone/secrets.txt"}, {}, "must start with http"),
        ({}, {"job_file": ("job.pdf", b"not a pdf", "application/pdf")}, "as a PDF"),
        ({}, {"job_file": ("job.txt", b"   ", "text/plain")}, "is empty"),
    ],
)
def test_bad_job_input(
    client: TestClient, form: dict[str, str], files: dict[str, tuple[str, bytes, str]], message: str
) -> None:
    response = client.post("/runs", files={**cv_upload(), **files}, data=form)
    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert client.get("/runs").json() == []  # nothing was started


@pytest.mark.parametrize(
    ("tex", "message"),
    [(b"\xff\xfe not utf-8", "UTF-8"), (b"no document here", "begin{document}")],
)
def test_bad_cv(client: TestClient, tex: bytes, message: str) -> None:
    response = client.post("/runs", files=cv_upload(tex), data={"job_text": "Backend job"})
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_missing_cv_is_a_validation_error(client: TestClient) -> None:
    assert client.post("/runs", data={"job_text": "Backend job"}).status_code == 422


def test_unknown_run(client: TestClient) -> None:
    for path in ("/runs/42", "/runs/42/review", "/runs/42/pdf", "/runs/42/report"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "42" in response.json()["detail"]


def test_review_and_decision_only_while_waiting(client: TestClient) -> None:
    run_id = start(client)
    client.post(f"/runs/{run_id}/decision", json={"action": "approve"})

    assert client.get(f"/runs/{run_id}/review").status_code == 409
    response = client.post(f"/runs/{run_id}/decision", json={"action": "approve"})
    assert response.status_code == 409
    assert "isn't waiting for review" in response.json()["detail"]


def test_decision_is_validated(client: TestClient) -> None:
    run_id = start(client)
    response = client.post(f"/runs/{run_id}/decision", json={"action": "maybe"})
    assert response.status_code == 422
    assert client.get(f"/runs/{run_id}").json()["status"] == "awaiting_review"


def test_hand_edit_over_http(client: TestClient) -> None:
    run_id = start(client)
    own = {**GOOD_ACME, "new_text": GOOD_ACME["new_text"].replace("Redis-backed", "Rust")}
    decision = {"action": "edit", "plan": {"edits": [own]}}
    assert client.post(f"/runs/{run_id}/decision", json=decision).status_code == 202

    review = client.get(f"/runs/{run_id}/review").json()
    assert review["can_approve"] is True  # a new skill in your own edit only warns
    [problem] = [p for p in review["entries"][0]["problems"] if p["rule"] == "new_skill"]
    assert problem["severity"] == "warning"
