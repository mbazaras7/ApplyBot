# ApplyBot

CLI agent that tailors a LaTeX CV ("Jake's Resume" template) to a job description, with a mandatory human review step before compiling to PDF.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and ANTHROPIC_MODEL
```

## Run

```bash
uv run applybot tailor path/to/main.tex --job-text "..."
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
