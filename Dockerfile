# syntax=docker/dockerfile:1
#
# Stages:
#   runtime  Python 3.14 + Tectonic + nano + a non-root user: what the app runs on
#   deps     + uv, the dependencies, and the TeX packages and embedding model downloaded
#            now, so a run needs no network except the Anthropic API (slow, rarely rebuilt)
#   project  + ApplyBot itself (fast: the only stage a code change rebuilds)
#   app      runtime + the finished .venv and caches; no uv, no build leftovers
#            docker run -it --rm --env-file .env -v "${PWD}:/work" applybot tailor ...
#   test     project + dev dependencies and the tests; CI runs the compile tests in it

FROM python:3.14-slim AS runtime

ARG TECTONIC_VERSION=0.17.0
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# nano is the editor for the review step's "edit" option.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl nano \
    && curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
       | tar -xz -C /usr/local/bin \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home applybot && mkdir /app /work && chown applybot /app /work
USER applybot
WORKDIR /app


FROM runtime AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# The cache mount gives uv a download cache that lives outside the image: rebuilds reuse
# it, but none of it ends up in a layer.
COPY --chown=applybot pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/home/applybot/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-dev --no-install-project

# Download what a run needs, before copying any code, so code changes don't redo this.
# The TeX run uses the same two compatibility lines as latex/compiler.py's WRAPPER, and
# the model/cache path match tools/cv_index.py; the tests inside the image catch drift.
COPY --chown=applybot tests/fixtures/sample_cv.tex /tmp/warmup/cv.tex
RUN cd /tmp/warmup \
    && printf '%s\n' '\providecommand{\pdfglyphtounicode}[2]{}' \
         '\ifdefined\pdfgentounicode\else\newcount\pdfgentounicode\fi' '\input{cv.tex}' > build.tex \
    && printf '%s\n' '\documentclass{article}' '\begin{document}' 'warm up' '\end{document}' > plain.tex \
    && tectonic build.tex && tectonic plain.tex \
    && python -c "from pathlib import Path; from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=str(Path.home() / '.cache/applybot/fastembed')); \
print('model cached')" \
    && rm -rf /tmp/warmup


FROM deps AS project
COPY --chown=applybot README.md ./
COPY --chown=applybot src ./src
# --no-editable installs ApplyBot into .venv itself, so the app needs no copy of src/.
RUN --mount=type=cache,target=/home/applybot/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-dev --no-editable


FROM runtime AS app
COPY --from=project --chown=applybot /app/.venv /app/.venv
COPY --from=deps --chown=applybot /home/applybot/.cache /home/applybot/.cache
ENV EDITOR=nano
WORKDIR /work
ENTRYPOINT ["applybot"]
CMD ["--help"]


FROM project AS test
RUN --mount=type=cache,target=/home/applybot/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked
COPY --chown=applybot tests ./tests
CMD ["pytest", "-q"]
