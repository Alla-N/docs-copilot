# syntax=docker/dockerfile:1
#
# The agent service as a container. Build context is agent/, not the repo root:
#
#   docker build -t copilot-agent agent
#
# Two stages. The builder has uv, its cache and the build backend; the runtime image gets
# only the finished virtualenv. Secrets are never part of the image: they arrive as
# environment variables at `docker run` time (see agent/README.md).

# One base image for both stages, on purpose. A virtualenv is not relocatable: .venv/bin/python
# is a symlink to the interpreter that created it (/usr/local/bin/python3.13 here). A runtime
# stage on any other image would hold a venv pointing at a Python that is not there.
ARG PYTHON_IMAGE=python:3.13-slim-trixie

# ---- builder ------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# The same uv as the Mac that wrote uv.lock (agent/.venv/pyvenv.cfg) and as CI
# (.github/workflows/agent.yml). Bump all three together.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

# COMPILE_BYTECODE: write the .pyc files now. At runtime the venv is read-only to the app
#   user, so Python could not write them, and every start would recompile in memory.
# LINK_MODE=copy: the cache is a separate mount, so hardlinks into it are not possible.
# PYTHON_DOWNLOADS=0: use the image's Python, never a uv-managed one. The runtime stage
#   only has the image's Python, so a venv built on anything else would not run there.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Layer 1: third-party dependencies only, from the lock file. The files are bind-mounted,
# not copied, so this layer's cache key is exactly pyproject.toml + uv.lock + .python-version:
# editing a .py file does not invalidate it. --locked fails if uv.lock is out of date with
# pyproject.toml (npm ci's check); --no-dev leaves pytest and ruff out.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project

# Layer 2: our own package. --no-editable installs it into site-packages as a built package
# instead of a link back to /app/src, so the venv alone is the whole application.
# README.md is needed because pyproject.toml declares it: uv_build refuses to build without it.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime ------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# A system user with a fixed id. USER below is numeric so Kubernetes runAsNonRoot can verify
# it without reading /etc/passwd (it refuses a named user).
RUN groupadd --system --gid 999 app \
 && useradd --system --gid 999 --uid 999 --create-home app

# No --chown: the venv stays owned by root, so the process can read and run its code but not
# rewrite it. (uv's example chowns /app to the app user; this is a deliberate difference.)
COPY --from=builder /app/.venv /app/.venv

# The venv's bin first on PATH: `uvicorn` and `python` below are the venv's. No uv in this
# image, so no `uv run` either (it would try to sync at every start).
# PYTHONUNBUFFERED: logs reach `docker logs` (and CloudWatch in 2b) as they are written,
# not when a buffer fills, so a crash cannot swallow its last lines.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER 999:999

# Documentation only: publishing is `docker run -p`, which should be -p 127.0.0.1:8000:8000.
EXPOSE 8000

# Exec form (a JSON list): uvicorn itself is PID 1 and receives the SIGTERM that `docker stop`
# and ECS send, so the lifespan closes the database pool. The shell form would put
# /bin/sh -c in between.
# 0.0.0.0: the container has its own loopback; Docker's port forward arrives on the container's
# network interface, so a server bound to 127.0.0.1 in here is unreachable from outside.
# --factory: create_app() runs at startup and reads settings then, as in local dev.
# One process per container: scale with more containers (ECS tasks), not workers inside one.
CMD ["uvicorn", "--factory", "copilot_agent.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
