# syntax=docker/dockerfile:1
# 3.10: the bundled ranking_phase_fields package targets Python <3.11.
ARG PYTHON_VERSION=3.10

# Stage 1: build the locked environment with uv
FROM python:${PYTHON_VERSION}-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
# Install the exact locked dependency set from uv.lock plus the project (with its
# bundled reference data) into /app/.venv. uv resolves torch from the CPU index
# per the pyproject/uv.lock config; tensorflow-cpu is CPU-only from PyPI. No
# manual CPU-torch step is needed.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-editable

# Stage 2: minimal runtime image (same base, so native libs match)
FROM python:${PYTHON_VERSION}-slim
RUN groupadd -r lmds && useradd -r -g lmds -u 1000 -m -d /home/lmds lmds
WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# The platform runs the container as uid 1000. The runner creates /data/input
# and /data/output, so /app and all of /data (and /home/lmds) must be writable
# by that user. (Model reference data is bundled in the package, not an asset.)
RUN mkdir -p /data/input /data/output && chown -R 1000:1000 /app /data /home/lmds
USER 1000:1000
ENTRYPOINT ["model-run"]
