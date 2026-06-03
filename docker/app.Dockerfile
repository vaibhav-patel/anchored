# syntax=docker/dockerfile:1
FROM python:3.11-slim

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl is used by the healthcheck / debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY anchored ./anchored
RUN uv pip install --system -e ".[dev]"

# Source is bind-mounted at runtime via compose for fast iteration.
CMD ["sleep", "infinity"]
