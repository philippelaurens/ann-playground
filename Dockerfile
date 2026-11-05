# ==============================================================================
# Multi-stage Dockerfile for ANN Playground
# Optimized with BuildKit cache mounts and explicit wheels
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# Install uv from official binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Synchronize dependencies using BuildKit cache mount to avoid re-downloading wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-python-downloads

# ------------------------------------------------------------------------------
# Stage 2: Runtime Environment
# ------------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Copy isolated pre-built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project configuration and source tree
COPY pyproject.toml ./
COPY src/ ./src/

EXPOSE 8501 5000

# Default command launches the Streamlit entry point
CMD ["streamlit", "run", "src/ann_playground/apps/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
