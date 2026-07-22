FROM ghcr.io/astral-sh/uv:0.11.30 AS uv
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --frozen --no-dev && useradd --create-home app \
    && mkdir -p /data/objects /data/derived \
    && chown -R app:app /data

USER app
EXPOSE 8000 8501
CMD ["uv", "run", "uvicorn", "nirman_netra.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
