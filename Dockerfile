ARG UV_VERSION=0.5.11
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_tools

FROM python:3.12-slim AS magpie

COPY --from=uv_tools /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Resolve dependencies before copying source so edits do not invalidate the
# dependency layer. aviary-mcp is a public PyPI release, so the frozen lock
# resolves without any registry credential.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY magpie/ ./magpie/
COPY app/ ./app/
COPY scenarios/ ./scenarios/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MAGPIE_HOST=0.0.0.0 \
    MAGPIE_PORT=7351 \
    MAGPIE_RUNTIME_DIR=/var/lib/magpie

# The runtime dir is a mount point for a volume; state never lives in the image.
RUN mkdir -p /var/lib/magpie

EXPOSE 7351

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=6 \
    CMD curl -fsS "http://127.0.0.1:${MAGPIE_PORT}/birdz" >/dev/null || exit 1

CMD ["python", "-m", "magpie.server"]
