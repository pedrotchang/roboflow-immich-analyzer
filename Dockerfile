# syntax=docker/dockerfile:1
# CPU-only client image (~150 MB). All GPU work happens in the Roboflow
# Inference server; this container just moves bytes and does arithmetic.

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first so source edits don't bust the layer cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
LABEL org.opencontainers.image.title="roboflow-immich-analyzer" \
      org.opencontainers.image.description="Triage an Immich library with a self-hosted Roboflow Inference server" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER 1000:1000
ENTRYPOINT ["immich-analyzer"]
CMD ["run"]
