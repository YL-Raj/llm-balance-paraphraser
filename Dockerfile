# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="llm-balance-paraphraser"
LABEL description="Open-source real-time LLM token analyzer, paraphraser and compressor"
LABEL org.opencontainers.image.source="https://github.com/YOUR_USERNAME/llm-balance-paraphraser"

WORKDIR /app

# Runtime system deps only (curl needed for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/       ./app/
COPY frontend/  ./frontend/
COPY .env.example .env.example

# Non-root user
RUN adduser --disabled-password --gecos "" --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check — hits the lightweight /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint script handles Ollama readiness before starting uvicorn
COPY --chown=appuser:appuser scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
