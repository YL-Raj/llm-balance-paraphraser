"""
app/core/config.py
Central configuration — driven by environment variables or .env file.
"""

from __future__ import annotations

import os
from typing import List
from pydantic_settings import BaseSettings


class BackendConfig:
    """Minimal descriptor for one LLM backend."""
    def __init__(self, url: str, model: str, weight: int = 1):
        self.url   = url.rstrip("/")
        self.model = model
        self.weight = weight          # relative load-balancing weight

    def __repr__(self) -> str:
        return f"BackendConfig(url={self.url!r}, model={self.model!r})"


class Settings(BaseSettings):
    # ── API ──────────────────────────────────────────────────────────────────
    API_KEY: str = ""                  # empty = no auth (dev mode)
    CORS_ORIGINS: List[str] = ["*"]

    # ── Defaults ─────────────────────────────────────────────────────────────
    DEFAULT_MAX_WORDS: int   = 120
    DEFAULT_MAX_TOKENS: int  = 200
    DEFAULT_MODE: str        = "paraphrase"

    # ── Budget enforcement ────────────────────────────────────────────────────
    COMPRESSION_PASSES: int  = 2       # max re-compression passes
    HARD_TRUNCATE: bool      = True    # final fallback: truncate by word count

    # ── Backend pool ─────────────────────────────────────────────────────────
    # Comma-separated: url|model|weight  e.g.
    #   http://localhost:11434|llama3:8b|2,http://gpu2:11434|mistral:7b|1
    LLM_BACKEND_STR: str = "http://localhost:11434|llama3:8b|1"

    # ── Health checks ────────────────────────────────────────────────────────
    HEALTH_CHECK_INTERVAL: int = 30    # seconds
    BACKEND_TIMEOUT: float     = 60.0  # seconds per request

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def LLM_BACKENDS(self) -> List[BackendConfig]:
        backends = []
        for entry in self.LLM_BACKEND_STR.split(","):
            parts = entry.strip().split("|")
            if len(parts) == 2:
                url, model = parts
                weight = 1
            elif len(parts) == 3:
                url, model, weight = parts
                weight = int(weight)
            else:
                continue
            backends.append(BackendConfig(url=url, model=model, weight=weight))
        return backends if backends else [
            BackendConfig("http://localhost:11434", "llama3:8b", 1)
        ]


settings = Settings()
