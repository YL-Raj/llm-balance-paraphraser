"""
app/core/backend_pool.py
Manages a pool of LLM backends:
  - Weighted round-robin selection
  - Async health checks
  - Per-backend latency / error tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from app.core.config import BackendConfig, settings

logger = logging.getLogger(__name__)


@dataclass
class BackendStats:
    url: str
    model: str
    weight: int
    healthy: bool = True
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    _latency_window: List[float] = field(default_factory=list)

    def record(self, latency_ms: float, success: bool):
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self._latency_window.append(latency_ms)
        if len(self._latency_window) > 50:          # rolling window
            self._latency_window.pop(0)
        self.avg_latency_ms = sum(self._latency_window) / len(self._latency_window)

    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "model": self.model,
            "healthy": self.healthy,
            "total_requests": self.total_requests,
            "error_rate": round(self.error_rate(), 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class BackendPool:
    """Thread-safe pool with weighted round-robin + health-check eviction."""

    def __init__(self, backends: List[BackendConfig]):
        self._stats: List[BackendStats] = [
            BackendStats(url=b.url, model=b.model, weight=b.weight)
            for b in backends
        ]
        self._rr_index: int = 0
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=settings.BACKEND_TIMEOUT)

    # ── Selection ──────────────────────────────────────────────────────────

    async def pick(self) -> Optional[BackendStats]:
        """Weighted round-robin over healthy backends."""
        async with self._lock:
            healthy = [s for s in self._stats if s.healthy]
            if not healthy:
                # all down: allow any and hope for the best
                healthy = self._stats
            # build weighted list
            pool = []
            for s in healthy:
                pool.extend([s] * s.weight)
            if not pool:
                return None
            chosen = pool[self._rr_index % len(pool)]
            self._rr_index += 1
            return chosen

    # ── Inference call ─────────────────────────────────────────────────────

    async def complete(self, prompt: str, max_tokens: int) -> Dict:
        """Send prompt to a backend and return raw completion data."""
        backend = await self.pick()
        if backend is None:
            raise RuntimeError("No available LLM backends.")

        start = time.perf_counter()
        success = False
        try:
            # ── Ollama-compatible API (also works with LM Studio / Jan) ────
            payload = {
                "model": backend.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            }
            resp = await self._client.post(
                f"{backend.url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            success = True
            return {
                "text": data.get("response", "").strip(),
                "model": backend.model,
                "backend_url": backend.url,
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            }
        except Exception as exc:
            logger.warning("Backend %s failed: %s", backend.url, exc)
            backend.healthy = False         # mark down until next health check
            raise
        finally:
            latency = (time.perf_counter() - start) * 1000
            backend.record(latency, success)

    # ── Health checks ──────────────────────────────────────────────────────

    async def _check_health(self, stat: BackendStats):
        try:
            resp = await self._client.get(f"{stat.url}/api/tags", timeout=5.0)
            stat.healthy = resp.status_code == 200
        except Exception:
            stat.healthy = False
        logger.debug("Health %s → %s", stat.url, "UP" if stat.healthy else "DOWN")

    async def _health_loop(self):
        while True:
            await asyncio.gather(*[self._check_health(s) for s in self._stats])
            await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)

    async def start_health_checks(self):
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("Health-check loop started (interval=%ds)", settings.HEALTH_CHECK_INTERVAL)

    async def stop_health_checks(self):
        if self._health_task:
            self._health_task.cancel()
        await self._client.aclose()

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> List[Dict]:
        return [s.to_dict() for s in self._stats]
