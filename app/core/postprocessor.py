"""
app/core/postprocessor.py
Enforces the hard word/token budget on LLM output.
Strategy (in order):
  1. If within budget → return as-is.
  2. If over budget → request a compression pass from the LLM.
  3. Repeat up to COMPRESSION_PASSES times.
  4. Hard truncate as final fallback (if settings.HARD_TRUNCATE).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Callable, Awaitable

from app.core.config import settings
from app.core.preprocessor import build_compression_prompt, estimate_tokens

logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    return len(text.split())


def hard_truncate(text: str, max_words: int) -> str:
    """Truncate to max_words, appending '…' if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def clean_output(text: str) -> str:
    """Strip common LLM artefacts from the output."""
    # Remove leading "OUTPUT:" echo
    text = re.sub(r"^OUTPUT\s*:\s*", "", text, flags=re.IGNORECASE)
    # Remove repeated dashes / markdown fences
    text = re.sub(r"^---+\s*", "", text)
    text = re.sub(r"\s*---+$", "", text)
    # Strip wrapping quotes
    text = text.strip().strip('"').strip("'")
    return text.strip()


# Type alias for the LLM call dependency
LLMCallable = Callable[[str, int], Awaitable[Dict]]


class PostProcessor:
    """
    Applies budget enforcement with optional multi-pass LLM compression.

    Usage:
        pp = PostProcessor(llm_complete_fn=pool.complete)
        result = await pp.enforce(raw_output, max_words=120, max_tokens=200)
    """

    def __init__(self, llm_complete_fn: LLMCallable):
        self._llm = llm_complete_fn
        self._max_passes = settings.COMPRESSION_PASSES

    async def enforce(
        self,
        raw_output: str,
        max_words: int,
        max_tokens: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> Dict:
        """
        Returns enriched result dict:
        {
            "output_text": str,
            "word_count": int,
            "compression_passes": int,
            "was_truncated": bool,
            ...original meta fields...
        }
        """
        meta = meta or {}
        text = clean_output(raw_output)
        passes = 0
        truncated = False

        for attempt in range(self._max_passes):
            wc = count_words(text)
            if wc <= max_words:
                break                   # within budget ✓

            logger.info(
                "Output %d words > budget %d — compression pass %d/%d",
                wc, max_words, attempt + 1, self._max_passes,
            )
            compress_prompt = build_compression_prompt(text, max_words)
            compress_tokens = max_tokens or max_words * 2

            try:
                resp = await self._llm(compress_prompt, compress_tokens)
                text = clean_output(resp.get("text", text))
                passes += 1
                # Update meta with the compression call's model info
                meta["model_used"] = resp.get("model", meta.get("model_used"))
            except Exception as exc:
                logger.warning("Compression pass failed: %s", exc)
                break

        # Final safety net
        wc = count_words(text)
        if wc > max_words and settings.HARD_TRUNCATE:
            text = hard_truncate(text, max_words)
            truncated = True
            wc = count_words(text)

        return {
            **meta,
            "output_text": text,
            "word_count": wc,
            "compression_passes": passes,
            "was_truncated": truncated,
        }
