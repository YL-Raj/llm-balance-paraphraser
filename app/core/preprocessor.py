"""
app/core/preprocessor.py
Normalises raw input text and builds a tight, mode-aware prompt
that minimises token waste while preserving full intent control.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ── Mode → compact system instruction mapping ───────────────────────────────
MODE_INSTRUCTIONS: dict[str, str] = {
    "paraphrase": (
        "Rewrite the text below. Preserve all key meaning. "
        "Use clear, natural language."
    ),
    "summary": (
        "Summarise the text below. Keep the most important facts only."
    ),
    "bullet_points": (
        "Convert the text below into concise bullet points. "
        "Each bullet: one idea, ≤15 words."
    ),
    "keywords_only": (
        "Extract the most important keywords and short phrases from the text below. "
        "Return them as a comma-separated list, no sentences."
    ),
    "formal": (
        "Rewrite the text below in a formal, professional tone. "
        "Preserve all meaning."
    ),
    "simplify": (
        "Rewrite the text below so a non-expert can understand it easily. "
        "Keep sentences short."
    ),
    "technical": (
        "Rewrite the text below using precise technical language. "
        "Be concise and exact."
    ),
}

DEFAULT_MODE = "paraphrase"


def normalize_text(text: str) -> str:
    """
    Clean raw input:
    - Unicode normalisation (NFC)
    - Collapse excessive whitespace / newlines
    - Strip zero-width and control characters
    - Preserve sentence boundaries
    """
    # NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width and control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", "", text)
    # Collapse 3+ newlines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs on same line
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 chars per token for English.
    Good enough for budget checks without a full tokenizer dependency.
    """
    return max(1, len(text) // 4)


def keyword_compress(text: str, max_keywords: int = 30) -> str:
    """
    Ultra-lightweight keyword extraction via TF-like scoring.
    Used as a pre-pass when the input is very large relative to the budget.
    Falls back to the first N words if no meaningful keywords found.
    """
    # Strip punctuation, lowercase, split
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    STOPWORDS = {
        "the", "and", "for", "are", "but", "not", "you", "all", "any",
        "can", "has", "had", "was", "were", "will", "with", "this", "that",
        "from", "they", "have", "been", "their", "there", "when", "what",
        "which", "who", "how", "its", "than", "then", "also", "into",
        "more", "such", "each", "about", "these", "those", "being",
    }

    freq: dict[str, int] = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1

    top = sorted(freq, key=lambda w: freq[w], reverse=True)[:max_keywords]
    return ", ".join(top) if top else " ".join(text.split()[:max_keywords])


def build_prompt(
    text: str,
    mode: str,
    max_words: int,
    domain: Optional[str] = None,
) -> str:
    """
    Assemble a compact prompt for the LLM.

    Structure (intentionally minimal to save tokens):
      [DOMAIN line — only if provided]
      [INSTRUCTION]
      OUTPUT LIMIT: ≤ N words.
      ---
      [INPUT TEXT]
      ---
      OUTPUT:
    """
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS[DEFAULT_MODE])
    domain_line = f"Domain: {domain}.\n" if domain else ""

    prompt = (
        f"{domain_line}"
        f"{instruction}\n"
        f"Output limit: ≤{max_words} words. Do not exceed this limit.\n"
        f"---\n"
        f"{text}\n"
        f"---\n"
        f"OUTPUT:"
    )
    return prompt


def build_compression_prompt(text: str, max_words: int) -> str:
    """Second-pass prompt used when output still exceeds the budget."""
    return (
        f"Compress the following text to ≤{max_words} words. "
        f"Keep all key information. Do not add new content.\n"
        f"---\n{text}\n---\nOUTPUT:"
    )


class Preprocessor:
    """Stateless pipeline: normalize → (optional compress) → build prompt."""

    def __init__(self, compression_threshold: float = 3.0):
        # If input token count > budget * threshold, pre-compress with keywords
        self.compression_threshold = compression_threshold

    def process(
        self,
        raw_text: str,
        mode: str,
        max_words: int,
        domain: Optional[str] = None,
    ) -> dict:
        """
        Returns:
          {
            "prompt": str,
            "clean_text": str,
            "was_pre_compressed": bool,
            "estimated_input_tokens": int,
          }
        """
        clean = normalize_text(raw_text)
        est_tokens = estimate_tokens(clean)
        budget_tokens = max_words  # ~1 token ≈ 0.75 words, conservative

        pre_compressed = False
        if mode == "keywords_only" or (
            est_tokens > budget_tokens * self.compression_threshold
        ):
            # Pre-compress: reduce context sent to LLM
            clean = keyword_compress(clean, max_keywords=min(60, budget_tokens))
            pre_compressed = True

        prompt = build_prompt(clean, mode, max_words, domain)

        return {
            "prompt": prompt,
            "clean_text": clean,
            "was_pre_compressed": pre_compressed,
            "estimated_input_tokens": estimate_tokens(prompt),
        }
