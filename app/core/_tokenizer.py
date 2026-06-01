"""
app/core/_tokenizer.py
======================
Token counting with tiktoken (exact) or calibrated heuristic (fallback).

Loaded lazily — tiktoken's BPE file is downloaded on first call to count_tokens().
All network / import errors are silently caught; the heuristic is used instead.

Install tiktoken for exact counts:
    pip install tiktoken
"""
from __future__ import annotations

_enc = None
_tiktoken_ok = False
_tiktoken_tried = False

TOKENIZER_NAME = "heuristic(chars÷4)"
TOKENIZER_AVAILABLE = False


def _try_init_tiktoken():
    """One-shot lazy init — swallows every possible failure."""
    global _enc, _tiktoken_ok, _tiktoken_tried, TOKENIZER_NAME, TOKENIZER_AVAILABLE
    if _tiktoken_tried:
        return
    _tiktoken_tried = True
    try:
        import tiktoken as _tk  # noqa: PLC0415
        _enc = _tk.get_encoding("cl100k_base")
        _tiktoken_ok = True
        TOKENIZER_NAME = "tiktoken/cl100k_base"
        TOKENIZER_AVAILABLE = True
    except Exception:  # noqa: BLE001
        # ImportError, OSError, requests.ProxyError, SSL errors …
        pass


def count_tokens(text: str) -> int:
    """
    Return the number of tokens in *text*.

    Uses tiktoken (cl100k_base) when available — matches GPT-4 / most
    open models within a few percent.  Falls back to ``len(text) // 4``
    (English prose average ≈ 4 chars/token, ±5 %).
    """
    _try_init_tiktoken()
    if _tiktoken_ok and _enc is not None:
        return len(_enc.encode(text))
    return max(1, len(text) // 4)
