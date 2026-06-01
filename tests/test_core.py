"""
tests/test_core.py
Unit tests for preprocessor and postprocessor — no LLM required.
"""

import pytest
from app.core.preprocessor import (
    normalize_text,
    estimate_tokens,
    keyword_compress,
    build_prompt,
    build_compression_prompt,
    Preprocessor,
)
from app.core.postprocessor import count_words, clean_output, hard_truncate


def test_normalize_strips_control_chars():
    raw = "Hello\x00 World\x01"
    assert "\x00" not in normalize_text(raw)


def test_normalize_collapses_whitespace():
    raw = "word1   word2\t\tword3"
    assert "  " not in normalize_text(raw)


def test_normalize_collapses_excess_newlines():
    raw = "para1\n\n\n\n\npara2"
    assert "\n\n\n" not in normalize_text(raw)


def test_estimate_tokens_basic():
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_minimum():
    assert estimate_tokens("hi") >= 1


def test_keyword_compress_returns_string():
    text = "The quick brown fox jumps over the lazy dog near the riverbank"
    result = keyword_compress(text, max_keywords=5)
    assert isinstance(result, str) and len(result) > 0


def test_build_prompt_contains_limit():
    prompt = build_prompt("Some text.", "summary", 50)
    assert "50" in prompt


def test_build_prompt_with_domain():
    prompt = build_prompt("text", "paraphrase", 80, domain="legal")
    assert "legal" in prompt.lower()


def test_build_prompt_ends_with_output_marker():
    prompt = build_prompt("text", "keywords_only", 30)
    assert prompt.strip().endswith("OUTPUT:")


def test_build_compression_prompt():
    prompt = build_compression_prompt("some long text here", 20)
    assert "20" in prompt and "some long text here" in prompt


def test_preprocessor_returns_dict():
    pp = Preprocessor()
    result = pp.process("Hello world this is a test.", "paraphrase", 50)
    assert all(k in result for k in ["prompt", "clean_text", "was_pre_compressed", "estimated_input_tokens"])


def test_preprocessor_keywords_mode_compresses():
    pp = Preprocessor()
    result = pp.process("word " * 500, "keywords_only", 30)
    assert result["was_pre_compressed"] is True


def test_count_words():
    assert count_words("hello world foo bar") == 4
    assert count_words("") == 0


def test_clean_output_strips_prefix():
    assert clean_output("OUTPUT: hello world") == "hello world"


def test_hard_truncate_exact():
    text = " ".join(f"word{i}" for i in range(200))
    truncated = hard_truncate(text, 50)
    assert len(truncated.rstrip(" …").split()) == 50


def test_hard_truncate_within_budget():
    text = "only five words here"
    assert hard_truncate(text, 20) == text
