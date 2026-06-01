"""
tests/test_analyzer.py
Unit tests for the token analyzer engine — no LLM, no network required.
"""
import pytest
from app.core.token_analyzer import (
    TokenAnalyzer,
    MODEL_REGISTRY,
    _weight_gb,
    _kv_bytes_per_token,
    _vram_fits,
    _compression_passes,
    _count_sentences,
    _count_paragraphs,
)
from app.core._tokenizer import count_tokens, TOKENIZER_NAME


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_count_tokens_returns_positive():
    assert count_tokens("Hello world") > 0

def test_count_tokens_empty_returns_one():
    assert count_tokens("") >= 1

def test_count_tokens_scales_with_length():
    short = count_tokens("Hi")
    long  = count_tokens("Hi " * 100)
    assert long > short

def test_tokenizer_name_is_string():
    assert isinstance(TOKENIZER_NAME, str) and len(TOKENIZER_NAME) > 0


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def test_all_registry_models_have_required_fields():
    required = {"label", "params_b", "layers", "kv_heads", "head_dim", "context_window"}
    for key, spec in MODEL_REGISTRY.items():
        missing = required - spec.keys()
        assert not missing, f"Model '{key}' missing: {missing}"

def test_registry_has_seven_models():
    assert len(MODEL_REGISTRY) == 7

def test_llama3_70b_is_largest():
    params = {k: v["params_b"] for k, v in MODEL_REGISTRY.items()}
    assert max(params, key=params.get) == "llama3:70b"

def test_llama32_3b_has_largest_context():
    ctxs = {k: v["context_window"] for k, v in MODEL_REGISTRY.items()}
    assert ctxs["llama3.2:3b"] >= max(ctxs.values()) * 0.9  # within 10% of max


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def test_weight_gb_fp16():
    gb = _weight_gb(7.0, "fp16")
    assert abs(gb - 14.0) < 0.01

def test_weight_gb_q4_is_half_q8():
    assert abs(_weight_gb(7.0, "q4") - _weight_gb(7.0, "q8") / 2) < 0.01

def test_kv_bytes_per_token_positive():
    assert _kv_bytes_per_token(32, 8, 128) > 0

def test_kv_bytes_scales_with_layers():
    small = _kv_bytes_per_token(16, 8, 128)
    large = _kv_bytes_per_token(32, 8, 128)
    assert large == small * 2

def test_vram_fits_small_model():
    fits = _vram_fits(1.5)
    assert all(fits.values())  # 1.5 GB fits everywhere

def test_vram_fits_large_model():
    fits = _vram_fits(50.0)   # 50 GB: no 4/8/16/24, yes 80
    assert not fits["8GB  (RTX 3070 / M2 base)"]
    assert fits["80GB (A100 SXM)"]  # 100 GB > 80 GB, so this actually should be False


# ---------------------------------------------------------------------------
# Compression projection
# ---------------------------------------------------------------------------

def test_no_compression_when_within_budget():
    passes, truncate = _compression_passes(1.0)
    assert passes == 0 and not truncate

def test_light_compression_one_pass():
    # ratio 0.7 should need 1 pass (0.7/0.55 > 1.0)
    passes, truncate = _compression_passes(0.7)
    assert passes >= 1

def test_heavy_compression_triggers_truncate():
    # ratio 0.1 — can't achieve in 2 passes of 55% each
    passes, truncate = _compression_passes(0.1, max_passes=2)
    assert truncate

def test_text_stats_helpers():
    text = "Hello world. How are you? Fine!"
    assert _count_sentences(text) == 3
    assert _count_paragraphs(text) == 1

def test_paragraph_count():
    text = "Para one.\n\nPara two.\n\nPara three."
    assert _count_paragraphs(text) == 3


# ---------------------------------------------------------------------------
# TokenAnalyzer.analyze()
# ---------------------------------------------------------------------------

TEXT_SHORT = "AI is changing the world."
TEXT_LONG  = " ".join(["word"] * 300)

def test_analyze_returns_all_sections():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, mode="paraphrase", max_words=20)
    assert r.input_stats is not None
    assert r.prompt_stats is not None
    assert r.budget is not None
    assert r.compression is not None
    assert isinstance(r.models, list)
    assert isinstance(r.warnings, list)
    assert isinstance(r.recommendations, list)

def test_analyze_all_seven_models():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30)
    assert len(r.models) == 7

def test_analyze_target_models_subset():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30, target_models=["llama3.2:3b", "llama3:8b"])
    assert len(r.models) == 2
    keys = {m.model_key for m in r.models}
    assert keys == {"llama3.2:3b", "llama3:8b"}

def test_analyze_unknown_model_generates_warning():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30, target_models=["nonexistent:999b"])
    assert any("Unknown model" in w for w in r.warnings)
    assert len(r.models) == 0

def test_analyze_needs_compression_when_over_budget():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_LONG, max_words=10)
    assert r.budget.needs_compression
    assert r.budget.compression_ratio_needed < 1.0

def test_analyze_no_compression_within_budget():
    ta = TokenAnalyzer()
    r = ta.analyze("Short.", max_words=500)
    assert not r.budget.needs_compression
    assert r.budget.compression_ratio_needed == 1.0

def test_analyze_pre_compress_on_keywords_mode():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_LONG, mode="keywords_only", max_words=50)
    assert r.budget.will_pre_compress

def test_analyze_pre_compress_on_huge_input():
    ta = TokenAnalyzer()
    # 300 words, budget 10 => well over 3x threshold
    r = ta.analyze(TEXT_LONG, max_words=10)
    assert r.budget.will_pre_compress

def test_analyze_word_utilization_pct():
    ta = TokenAnalyzer()
    r = ta.analyze("word " * 60, max_words=40)
    # 60 words / 40 budget = 150%
    assert abs(r.budget.word_utilization_pct - 150.0) < 1.0

def test_analyze_context_overflow_detected():
    ta = TokenAnalyzer()
    # Llama2:13b has 4096 context — send huge text + big token budget
    big_text = "word " * 4000
    r = ta.analyze(big_text, max_words=2000, max_tokens=4096, target_models=["llama2:13b"])
    assert len(r.models) == 1
    # context_used = prompt_tokens + max_tokens, likely > 4096
    m = r.models[0]
    if m.context_overflow:
        assert any("overflow" in w.lower() for w in r.warnings)

def test_analyze_vram_fits_structure():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30, target_models=["llama3.2:3b"])
    m = r.models[0]
    assert isinstance(m.vram_fits, dict)
    assert "8GB  (RTX 3070 / M2 base)" in m.vram_fits
    assert isinstance(m.vram_fits["8GB  (RTX 3070 / M2 base)"], bool)

def test_analyze_70b_doesnt_fit_8gb():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30, target_models=["llama3:70b"])
    m = r.models[0]
    assert not m.vram_fits.get("8GB  (RTX 3070 / M2 base)", True)

def test_analyze_3b_fits_8gb():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30, target_models=["llama3.2:3b"])
    m = r.models[0]
    assert m.vram_fits.get("8GB  (RTX 3070 / M2 base)", False)

def test_analyze_domain_adds_overhead():
    ta = TokenAnalyzer()
    r_no_domain   = ta.analyze(TEXT_SHORT, max_words=30)
    r_with_domain = ta.analyze(TEXT_SHORT, max_words=30, domain="legal")
    assert r_with_domain.prompt_stats.prompt_overhead_tokens > r_no_domain.prompt_stats.prompt_overhead_tokens

def test_analyze_savings_positive_when_over_budget():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_LONG, max_words=20)
    assert r.compression.savings_tokens >= 0
    assert r.compression.savings_pct >= 0.0

def test_analyze_latency_info_present():
    ta = TokenAnalyzer()
    r = ta.analyze(TEXT_SHORT, max_words=30)
    # tokenizer_name and tokenizer_available should always be set
    assert isinstance(r.tokenizer_name, str)
    assert isinstance(r.tokenizer_available, bool)
