"""
app/core/token_analyzer.py
Professional token & memory analysis engine.

No LLM call required — all analysis is local and instant.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.core._tokenizer import count_tokens, TOKENIZER_NAME, TOKENIZER_AVAILABLE

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# params_b: billions of parameters
# layers: transformer layers
# kv_heads: number of KV attention heads
# head_dim: per-head dimension
# context_window: max tokens
MODEL_REGISTRY: Dict[str, Dict] = {
    "llama3.2:3b": {
        "label": "Llama 3.2 - 3B",
        "params_b": 3.21, "layers": 28, "kv_heads": 8,  "head_dim": 64,
        "context_window": 131_072,
    },
    "phi3:mini": {
        "label": "Phi-3 Mini - 3.8B",
        "params_b": 3.82, "layers": 32, "kv_heads": 32, "head_dim": 96,
        "context_window": 128_000,
    },
    "mistral:7b": {
        "label": "Mistral - 7B",
        "params_b": 7.24, "layers": 32, "kv_heads": 8,  "head_dim": 128,
        "context_window": 32_768,
    },
    "llama3:8b": {
        "label": "Llama 3 - 8B",
        "params_b": 8.03, "layers": 32, "kv_heads": 8,  "head_dim": 128,
        "context_window": 8_192,
    },
    "gemma2:9b": {
        "label": "Gemma 2 - 9B",
        "params_b": 9.24, "layers": 42, "kv_heads": 8,  "head_dim": 256,
        "context_window": 8_192,
    },
    "llama2:13b": {
        "label": "Llama 2 - 13B",
        "params_b": 13.02, "layers": 40, "kv_heads": 40, "head_dim": 128,
        "context_window": 4_096,
    },
    "llama3:70b": {
        "label": "Llama 3 - 70B",
        "params_b": 70.55, "layers": 80, "kv_heads": 8,  "head_dim": 128,
        "context_window": 8_192,
    },
}

QUANT_BYTES: Dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "q8": 1.0, "q4": 0.5, "q2": 0.25,
}

VRAM_TIERS = {
    "4GB  (GTX 1650 / M1 base)": 4,
    "8GB  (RTX 3070 / M2 base)": 8,
    "16GB (RTX 4080 / M2 Pro)":  16,
    "24GB (RTX 4090 / M3 Max)":  24,
    "48GB (A6000 / M2 Ultra)":   48,
    "80GB (A100 SXM)":           80,
}

PROMPT_OVERHEAD_TOKENS = 55


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InputStats:
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    token_count: int
    tokenizer: str
    avg_tokens_per_word: float
    avg_chars_per_token: float


@dataclass
class PromptStats:
    prompt_token_count: int
    prompt_overhead_tokens: int
    content_tokens: int
    max_output_tokens: int
    total_context_tokens: int


@dataclass
class BudgetMetrics:
    max_words: int
    max_tokens: int
    input_word_count: int
    input_token_count: int
    word_utilization_pct: float
    token_utilization_pct: float
    needs_compression: bool
    compression_ratio_needed: float
    estimated_output_tokens: int
    will_pre_compress: bool


@dataclass
class ModelMemoryProfile:
    model_key: str
    label: str
    params_b: float
    context_window: int
    weight_fp16_gb: float
    weight_q8_gb: float
    weight_q4_gb: float
    kv_cache_mb: float
    kv_cache_gb: float
    total_fp16_gb: float
    total_q4_gb: float
    context_used_tokens: int
    context_window_tokens: int
    context_utilization_pct: float
    context_overflow: bool
    vram_fits: Dict[str, bool]


@dataclass
class CompressionProjection:
    input_tokens: int
    target_tokens: int
    ratio_needed: float
    savings_tokens: int
    savings_pct: float
    passes_estimated: int
    will_hard_truncate: bool


@dataclass
class AnalysisResult:
    input_stats: InputStats
    prompt_stats: PromptStats
    budget: BudgetMetrics
    models: List[ModelMemoryProfile]
    compression: CompressionProjection
    tokenizer_available: bool
    tokenizer_name: str
    warnings: List[str]
    recommendations: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_sentences(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))

def _count_paragraphs(text: str) -> int:
    return max(1, len([p for p in text.split("\n\n") if p.strip()]))

def _weight_gb(params_b: float, quant: str) -> float:
    return params_b * QUANT_BYTES.get(quant, 2.0)

def _kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> float:
    return 2 * layers * kv_heads * head_dim * 2  # FP16

def _vram_fits(total_gb: float) -> Dict[str, bool]:
    return {label: total_gb <= limit for label, limit in VRAM_TIERS.items()}

def _compression_passes(ratio_needed: float, max_passes: int = 2) -> Tuple[int, bool]:
    if ratio_needed >= 1.0:
        return 0, False
    remaining = ratio_needed
    for p in range(1, max_passes + 1):
        remaining = remaining / 0.55
        if remaining >= 1.0:
            return p, False
    return max_passes, True


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class TokenAnalyzer:
    def __init__(self, compression_threshold: float = 3.0, max_compression_passes: int = 2):
        self.compression_threshold = compression_threshold
        self.max_compression_passes = max_compression_passes

    def analyze(
        self,
        text: str,
        mode: str = "paraphrase",
        max_words: int = 120,
        max_tokens: Optional[int] = None,
        domain: Optional[str] = None,
        target_models: Optional[List[str]] = None,
    ) -> AnalysisResult:
        warnings: List[str] = []
        recommendations: List[str] = []

        text = unicodedata.normalize("NFC", text).strip()

        # 1. Input statistics
        words = text.split()
        word_count = len(words)
        token_count = count_tokens(text)
        char_count = len(text)
        sentences = _count_sentences(text)
        paragraphs = _count_paragraphs(text)
        avg_tpw = round(token_count / max(word_count, 1), 3)
        avg_cpt = round(char_count / max(token_count, 1), 3)

        input_stats = InputStats(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentences,
            paragraph_count=paragraphs,
            token_count=token_count,
            tokenizer=TOKENIZER_NAME,
            avg_tokens_per_word=avg_tpw,
            avg_chars_per_token=avg_cpt,
        )

        # 2. Prompt statistics
        domain_tokens = count_tokens(f"Domain: {domain}.\n") if domain else 0
        prompt_overhead = PROMPT_OVERHEAD_TOKENS + domain_tokens
        keyword_budget = min(60, max_words)
        will_pre_compress = (
            mode == "keywords_only"
            or (token_count > max_words * self.compression_threshold)
        )
        content_in_prompt = (
            count_tokens(", ".join(words[:keyword_budget])) if will_pre_compress else token_count
        )
        effective_max_tokens = max_tokens or (max_words * 2)
        prompt_tokens = prompt_overhead + content_in_prompt

        prompt_stats = PromptStats(
            prompt_token_count=prompt_tokens,
            prompt_overhead_tokens=prompt_overhead,
            content_tokens=content_in_prompt,
            max_output_tokens=effective_max_tokens,
            total_context_tokens=prompt_tokens + effective_max_tokens,
        )

        # 3. Budget metrics
        word_util = round((word_count / max(max_words, 1)) * 100, 2)
        needs_compression = word_count > max_words
        ratio_needed = round(max_words / max(word_count, 1), 4) if needs_compression else 1.0

        budget = BudgetMetrics(
            max_words=max_words,
            max_tokens=effective_max_tokens,
            input_word_count=word_count,
            input_token_count=token_count,
            word_utilization_pct=word_util,
            token_utilization_pct=0.0,
            needs_compression=needs_compression,
            compression_ratio_needed=ratio_needed,
            estimated_output_tokens=effective_max_tokens,
            will_pre_compress=will_pre_compress,
        )

        # 4. Model memory profiles
        keys = target_models if target_models else list(MODEL_REGISTRY.keys())
        model_profiles: List[ModelMemoryProfile] = []

        for key in keys:
            spec = MODEL_REGISTRY.get(key)
            if spec is None:
                warnings.append(f"Unknown model '{key}' — skipped.")
                continue

            wt_fp16 = _weight_gb(spec["params_b"], "fp16")
            wt_q8   = _weight_gb(spec["params_b"], "q8")
            wt_q4   = _weight_gb(spec["params_b"], "q4")

            kv_bpt = _kv_bytes_per_token(spec["layers"], spec["kv_heads"], spec["head_dim"])
            kv_bytes = kv_bpt * prompt_stats.total_context_tokens
            kv_mb = round(kv_bytes / 1e6, 3)
            kv_gb = round(kv_bytes / 1e9, 4)

            total_fp16 = round(wt_fp16 + kv_gb, 3)
            total_q4   = round(wt_q4  + kv_gb, 3)

            ctx_used = prompt_stats.total_context_tokens
            ctx_win  = spec["context_window"]
            ctx_pct  = round((ctx_used / ctx_win) * 100, 2)
            ctx_overflow = ctx_used > ctx_win

            if ctx_overflow:
                warnings.append(
                    f"{spec['label']}: context window overflow "
                    f"({ctx_used} tokens > {ctx_win} limit). "
                    "Reduce input or lower max_tokens."
                )

            model_profiles.append(ModelMemoryProfile(
                model_key=key,
                label=spec["label"],
                params_b=spec["params_b"],
                context_window=ctx_win,
                weight_fp16_gb=round(wt_fp16, 3),
                weight_q8_gb=round(wt_q8, 3),
                weight_q4_gb=round(wt_q4, 3),
                kv_cache_mb=kv_mb,
                kv_cache_gb=kv_gb,
                total_fp16_gb=total_fp16,
                total_q4_gb=total_q4,
                context_used_tokens=ctx_used,
                context_window_tokens=ctx_win,
                context_utilization_pct=ctx_pct,
                context_overflow=ctx_overflow,
                vram_fits=_vram_fits(total_q4),
            ))

        # 5. Compression projection
        passes_est, will_truncate = _compression_passes(ratio_needed, self.max_compression_passes)
        savings_tokens = max(0, token_count - effective_max_tokens)
        savings_pct = round((savings_tokens / max(token_count, 1)) * 100, 2)

        compression = CompressionProjection(
            input_tokens=token_count,
            target_tokens=effective_max_tokens,
            ratio_needed=ratio_needed,
            savings_tokens=savings_tokens,
            savings_pct=savings_pct,
            passes_estimated=passes_est,
            will_hard_truncate=will_truncate,
        )

        # 6. Warnings & recommendations
        if word_count < 10:
            warnings.append("Input is very short — analysis may be low quality.")
        if word_count > max_words * 10:
            warnings.append(
                f"Input ({word_count} words) is {round(word_count/max_words,1)}x "
                "the output budget. Heavy compression required."
            )
        if will_pre_compress:
            recommendations.append(
                "Pre-compression will reduce context to keywords before the LLM call — "
                "saves tokens but may lose nuance. Increase max_words if precision matters."
            )
        if will_truncate:
            recommendations.append(
                f"Hard truncation will trigger after {self.max_compression_passes} "
                "compression pass(es). Consider raising max_words or simplifying input."
            )
        if not TOKENIZER_AVAILABLE:
            recommendations.append(
                "Install tiktoken (`pip install tiktoken`) for exact token counts. "
                "Current counts are estimates (chars / 4)."
            )

        fitting_8gb = [
            m for m in model_profiles
            if m.vram_fits.get("8GB  (RTX 3070 / M2 base)", False) and not m.context_overflow
        ]
        if fitting_8gb:
            cheapest = min(fitting_8gb, key=lambda m: m.params_b)
            recommendations.append(
                f"For 8 GB VRAM, '{cheapest.label}' (Q4) is the lightest capable model "
                f"for this request ({cheapest.total_q4_gb:.2f} GB total)."
            )

        return AnalysisResult(
            input_stats=input_stats,
            prompt_stats=prompt_stats,
            budget=budget,
            models=model_profiles,
            compression=compression,
            tokenizer_available=TOKENIZER_AVAILABLE,
            tokenizer_name=TOKENIZER_NAME,
            warnings=warnings,
            recommendations=recommendations,
        )
