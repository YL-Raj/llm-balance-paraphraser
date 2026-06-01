"""
app/api/schemas.py
Request and response models with validation.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Mode(str, Enum):
    paraphrase    = "paraphrase"
    summary       = "summary"
    bullet_points = "bullet_points"
    keywords_only = "keywords_only"
    formal        = "formal"
    simplify      = "simplify"
    technical     = "technical"


class ProcessRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=32_000,
        description="The raw input text to process (any domain, any language).",
    )
    mode: Mode = Field(Mode.paraphrase, description="Processing mode.")
    max_words: int = Field(120, ge=10, le=2000, description="Hard maximum word count for the output.")
    max_tokens: Optional[int] = Field(None, ge=10, le=4096,
        description="Hard maximum token budget (overrides max_words estimate if set).")
    domain: Optional[str] = Field(None, max_length=64,
        description="Optional domain hint (e.g. 'legal', 'trading', 'medical').")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The quarterly earnings report indicates a significant uptick in revenue.",
                "mode": "summary",
                "max_words": 40,
                "domain": "finance",
            }
        }
    }


class ProcessResponse(BaseModel):
    output_text: str
    word_count: int
    compression_passes: int
    was_truncated: bool
    was_pre_compressed: bool
    estimated_input_tokens: int
    model_used: Optional[str]
    backend_url: Optional[str]
    latency_ms: float
    mode: str
    max_words: int


class HealthResponse(BaseModel):
    status: str
    backends: List[dict]


class BackendStatus(BaseModel):
    url: str
    model: str
    healthy: bool
    total_requests: int
    error_rate: float
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# /analyze endpoint schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32_000, description="Raw input text to analyze.")
    mode: Mode = Field(Mode.paraphrase, description="Intended processing mode.")
    max_words: int = Field(120, ge=10, le=2000, description="Target output word budget.")
    max_tokens: Optional[int] = Field(None, ge=10, le=4096,
        description="Optional hard token budget (overrides max_words x 2 estimate).")
    domain: Optional[str] = Field(None, max_length=64, description="Domain hint.")
    target_models: Optional[List[str]] = Field(None,
        description=(
            "Subset of model keys to include in memory analysis. "
            "Defaults to all 7 registered models. "
            "Options: llama3.2:3b, phi3:mini, mistral:7b, llama3:8b, "
            "gemma2:9b, llama2:13b, llama3:70b"
        ),
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The quarterly earnings report indicates a significant uptick in revenue.",
                "mode": "summary",
                "max_words": 40,
                "domain": "finance",
            }
        }
    }


class InputStatsSchema(BaseModel):
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    token_count: int
    tokenizer: str
    avg_tokens_per_word: float
    avg_chars_per_token: float


class PromptStatsSchema(BaseModel):
    prompt_token_count: int
    prompt_overhead_tokens: int
    content_tokens: int
    max_output_tokens: int
    total_context_tokens: int


class BudgetMetricsSchema(BaseModel):
    max_words: int
    max_tokens: int
    input_word_count: int
    input_token_count: int
    word_utilization_pct: float
    needs_compression: bool
    compression_ratio_needed: float
    estimated_output_tokens: int
    will_pre_compress: bool


class VRAMFitsSchema(BaseModel):
    tier: str
    vram_gb: int
    fits: bool


class ModelMemorySchema(BaseModel):
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
    vram_fits: List[VRAMFitsSchema]


class CompressionSchema(BaseModel):
    input_tokens: int
    target_tokens: int
    ratio_needed: float
    savings_tokens: int
    savings_pct: float
    passes_estimated: int
    will_hard_truncate: bool


class AnalyzeResponse(BaseModel):
    input_stats: InputStatsSchema
    prompt_stats: PromptStatsSchema
    budget: BudgetMetricsSchema
    models: List[ModelMemorySchema]
    compression: CompressionSchema
    tokenizer_available: bool
    tokenizer_name: str
    warnings: List[str]
    recommendations: List[str]
    analysis_latency_ms: float
