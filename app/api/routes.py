"""
app/api/routes.py
All API endpoints.
"""

from __future__ import annotations

import time
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from app.api.schemas import (
    ProcessRequest,
    ProcessResponse,
    HealthResponse,
    BackendStatus,
    AnalyzeRequest,
    AnalyzeResponse,
    InputStatsSchema,
    PromptStatsSchema,
    BudgetMetricsSchema,
    ModelMemorySchema,
    VRAMFitsSchema,
    CompressionSchema,
)
from app.core.backend_pool import BackendPool
from app.core.preprocessor import Preprocessor
from app.core.postprocessor import PostProcessor
from app.core.config import settings
from app.core.token_analyzer import TokenAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    if not settings.API_KEY:
        return
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


_preprocessor = Preprocessor()
_analyzer = TokenAnalyzer()


def get_pool(request: Request) -> BackendPool:
    return request.app.state.backend_pool


@router.post(
    "/process",
    response_model=ProcessResponse,
    summary="Process text",
    description=(
        "Accepts any text, applies the requested mode (paraphrase / summary / "
        "keywords / etc.), enforces the word budget, and returns the result."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def process_text(body: ProcessRequest, pool: BackendPool = Depends(get_pool)):
    t0 = time.perf_counter()

    pre = _preprocessor.process(
        raw_text=body.text,
        mode=body.mode.value,
        max_words=body.max_words,
        domain=body.domain,
    )

    max_tokens = body.max_tokens or body.max_words * 2

    try:
        llm_result = await pool.complete(pre["prompt"], max_tokens)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {exc}")

    pp = PostProcessor(llm_complete_fn=pool.complete)
    result = await pp.enforce(
        raw_output=llm_result["text"],
        max_words=body.max_words,
        max_tokens=max_tokens,
        meta={
            "model_used":  llm_result.get("model"),
            "backend_url": llm_result.get("backend_url"),
        },
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "mode=%s words=%d/%d passes=%d truncated=%s latency=%.0fms",
        body.mode.value, result["word_count"], body.max_words,
        result["compression_passes"], result["was_truncated"], latency_ms,
    )

    return ProcessResponse(
        output_text=result["output_text"],
        word_count=result["word_count"],
        compression_passes=result["compression_passes"],
        was_truncated=result["was_truncated"],
        was_pre_compressed=pre["was_pre_compressed"],
        estimated_input_tokens=pre["estimated_input_tokens"],
        model_used=result.get("model_used"),
        backend_url=result.get("backend_url"),
        latency_ms=latency_ms,
        mode=body.mode.value,
        max_words=body.max_words,
    )


@router.get(
    "/backends",
    response_model=List[BackendStatus],
    summary="Backend pool status",
    dependencies=[Depends(verify_api_key)],
)
async def backend_status(pool: BackendPool = Depends(get_pool)):
    return pool.status()


@router.get("/modes", summary="Available processing modes")
async def list_modes():
    from app.core.preprocessor import MODE_INSTRUCTIONS
    return {
        "modes": [
            {"name": k, "description": v}
            for k, v in MODE_INSTRUCTIONS.items()
        ]
    }


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(pool: BackendPool = Depends(get_pool)):
    return {"status": "ok", "backends": pool.status()}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Token & memory analysis",
    description=(
        "Instant local analysis of your text against the word/token budget and "
        "7 open-source LLM models. Returns token counts, GPU/RAM memory footprints, "
        "KV-cache sizes, compression projections, and VRAM fit flags. "
        "No LLM call required."
    ),
    tags=["tools"],
)
async def analyze_text(body: AnalyzeRequest):
    t0 = time.perf_counter()

    result = _analyzer.analyze(
        text=body.text,
        mode=body.mode.value,
        max_words=body.max_words,
        max_tokens=body.max_tokens,
        domain=body.domain,
        target_models=body.target_models,
    )

    def _map_model(m) -> ModelMemorySchema:
        vram_list = [
            VRAMFitsSchema(tier=tier, vram_gb=limit, fits=fits)
            for tier, (limit, fits) in zip(
                m.vram_fits.keys(),
                [(v, m.vram_fits[k]) for k, v in [
                    ("4GB  (GTX 1650 / M1 base)", 4),
                    ("8GB  (RTX 3070 / M2 base)", 8),
                    ("16GB (RTX 4080 / M2 Pro)",  16),
                    ("24GB (RTX 4090 / M3 Max)",  24),
                    ("48GB (A6000 / M2 Ultra)",   48),
                    ("80GB (A100 SXM)",            80),
                ] if k in m.vram_fits]
            )
        ]
        return ModelMemorySchema(
            model_key=m.model_key,
            label=m.label,
            params_b=m.params_b,
            context_window=m.context_window,
            weight_fp16_gb=m.weight_fp16_gb,
            weight_q8_gb=m.weight_q8_gb,
            weight_q4_gb=m.weight_q4_gb,
            kv_cache_mb=m.kv_cache_mb,
            kv_cache_gb=m.kv_cache_gb,
            total_fp16_gb=m.total_fp16_gb,
            total_q4_gb=m.total_q4_gb,
            context_used_tokens=m.context_used_tokens,
            context_window_tokens=m.context_window_tokens,
            context_utilization_pct=m.context_utilization_pct,
            context_overflow=m.context_overflow,
            vram_fits=vram_list,
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    i = result.input_stats
    p = result.prompt_stats
    b = result.budget
    c = result.compression

    return AnalyzeResponse(
        input_stats=InputStatsSchema(
            char_count=i.char_count,
            word_count=i.word_count,
            sentence_count=i.sentence_count,
            paragraph_count=i.paragraph_count,
            token_count=i.token_count,
            tokenizer=i.tokenizer,
            avg_tokens_per_word=i.avg_tokens_per_word,
            avg_chars_per_token=i.avg_chars_per_token,
        ),
        prompt_stats=PromptStatsSchema(
            prompt_token_count=p.prompt_token_count,
            prompt_overhead_tokens=p.prompt_overhead_tokens,
            content_tokens=p.content_tokens,
            max_output_tokens=p.max_output_tokens,
            total_context_tokens=p.total_context_tokens,
        ),
        budget=BudgetMetricsSchema(
            max_words=b.max_words,
            max_tokens=b.max_tokens,
            input_word_count=b.input_word_count,
            input_token_count=b.input_token_count,
            word_utilization_pct=b.word_utilization_pct,
            needs_compression=b.needs_compression,
            compression_ratio_needed=b.compression_ratio_needed,
            estimated_output_tokens=b.estimated_output_tokens,
            will_pre_compress=b.will_pre_compress,
        ),
        models=[_map_model(m) for m in result.models],
        compression=CompressionSchema(
            input_tokens=c.input_tokens,
            target_tokens=c.target_tokens,
            ratio_needed=c.ratio_needed,
            savings_tokens=c.savings_tokens,
            savings_pct=c.savings_pct,
            passes_estimated=c.passes_estimated,
            will_hard_truncate=c.will_hard_truncate,
        ),
        tokenizer_available=result.tokenizer_available,
        tokenizer_name=result.tokenizer_name,
        warnings=result.warnings,
        recommendations=result.recommendations,
        analysis_latency_ms=latency_ms,
    )
