"""
llm-balance-paraphraser
=======================
Real-time LLM load-balancer, paraphraser, and token compressor.
Accepts any topic/scenario, analyzes + paraphrases via configurable
backend rules, and enforces a strict max word/token budget.
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.backend_pool import BackendPool
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("llm_balancer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise backend pool. Shutdown: clean up."""
    logger.info("Starting llm-balance-paraphraser ...")
    pool: BackendPool = app.state.backend_pool
    await pool.start_health_checks()
    yield
    logger.info("Shutting down ...")
    await pool.stop_health_checks()


def create_app() -> FastAPI:
    app = FastAPI(
        title="llm-balance-paraphraser",
        description=(
            "Open-source, real-time service for analyzing, paraphrasing, "
            "and compressing arbitrary text with strict word/token limits."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Shared state
    app.state.backend_pool = BackendPool(backends=settings.LLM_BACKENDS)

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_latency_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Latency-Ms"] = str(
            round((time.perf_counter() - start) * 1000, 2)
        )
        return response

    # Static frontend
    static_dir = Path(__file__).parent.parent / "frontend"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"message": "llm-balance-paraphraser API", "docs": "/docs"})

    @app.get("/health", tags=["system"])
    async def health(request: Request):
        pool: BackendPool = request.app.state.backend_pool
        return {"status": "ok", "backends": pool.status()}

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )

    return app


app = create_app()
