# llm-balance-paraphraser

> **Open-source, real-time LLM token analyzer, paraphraser, and compressor.**  
> Runs 100% locally — no API keys, no subscriptions, no data leaving your machine.

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-black?logo=ollama)](https://ollama.com)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker)](https://docs.docker.com/compose)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passing-brightgreen)](#testing)

---

## What it does

| Feature | Description |
|---|---|
| 🔬 **Token Analyzer** | Instant token count, KV-cache size, GPU memory footprint for 7 OSS models — **no LLM call needed** |
| ⚡ **Text Processor** | Paraphrase, summarize, bullet-point, formalize or simplify any text with a strict word/token budget |
| ⚖️ **Load Balancer** | Weighted round-robin across multiple Ollama backends with health checks and auto-failover |
| 🗜️ **Compressor** | Multi-pass LLM compression + hard truncation fallback to always stay within budget |
| 🌐 **Web UI** | Dark-themed real-time dashboard — analyze as you type, process with one click |

---

## Quick start (Docker — recommended)

**Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
git clone https://github.com/YOUR_USERNAME/llm-balance-paraphraser.git
cd llm-balance-paraphraser

docker compose up --build
```

That's it. On first run Docker will:
1. Pull the Ollama image (~300 MB)
2. Download `llama3.2:3b` model (~2 GB) — cached permanently in a Docker volume
3. Start the API + frontend

Open **http://localhost:8000** in your browser.

> **Subsequent runs:** `docker compose up` — model is already cached, starts in seconds.

---

## Quick start (local Python)

```bash
git clone https://github.com/YOUR_USERNAME/llm-balance-paraphraser.git
cd llm-balance-paraphraser

# Install dependencies
pip install -r requirements.txt

# Install & start Ollama (https://ollama.com)
ollama pull llama3.2:3b
ollama serve &

# Start the API
uvicorn app.main:app --reload

# Open http://localhost:8000
```

---

## Architecture

```
Browser / API client
        │
        ▼
┌─────────────────────────────────┐
│       FastAPI (port 8000)        │
│                                  │
│  POST /api/v1/analyze  ◄──────── Token Analyzer (no LLM, instant)
│  POST /api/v1/process  ◄──────── Preprocessor → LLM → Postprocessor
│  GET  /api/v1/backends ◄──────── Backend pool status
│  GET  /api/v1/modes    ◄──────── Available modes
│  GET  /health                    │
│  GET  /  (frontend UI)           │
└──────────────┬──────────────────┘
               │
     ┌─────────▼──────────┐
     │   Backend Pool      │   Weighted round-robin + health checks
     │  (BackendPool)      │
     └──┬──────────────┬──┘
        │              │
   ┌────▼────┐    ┌────▼────┐
   │ Ollama  │    │ Ollama2 │   (optional multi-backend)
   │ :11434  │    │ :11435  │
   └─────────┘    └─────────┘
```

### Request pipeline (`/process`)

```
Raw text
  → Preprocessor   (normalize → estimate tokens → build prompt)
  → BackendPool    (pick healthiest backend, weighted round-robin)
  → Ollama LLM     (generate)
  → PostProcessor  (clean → check budget → re-compress if over → hard truncate)
  → Response       (word_count, compression_passes, was_truncated, latency_ms …)
```

### Token Analyzer (`/analyze`) — no LLM, sub-millisecond

```
Raw text
  → count_tokens()     (tiktoken cl100k_base, or chars÷4 heuristic)
  → prompt overhead    (instruction + delimiters + domain line)
  → per-model memory   (weights FP16/Q8/Q4 + KV cache @ FP16)
  → VRAM fit flags     (6 GPU tiers from 4 GB to 80 GB)
  → compression plan   (passes needed, hard-truncate risk)
  → Response           (full breakdown, 0.5–2 ms)
```

---

## Project structure

```
llm-balance-paraphraser/
│
├── app/
│   ├── main.py                 ← FastAPI app factory, static serving, CORS
│   ├── api/
│   │   ├── routes.py           ← All endpoints (/process, /analyze, /backends, /modes)
│   │   └── schemas.py          ← Pydantic request/response models
│   └── core/
│       ├── _tokenizer.py       ← Lazy tiktoken loader with heuristic fallback
│       ├── token_analyzer.py   ← Memory profiler: 7 models × 6 VRAM tiers
│       ├── preprocessor.py     ← Text normalization + prompt builder
│       ├── postprocessor.py    ← Budget enforcement + multi-pass compression
│       ├── backend_pool.py     ← Weighted round-robin + health checks
│       └── config.py           ← All settings via environment variables
│
├── frontend/
│   └── index.html              ← Single-file dark UI (Analyze + Process tabs)
│
├── tests/
│   ├── test_core.py            ← 16 tests: preprocessor, postprocessor
│   └── test_analyzer.py        ← 35 tests: token analyzer engine
│
├── scripts/
│   └── entrypoint.sh           ← Docker startup: wait for Ollama → pull model → serve
│
├── Dockerfile                  ← Multi-stage build (builder + slim runtime)
├── docker-compose.yml          ← App + Ollama + named volume for model cache
├── requirements.txt
├── .env.example                ← All configuration options documented
└── .gitignore                  ← Excludes .env, __pycache__, venv
```

---

## API reference

### `POST /api/v1/analyze` — Token & memory analysis (no LLM)

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "mode": "summary",
    "max_words": 80,
    "domain": "technology"
  }'
```

**Response includes:**
- `input_stats` — token count, word count, chars, sentences, avg tokens/word
- `prompt_stats` — prompt overhead, content tokens, total context tokens
- `budget` — utilisation %, needs_compression, compression ratio needed
- `models[]` — for each of 7 models: weights Q4 GB, KV cache MB, total GB, VRAM fit per tier
- `compression` — savings tokens, savings %, estimated passes, hard truncate risk
- `warnings[]` — context overflow, budget alerts
- `recommendations[]` — smallest fitting model, tiktoken install tip

### `POST /api/v1/process` — Paraphrase / summarize via LLM

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "mode": "summary",
    "max_words": 60,
    "domain": "finance"
  }'
```

**Modes:** `paraphrase` · `summary` · `bullet_points` · `keywords_only` · `formal` · `simplify` · `technical`

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server + backend pool status |
| `GET` | `/api/v1/backends` | Per-backend stats (latency, error rate) |
| `GET` | `/api/v1/modes` | Available modes with descriptions |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Configuration

All settings are environment variables (copy `.env.example` → `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND_STR` | `http://ollama:11434\|llama3.2:3b\|1` | Backend URL\|model\|weight (comma-separated for multiple) |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model to auto-pull on startup |
| `DEFAULT_MAX_WORDS` | `120` | Default output word budget |
| `COMPRESSION_PASSES` | `2` | Max LLM re-compression passes |
| `HARD_TRUNCATE` | `true` | Final fallback truncation |
| `API_KEY` | _(empty)_ | Set to enable API key auth |
| `SKIP_MODEL_PULL` | `false` | Skip auto-pull (air-gapped envs) |
| `WORKERS` | `2` | Uvicorn worker count |

### Multiple backends (load balancing)

```bash
# In docker-compose.yml or .env:
LLM_BACKEND_STR=http://ollama:11434|llama3.2:3b|2,http://ollama2:11434|mistral:7b|1
# Weight 2:1 — llama gets twice the traffic
```

### GPU acceleration

Uncomment the `deploy` block in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## Supported models (Token Analyzer)

| Model | Params | Q4 Weight | Context | Min VRAM (Q4) |
|---|---|---|---|---|
| Llama 3.2 3B | 3.21B | 1.6 GB | 128K | 4 GB |
| Phi-3 Mini | 3.82B | 1.9 GB | 128K | 4 GB |
| Mistral 7B | 7.24B | 3.6 GB | 32K | 8 GB |
| Llama 3 8B | 8.03B | 4.0 GB | 8K | 8 GB |
| Gemma 2 9B | 9.24B | 4.6 GB | 8K | 8 GB |
| Llama 2 13B | 13.02B | 6.5 GB | 4K | 8 GB |
| Llama 3 70B | 70.55B | 35.3 GB | 8K | 48 GB |

---

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
# 51 tests, 0 failures
```

Tests cover: text normalization, token estimation, prompt building, compression logic, all 7 model memory profiles, VRAM fit flags, context overflow detection, budget metrics.

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make changes + add tests
4. Run `pytest tests/` — all must pass
5. Commit: `git commit -m "feat: description"`
6. Push + open a Pull Request

Ideas welcome: new models, new modes, streaming support, metrics dashboard, OpenAI-compatible API shim.

---

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and distribute.
