#!/usr/bin/env sh
# scripts/entrypoint.sh
# ---------------------------------------------------------------------------
# Startup sequence:
#   1. Wait for Ollama API to be ready (up to 120s)
#   2. Pull the configured model if not already cached
#   3. Start uvicorn
#
# Environment variables (set in docker-compose / deployment):
#   OLLAMA_HOST          - Ollama base URL  (default: http://ollama:11434)
#   OLLAMA_MODEL         - Model to pull    (default: llama3.2:3b)
#   SKIP_MODEL_PULL      - Set to "true" to skip auto-pull (e.g. air-gapped)
#   WORKERS              - Uvicorn worker count (default: 2)
# ---------------------------------------------------------------------------
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
SKIP_MODEL_PULL="${SKIP_MODEL_PULL:-false}"
WORKERS="${WORKERS:-2}"
MAX_WAIT=120   # seconds

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         llm-balance-paraphraser  starting up         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Ollama URL : $OLLAMA_HOST"
echo "  Model      : $OLLAMA_MODEL"
echo "  Workers    : $WORKERS"
echo ""

# ── 1. Wait for Ollama ───────────────────────────────────────────────────────
if [ "$SKIP_MODEL_PULL" != "true" ]; then
    echo "[ 1/3 ] Waiting for Ollama to be ready..."
    elapsed=0
    until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
        if [ "$elapsed" -ge "$MAX_WAIT" ]; then
            echo "  ✗ Ollama did not become ready within ${MAX_WAIT}s."
            echo "    Continuing anyway — /process will return 503 until Ollama is up."
            break
        fi
        printf "  ... waiting (%ds)\r" "$elapsed"
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "  ✓ Ollama is ready."

    # ── 2. Pull model if missing ─────────────────────────────────────────────
    echo ""
    echo "[ 2/3 ] Checking model '$OLLAMA_MODEL'..."

    # List local models and check if ours is present
    MODELS=$(curl -sf "$OLLAMA_HOST/api/tags" | grep -o '"name":"[^"]*"' | grep -o '"[^"]*"$' | tr -d '"' || echo "")

    if echo "$MODELS" | grep -q "$OLLAMA_MODEL"; then
        echo "  ✓ Model '$OLLAMA_MODEL' already cached — skipping pull."
    else
        echo "  ↓ Pulling '$OLLAMA_MODEL' (this takes a few minutes on first run)..."
        curl -sf -X POST "$OLLAMA_HOST/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$OLLAMA_MODEL\", \"stream\": false}" \
            | grep -o '"status":"[^"]*"' | tail -1 || true
        echo "  ✓ Model pull complete."
    fi
else
    echo "[ 1/3 ] SKIP_MODEL_PULL=true — skipping Ollama wait and model pull."
    echo "[ 2/3 ] Skipped."
fi

# ── 3. Start API ─────────────────────────────────────────────────────────────
echo ""
echo "[ 3/3 ] Starting uvicorn (workers=$WORKERS)..."
echo ""

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "$WORKERS" \
    --log-level info \
    --access-log
