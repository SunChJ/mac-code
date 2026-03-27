#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
BIND_HOST="${LLAMA_BIND_HOST:-127.0.0.1}"
CTX="${CTX:-12288}"
THREADS="${THREADS:-4}"
MODEL="${MODEL:-$HOME/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo
  echo "Download it with:"
  echo "  $ROOT_DIR/download-qwen35b.sh"
  exit 1
fi

echo "Starting llama-server with:"
echo "  model: $MODEL"
echo "  ctx:   $CTX"
echo "  host:  $BIND_HOST:$PORT"
echo
echo "When the server is ready, run:"
echo "  cd \"$ROOT_DIR\" && LLAMA_URL=\"http://$BIND_HOST:$PORT\" python3 agent.py"
echo

exec llama-server \
  --model "$MODEL" \
  --alias "Qwen3.5-35B-A3B-UD-IQ2_M" \
  --port "$PORT" \
  --host "$BIND_HOST" \
  --flash-attn on \
  --ctx-size "$CTX" \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --n-gpu-layers 99 \
  --reasoning off \
  -np 1 \
  -t "$THREADS"
