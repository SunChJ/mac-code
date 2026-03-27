#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
BIND_HOST="${LLAMA_BIND_HOST:-127.0.0.1}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-4}"
REASONING="${REASONING:-auto}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"

if [[ $# -gt 0 ]]; then
  MODEL="$1"
else
  for candidate in \
    "$HOME/models/Qwen3.5-27B.Q3_K_S.gguf" \
    "$HOME/models/Qwen3.5-27B.Q2_K.gguf" \
    "$HOME/models/Qwen3.5-27B.Q3_K_M.gguf" \
    "$HOME/models/Qwen3.5-27B.Q4_K_S.gguf" \
    "$HOME/models/Qwen3.5-27B.Q4_K_M.gguf" \
    "$HOME/Downloads/Qwen3.5-27B.Q3_K_S.gguf" \
    "$HOME/Downloads/Qwen3.5-27B.Q2_K.gguf" \
    "$HOME/Downloads/Qwen3.5-27B.Q3_K_M.gguf" \
    "$HOME/Downloads/Qwen3.5-27B.Q4_K_S.gguf" \
    "$HOME/Downloads/Qwen3.5-27B.Q4_K_M.gguf"
  do
    if [[ -f "$candidate" ]]; then
      MODEL="$candidate"
      break
    fi
  done
fi

if [[ -z "${MODEL:-}" || ! -f "$MODEL" ]]; then
  echo "Model not found."
  echo "Expected one of:"
  echo "  ~/models/Qwen3.5-27B.Q3_K_S.gguf"
  echo "  ~/models/Qwen3.5-27B.Q2_K.gguf"
  echo "  ~/Downloads/Qwen3.5-27B.Q3_K_S.gguf"
  echo "  ~/Downloads/Qwen3.5-27B.Q2_K.gguf"
  echo
  echo "Or pass the path explicitly:"
  echo "  ./start-qwen27b.sh /full/path/to/Qwen3.5-27B.Q3_K_S.gguf"
  exit 1
fi

MODEL_NAME="$(basename "$MODEL" .gguf)"

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
  --alias "$MODEL_NAME" \
  --port "$PORT" \
  --host "$BIND_HOST" \
  --flash-attn on \
  --ctx-size "$CTX" \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --n-gpu-layers 99 \
  --reasoning "$REASONING" \
  --reasoning-format "$REASONING_FORMAT" \
  -np 1 \
  -t "$THREADS"
