#!/bin/zsh
set -euo pipefail

REPO="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
DEST="${DEST:-$HOME/models}"
QUANT="${1:-Q3_K_S}"

case "$QUANT" in
  Q2_K|Q3_K_S|Q3_K_M|Q4_K_S|Q4_K_M|Q8_0)
    FILE="Qwen3.5-27B.${QUANT}.gguf"
    ;;
  *)
    echo "Unsupported quant: $QUANT"
    echo "Supported: Q2_K Q3_K_S Q3_K_M Q4_K_S Q4_K_M Q8_0"
    exit 1
    ;;
esac

mkdir -p "$DEST"
exec hf download "$REPO" "$FILE" --local-dir "$DEST"
