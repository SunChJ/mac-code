#!/bin/zsh
set -euo pipefail

REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
FILE="Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
DEST="${DEST:-$HOME/models}"

mkdir -p "$DEST"
exec hf download "$REPO" "$FILE" --local-dir "$DEST"
