#!/usr/bin/env bash
# Local Ollama for the TrustRAG demo, installed in the user's home folder (no sudo).
#   scripts/ollama_local.sh setup   download the CPU-only Ollama build and the model
#   scripts/ollama_local.sh serve   start the server on 127.0.0.1:11434 (foreground)
set -euo pipefail
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/.local/ollama}"
MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
BIN="$OLLAMA_HOME/bin/ollama"

case "${1:-}" in
  setup)
    if [ ! -x "$BIN" ]; then
      mkdir -p "$OLLAMA_HOME"
      echo "Downloading Ollama (CPU-only files are kept, GPU libraries are skipped)..."
      curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -dc \
        | tar -x -C "$OLLAMA_HOME" --exclude='*cuda*' --exclude='*rocm*' --exclude='*vulkan*' --exclude='*mlx*'
    fi
    "$BIN" --version || true
    if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null; then
      OLLAMA_HOST=127.0.0.1:11434 "$BIN" serve >/tmp/ollama-setup.log 2>&1 &
      until curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done
    fi
    "$BIN" pull "$MODEL"
    echo "Ready. Start it later with: make ollama"
    ;;
  serve)
    [ -x "$BIN" ] || { echo "Ollama is not installed; run: make ollama-setup" >&2; exit 1; }
    exec env OLLAMA_HOST=127.0.0.1:11434 "$BIN" serve
    ;;
  *)
    echo "usage: $0 setup|serve" >&2; exit 2 ;;
esac
