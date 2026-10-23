#!/usr/bin/env bash
# Stop the RAG WebUI (LightRAG Server + NIM proxy).
cd "$(dirname "$0")"
for f in run/lightrag-server.pid run/nim_proxy.pid; do
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill "$pid" 2>/dev/null; then
      echo "[stopped] $(basename "$f" .pid) (pid $pid)"
    fi
    rm -f "$f"
  fi
done
