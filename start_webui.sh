#!/usr/bin/env bash
# Start the RAG WebUI: NIM adapter proxy (9622) + LightRAG Server (9621).
set -euo pipefail
cd "$(dirname "$0")"

set -a; source .env; set +a
mkdir -p run logs
PY="$(pwd)/venv/bin"

# 1) NIM adapter proxy (disables 'thinking' on chat calls for fast WebUI answers)
if ! curl -sf "http://${NIM_PROXY_HOST}:${NIM_PROXY_PORT}/healthz" >/dev/null 2>&1; then
  nohup "$PY/python" nim_proxy.py > logs/nim_proxy.log 2>&1 &
  echo $! > run/nim_proxy.pid
  for _ in $(seq 1 40); do
    curl -sf "http://${NIM_PROXY_HOST}:${NIM_PROXY_PORT}/healthz" >/dev/null 2>&1 && break
    sleep 0.5
  done
  echo "[ok] NIM proxy   -> http://${NIM_PROXY_HOST}:${NIM_PROXY_PORT} (pid $(cat run/nim_proxy.pid))"
else
  echo "[ok] NIM proxy already running"
fi

# 2) LightRAG Server (serves the WebUI and the REST API on the shared rag_storage)
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[ok] LightRAG server already running"
else
  nohup "$PY/lightrag-server" --host "$HOST" --port "$PORT" \
    > logs/lightrag-server.log 2>&1 &
  echo $! > run/lightrag-server.pid
fi

echo -n "[..] waiting for LightRAG server "
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo " up!"
    echo
    echo "  WebUI  : http://localhost:${PORT}/webui"
    echo "  API    : http://localhost:${PORT}/docs"
    echo "  Query  : python rag_system.py query \"your question\""
    echo "  Logs   : logs/lightrag-server.log"
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo
echo "[!] LightRAG server did not become healthy in 120s. Check logs/lightrag-server.log"
exit 1
