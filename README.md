# RAG-Anything + LightRAG WebUI

A multimodal RAG system ([HKUDS/RAG-Anything](https://github.com/hkuds/rag-anything))
over the UpToDate headache PDFs in `.drugs/RAG`, served through the
[LightRAG Server](https://github.com/HKUDS/LightRAG) WebUI.

```
.drugs/RAG/*.PDF
   │  mineru parse --tier flash        (layout-aware PDF → markdown, cached in output/)
   ▼
RAG-Anything content list              ({"type":"text","page_idx":N,...})
   │  insert_content_list()
   ▼
LightRAG  ── entity/relation KG (NetworkX) + vector index (NanoVectorDB) + KV
   │                     all file-based, in ./rag_storage  (shared with the server)
   ├── CLI    : python rag_system.py query "..."
   └── Server : lightrag-server  →  WebUI http://localhost:9621/webui
                     │
                     └── nim_proxy.py (127.0.0.1:9622)
                              injects {"enable_thinking": false} and forwards
                              to NVIDIA NIM (OpenAI-compatible)
```

## Models (NVIDIA NIM — verified against this account)

| Role | Model |
|---|---|
| LLM (extraction + answers) | `nvidia/nemotron-3.5-lightning-30b-a3b` (thinking disabled → ~1.3 s, clean JSON) |
| Embeddings (2048-dim) | `nvidia/nemotron-3-embed-1b` |
| Vision (image captioning, optional) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |

Many other models in the account's catalog return 404 (not deployed); the
above were probed and work. All configuration lives in `.env`.

## Quick start

```bash
# 1) Parse + index the PDFs (cached; re-run is incremental & deduped)
venv/bin/python rag_system.py ingest

# 2) Start the WebUI (proxy + server) and open it
./start_webui.sh
#    WebUI : http://localhost:9621/webui
#    API   : http://localhost:9621/docs

# 3) Ask questions
venv/bin/python rag_system.py query "What first-line drugs for acute migraine?"
venv/bin/python rag_system.py stats          # what's indexed
./stop_webui.sh                              # stop everything
```

Or upload/query directly in the WebUI — Documents tab lists the 6 indexed
PDFs, Retrieval tab supports `local / global / hybrid / mix / naive` modes,
and the Knowledge Graph tab renders the extracted entities.

## Files

| File | Purpose |
|---|---|
| `rag_system.py` | Ingestion + query CLI (RAG-Anything pipeline) |
| `nim_proxy.py` | NIM adapter for lightrag-server (disables reasoning mode) |
| `start_webui.sh` / `stop_webui.sh` | WebUI lifecycle |
| `.env` | Keys, models, ports, paths |
| `rag_storage/` | LightRAG index (graphml + vectors + KV) — delete to rebuild |
| `output/` | Cached MinerU parses (`*.mineru.md`) |
| `logs/`, `run/` | Server logs and pid files |

## Notes

- **Parsing**: RAG-Anything 1.4.3 calls the legacy `mineru -p` CLI, which no
  longer exists in the installed MinerU 4.x. We therefore parse with
  `mineru parse --tier flash` and feed RAG-Anything its native content-list
  format via `insert_content_list()` — same downstream pipeline (KG +
  embeddings + multimodal processors). The flash tier is text/layout only
  (no OCR engine models installed); the 1-page "cannot-miss" reprints are
  pure text so nothing is lost.
- **Re-ingesting**: `insert_content_list` is content-hash deduped; use
  `--force` to re-parse PDFs with MinerU. After ingesting while the server
  is running, restart it (`./stop_webui.sh && ./start_webui.sh`) so it
  reloads the graph from disk.
- **Concurrency**: the CLI prefers the server's `/query` endpoint when the
  WebUI is up, so it never writes to the index files the server has loaded.
- **Security**: the server binds `0.0.0.0:9621` without auth for local use.
  Put it behind a reverse proxy or set `AUTH_ACCOUNTS` / `LIGHTRAG_API_KEY`
  in `.env` before exposing it.
