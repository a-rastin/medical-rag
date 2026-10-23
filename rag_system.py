#!/usr/bin/env python
"""
RAG-Anything pipeline for the UpToDate headache PDFs in .drugs/RAG.

How it works
------------
1. Every PDF is parsed with MinerU 4 (flash tier: layout-aware text extraction)
   into a MinerU-style *content list* -- RAG-Anything's native multimodal
   document format ({"type": "text"|"image"|"table", "page_idx": N, ...}).
2. The content list is fed to RAG-Anything, which inserts the text into
   LightRAG (knowledge-graph entity/relation extraction + vector index) and
   routes any image/table/equation items to its multimodal processors.
3. Storage is file-based (rag_storage/: NetworkX graphml + NanoVectorDB +
   JSON KV) and is shared verbatim with `lightrag-server`, so the WebUI
   queries exactly the same index.

Usage
-----
  python rag_system.py ingest [--dir PATH] [--force]   # parse + index PDFs
  python rag_system.py query "question" [--mode mix]   # ask the corpus
  python rag_system.py stats                           # indexed documents
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Minimal .env loader (no external dependency)
# --------------------------------------------------------------------------
def _load_env(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
EMBED_DIM = int(os.environ.get("EMBEDDING_DIM", "2048"))
VISION_MODEL = os.environ.get("VISION_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

WORKING_DIR = Path(os.environ.get("WORKING_DIR", PROJECT_ROOT / "rag_storage"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
DEFAULT_DOCS_DIR = Path(os.environ.get("DRUGS_DIR", PROJECT_ROOT / ".drugs" / "RAG"))

SERVER_HOST = os.environ.get("HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("PORT", "9621"))

MINERU_BIN = os.environ.get("MINERU_BIN", "mineru")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=240.0, max_retries=2)

# --------------------------------------------------------------------------
# NVIDIA model functions (thinking disabled for fast, clean answers)
# --------------------------------------------------------------------------
async def nim_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LLM callable for LightRAG / RAG-Anything (returns the answer text)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    max_tokens = int(kwargs.get("max_tokens") or 2048)
    last_error = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if content:
                return content
            # Model ignored the thinking-disable switch: raise the token budget
            # so it can finish thinking and still emit the answer.
            max_tokens = min(max_tokens * 4, 8192)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"NIM chat call failed after 3 attempts: {last_error}")


async def nim_vision_func(prompt, system_prompt=None, history_messages=None,
                          image_data=None, **kwargs):
    """Vision callable for RAG-Anything's multimodal processors (best effort)."""
    content = [{"type": "text", "text": prompt}]
    if image_data:
        url = image_data if str(image_data).startswith("data:") else \
            f"data:image/jpeg;base64,{image_data}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": content})
    resp = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        max_tokens=int(kwargs.get("max_tokens") or 1024),
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


async def _nim_embed(texts):
    resp = await client.embeddings.create(model=EMBED_MODEL, input=list(texts))
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def make_embedding_func():
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=_nim_embed)


# --------------------------------------------------------------------------
# PDF -> MinerU-style content list
# --------------------------------------------------------------------------
PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+(\d+)\s*-->")
DETAILS_RE = re.compile(
    r"<details>.*?<summary>.*?</summary>\s*(.*?)\s*</details>", re.S
)
IMAGE_REF_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.M)
FURNITURE_RE = re.compile(
    r"^(Official reprint from UpToDate|www\.uptodate\.com|©\s*\d{4}\s*UpToDate"
    r"|© \d{4} UpToDate, Inc|Topic \d+ Version)"
)


def parse_pdf_with_mineru(pdf_path: Path, force: bool = False) -> str:
    """Run `mineru parse` (flash tier) and return the extracted markdown."""
    cache = OUTPUT_DIR / (pdf_path.stem + ".mineru.md")
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8", errors="ignore")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        MINERU_BIN, "parse", str(pdf_path),
        "--pages", "all", "--tier", "flash",
        "-o", str(cache),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not cache.exists():
        raise RuntimeError(
            f"mineru parse failed for {pdf_path.name}: "
            f"{(proc.stderr or proc.stdout).strip()[:400]}"
        )
    return cache.read_text(encoding="utf-8", errors="ignore")


def mineru_markdown_to_content_list(md: str) -> list:
    """Convert MinerU flash markdown into a RAG-Anything content list."""
    # Image placeholders in flash mode carry an OCR'd "image content" details
    # block: keep the text, drop the dead doc: reference.
    md = DETAILS_RE.sub(lambda m: "\n" + m.group(1) + "\n", md)
    md = IMAGE_REF_RE.sub("", md)

    parts = PAGE_RE.split(md)  # [pre, page, total, chunk, page, total, chunk, ...]
    items = []
    for i in range(1, len(parts), 3):
        page_no = int(parts[i])
        chunk = parts[i + 2]
        lines = [ln for ln in chunk.splitlines() if not FURNITURE_RE.match(ln.strip())]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if text:
            items.append({"type": "text", "text": text, "page_idx": page_no - 1})
    return items


def find_pdfs(folder: Path) -> list:
    pdfs = {p.resolve(): p for p in folder.glob("*") if p.suffix.lower() == ".pdf"}
    return sorted(pdfs.values(), key=lambda p: p.name.lower())


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
async def cmd_ingest(docs_dir: Path, force: bool) -> None:
    from raganything import RAGAnything, RAGAnythingConfig

    pdfs = find_pdfs(docs_dir)
    if not pdfs:
        print(f"No PDF files found in {docs_dir}")
        return
    print(f"Found {len(pdfs)} PDF(s) in {docs_dir}")

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser_output_dir=str(OUTPUT_DIR),
        parser="mineru",
        parse_method="auto",
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=nim_llm_func,
        vision_model_func=nim_vision_func,
        embedding_func=make_embedding_func(),
        lightrag_kwargs={
            "llm_model_name": LLM_MODEL,
            "llm_model_max_async": 6,
            "embedding_batch_num": 16,
            "embedding_func_max_async": 4,
        },
    )

    ok = failed = 0
    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        try:
            md = parse_pdf_with_mineru(pdf, force=force)
            content_list = mineru_markdown_to_content_list(md)
            print(f"  parsed: {len(content_list)} content blocks")
            if not content_list:
                print("  ! empty parse result, skipping")
                failed += 1
                continue
            await rag.insert_content_list(content_list, file_path=pdf.name)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ! FAILED: {exc}")

    await rag.finalize_storages()
    print(f"\nDone. Indexed OK: {ok}, failed: {failed}")
    print(f"Storage: {WORKING_DIR}")
    print("Start the WebUI with ./start_webui.sh -> http://localhost:9621/webui")


async def cmd_query(question: str, mode: str) -> None:
    # Prefer the running WebUI server (avoids touching files it has loaded).
    import httpx

    try:
        async with httpx.AsyncClient(timeout=180.0) as hc:
            r = await hc.post(
                f"http://{SERVER_HOST if SERVER_HOST != '0.0.0.0' else '127.0.0.1'}:{SERVER_PORT}/query",
                json={"query": question, "mode": mode},
            )
            if r.status_code == 200:
                print(r.json().get("response", "").strip())
                return
            print(f"(server responded {r.status_code}; falling back to local query)")
    except Exception:  # noqa: BLE001
        pass

    from raganything import RAGAnything, RAGAnythingConfig

    config = RAGAnythingConfig(working_dir=str(WORKING_DIR))
    rag = RAGAnything(
        config=config,
        llm_model_func=nim_llm_func,
        vision_model_func=nim_vision_func,
        embedding_func=make_embedding_func(),
    )
    answer = await rag.aquery(question, mode=mode)
    print(answer if isinstance(answer, str) else json.dumps(answer, indent=2))
    await rag.finalize_storages()


def cmd_stats() -> None:
    status_file = WORKING_DIR / "kv_store_doc_status.json"
    if not status_file.exists():
        print("No indexed documents yet. Run: python rag_system.py ingest")
        return
    docs = json.loads(status_file.read_text())
    print(f"Indexed documents: {len(docs)}  (storage: {WORKING_DIR})")
    for d in docs.values():
        name = d.get("file_path", "?")
        status = d.get("status", "?")
        chunks = d.get("chunks_count", "?")
        print(f"  [{status:>8}] {name}  ({chunks} chunks)")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-Anything ingestion & query CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="parse and index PDFs")
    p_ing.add_argument("--dir", default=str(DEFAULT_DOCS_DIR))
    p_ing.add_argument("--force", action="store_true", help="re-parse even if cached")

    p_q = sub.add_parser("query", help="query the indexed corpus")
    p_q.add_argument("question")
    p_q.add_argument("--mode", default="mix",
                     choices=["local", "global", "hybrid", "mix", "naive"])

    sub.add_parser("stats", help="show indexed documents")

    args = ap.parse_args()
    if args.cmd == "ingest":
        asyncio.run(cmd_ingest(Path(args.dir).expanduser().resolve(), args.force))
    elif args.cmd == "query":
        asyncio.run(cmd_query(args.question, args.mode))
    elif args.cmd == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
