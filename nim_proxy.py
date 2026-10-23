#!/usr/bin/env python
"""
NIM adapter proxy for LightRAG Server.

lightrag-server speaks the plain OpenAI protocol. The NVIDIA NIM endpoint is
OpenAI-compatible, but its reasoning models emit a long `reasoning_content`
stream before any answer unless "thinking" is explicitly disabled per request
— something lightrag-server cannot do. This tiny proxy injects

    chat_template_kwargs: {"enable_thinking": false}

into every chat completion request and forwards everything else (including
streaming) untouched to https://integrate.api.nvidia.com/v1.

Run:  python nim_proxy.py     (listens on NIM_PROXY_HOST:NIM_PROXY_PORT, default 127.0.0.1:9622)
"""

import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

NVIDIA_BASE = os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_KEY = os.environ.get("OPENAI_API_KEY", "")
DISABLE_THINKING = os.environ.get("NIM_DISABLE_THINKING", "true").lower() == "true"
HOST = os.environ.get("NIM_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("NIM_PROXY_PORT", "9622"))

app = FastAPI(title="NIM adapter for LightRAG", docs_url=None, redoc_url=None)

client = httpx.AsyncClient(
    base_url=NVIDIA_BASE,
    headers={"Authorization": f"Bearer {NVIDIA_KEY}"},
    timeout=httpx.Timeout(connect=15.0, read=600.0, write=120.0, pool=15.0),
    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "upstream": NVIDIA_BASE}


@app.get("/v1/models")
async def models():
    r = await client.get("/models")
    return JSONResponse(status_code=r.status_code, content=_json(r))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if DISABLE_THINKING:
        ctk = body.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            ctk.setdefault("enable_thinking", False)
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}

    if body.get("stream"):
        req = client.build_request("POST", "/chat/completions", json=body)
        upstream = await client.send(req, stream=True)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(upstream.aclose),
        )

    r = await client.post("/chat/completions", json=body)
    return JSONResponse(status_code=r.status_code, content=_json(r))


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    # NIM does not accept these OpenAI extras
    body.pop("encoding_format", None)
    body.pop("dimensions", None)
    r = await client.post("/embeddings", json=body)
    return JSONResponse(status_code=r.status_code, content=_json(r))


def _json(r: httpx.Response):
    try:
        return r.json()
    except Exception:
        return {"error": {"message": r.text[:500]}}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
