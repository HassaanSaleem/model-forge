"""FastAPI serving layer for the pooled summarizer.

The event loop never runs a forward pass: each request leases a replica from the
pool (bounding concurrency), then executes the blocking generate() on a thread
pool via run_in_executor. /health_check probes the pool — a saturated pool answers
503 so an orchestrator's liveness probe sheds load instead of queueing behind it.

create_app() takes the pool as an argument, so tests inject a pool of stub models
and never download weights.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from .pool import ModelPool


class SummarizeRequest(BaseModel):
    text: str


class BatchSummarizeRequest(BaseModel):
    texts: list[str]


def create_app(pool: ModelPool, max_workers: int | None = None) -> FastAPI:
    app = FastAPI(title="log-summarizer", version="0.1.0")
    executor = ThreadPoolExecutor(max_workers=max_workers or pool.size)

    async def _run(fn, *args):
        async with pool.lease() as model:
            return await asyncio.get_event_loop().run_in_executor(executor, fn, model, *args)

    @app.post("/summarize")
    async def summarize(request: SummarizeRequest) -> dict:
        summary = await _run(lambda m, t: m.summarize(t), request.text)
        return {"summary": summary}

    @app.post("/batch_summarize")
    async def batch_summarize(request: BatchSummarizeRequest) -> dict:
        summaries = await _run(lambda m, ts: m.summarize_batch(ts), request.texts)
        return {"summaries": summaries}

    @app.get("/health_check")
    async def health_check(response: Response) -> dict:
        if await pool.try_acquire():
            return {"status": "ok", "pool_size": pool.size}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "saturated", "pool_size": pool.size}

    return app
