"""Semaphore-bounded model pool.

Transformer inference is CPU/GPU-bound and not re-entrant, so a single model
instance serializes every request behind one generate() call. The pool holds N
independent model replicas: a semaphore admits at most N concurrent requests, a
round-robin iterator hands each admitted request a different replica, and requests
beyond N queue on the semaphore instead of piling onto a busy model. try_acquire()
exists for health checks — it probes whether capacity is available *right now*
without holding it, so a saturated pool reports unhealthy instead of hanging the
probe.
"""

from __future__ import annotations

import itertools
from asyncio import Lock, Semaphore, wait_for
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TypeVar

T = TypeVar("T")


class ModelPool:
    def __init__(self, size: int, loader: Callable[[], T]):
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.size = size
        self._semaphore = Semaphore(size)
        self._iterator_lock = Lock()
        self._replicas: list[T] = [loader() for _ in range(size)]
        self._iterator = itertools.cycle(self._replicas)

    async def try_acquire(self, timeout: float = 0.001) -> bool:
        """Probe for free capacity without keeping it."""
        try:
            await wait_for(self._semaphore.acquire(), timeout)
        except TimeoutError:
            return False
        self._semaphore.release()
        return True

    async def acquire(self) -> T:
        """Take a replica; blocks when all `size` replicas are leased."""
        await self._semaphore.acquire()
        async with self._iterator_lock:
            return next(self._iterator)

    def release(self) -> None:
        self._semaphore.release()

    @asynccontextmanager
    async def lease(self):
        """Async context manager: `async with pool.lease() as model: ...`"""
        model = await self.acquire()
        try:
            yield model
        finally:
            self.release()
