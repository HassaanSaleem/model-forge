import asyncio
import itertools

import pytest

from model_forge.summarizer.pool import ModelPool


class Stub:
    ids = itertools.count()

    def __init__(self):
        self.id = next(Stub.ids)


def test_pool_rejects_zero_size():
    with pytest.raises(ValueError):
        ModelPool(0, Stub)


async def test_round_robin_hands_out_distinct_replicas():
    pool = ModelPool(3, Stub)
    seen = set()
    for _ in range(3):
        model = await pool.acquire()
        seen.add(model.id)
        pool.release()
    assert len(seen) == 3


async def test_semaphore_caps_concurrency():
    pool = ModelPool(2, Stub)
    await pool.acquire()
    await pool.acquire()
    assert not await pool.try_acquire()
    pool.release()
    assert await pool.try_acquire()
    pool.release()


async def test_lease_releases_on_exit():
    pool = ModelPool(1, Stub)
    async with pool.lease():
        assert not await pool.try_acquire()
    assert await pool.try_acquire()


async def test_lease_releases_on_error():
    pool = ModelPool(1, Stub)
    with pytest.raises(RuntimeError):
        async with pool.lease():
            raise RuntimeError("inference blew up")
    assert await pool.try_acquire()


async def test_waiters_queue_until_release():
    pool = ModelPool(1, Stub)
    first = await pool.acquire()

    async def waiter():
        model = await pool.acquire()
        pool.release()
        return model.id

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    assert not task.done()  # blocked behind the lease
    pool.release()
    # a size-1 pool cycles back to its only replica once the lease frees it
    assert await asyncio.wait_for(task, 1) == first.id
