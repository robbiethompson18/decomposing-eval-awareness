"""Async parallel-query helper for sync target clients (duck-typed .query_chat())."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


async def query_parallel(
    items: list[tuple[Any, Any, dict[str, Any]]],
    result_fn: Callable[[str, dict[str, Any], int], dict[str, Any]],
    max_concurrent: int = 100,
) -> list[dict[str, Any]]:
    """
    Run parallel queries to target models.

    Args:
        items: List of (client, chat, metadata) tuples, where chat is a Chat
            (lib.chat.Chat) passed to client.query_chat().
        result_fn: Function (response, metadata, sample_idx) -> result dict
        max_concurrent: Max concurrent requests

    Returns:
        List of results in original order
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(items)
    current = [0]
    loop = asyncio.get_running_loop()
    # asyncio.to_thread uses the default executor, capped at min(32, cpus + 4)
    # threads — a dedicated executor lets max_concurrent actually be reached.
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def query_one(
        idx: int, client: Any, chat: Any, metadata: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            response = await loop.run_in_executor(executor, client.query_chat, chat)
            current[0] += 1
            sample_idx = metadata.get("sample_idx", 1)
            item_id = metadata.get("id", f"item_{idx}")
            print(f"  [{current[0]}/{total}] {item_id} sample {sample_idx}")
            return idx, result_fn(response, metadata, sample_idx)

    tasks = [
        query_one(i, client, chat, meta)
        for i, (client, chat, meta) in enumerate(items)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown(wait=False)

    # Collect valid results and sort by original index
    valid = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        else:
            valid.append(r)

    valid.sort(key=lambda x: x[0])
    return [r[1] for r in valid]
