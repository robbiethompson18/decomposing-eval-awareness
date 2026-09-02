"""Async parallel-query helper for sync target clients (duck-typed .query_chat())."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


async def query_parallel(
    items: list[tuple[Any, Any, dict[str, Any]]],
    result_fn: Callable[[str, dict[str, Any], int], dict[str, Any]],
    max_concurrent: int = 500,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Run parallel queries to target models.

    Args:
        items: List of (client, chat, metadata) tuples, where chat is a Chat
            (lib.chat.Chat) passed to client.query_chat_meta() when the client
            has one (OpenRouter: returns (text, meta) with the serving
            provider), else client.query_chat(). The per-call meta is passed
            to result_fn as metadata["call_meta"] ({} for clients without it).
        result_fn: Function (response, metadata, sample_idx) -> result dict
        max_concurrent: Max concurrent requests
        on_result: Optional callback invoked with each result dict as soon as
            that call completes (on the event loop, so never concurrently).
            Lets the caller persist progress without waiting for stragglers.

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

    def call(client: Any, chat: Any) -> tuple[str, dict[str, Any]]:
        if hasattr(client, "query_chat_meta"):
            return client.query_chat_meta(chat)
        return client.query_chat(chat), {}

    async def query_one(
        idx: int, client: Any, chat: Any, metadata: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            response, call_meta = await loop.run_in_executor(
                executor, call, client, chat
            )
            metadata = {**metadata, "call_meta": call_meta}
            current[0] += 1
            sample_idx = metadata.get("sample_idx", 1)
            item_id = metadata.get("id", f"item_{idx}")
            print(f"  [{current[0]}/{total}] {item_id} sample {sample_idx}")
            result = result_fn(response, metadata, sample_idx)
            if on_result is not None:
                on_result(result)
            return idx, result

    tasks = [
        query_one(i, client, chat, meta) for i, (client, chat, meta) in enumerate(items)
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
