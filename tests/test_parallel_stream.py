"""query_parallel streams each result to on_result as it completes, so a caller
can persist progress without waiting on the slowest call in the batch."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.parallel import query_parallel


class SlowClient:
    """query_chat sleeps for the number of seconds encoded in the chat."""

    def query_chat(self, chat: dict) -> str:
        time.sleep(chat["delay"])
        return f"done after {chat['delay']}"


def test_on_result_fires_before_stragglers_finish():
    client = SlowClient()
    items = [
        (client, {"delay": 0.6}, {"id": "slow", "sample_idx": 1}),
        (client, {"delay": 0.05}, {"id": "fast-a", "sample_idx": 1}),
        (client, {"delay": 0.05}, {"id": "fast-b", "sample_idx": 1}),
    ]
    arrivals: list[tuple[str, float]] = []
    t0 = time.monotonic()

    def build(response, meta, sample_idx):
        return {"id": meta["id"], "response": response}

    results = asyncio.run(
        query_parallel(
            items,
            build,
            max_concurrent=3,
            on_result=lambda r: arrivals.append((r["id"], time.monotonic() - t0)),
        )
    )
    # Return value keeps submission order; the stream did not.
    assert [r["id"] for r in results] == ["slow", "fast-a", "fast-b"]
    assert {a for a, _ in arrivals[:2]} == {"fast-a", "fast-b"}
    assert arrivals[-1][0] == "slow"
    fast_done = max(t for a, t in arrivals if a != "slow")
    assert fast_done < 0.4  # streamed well before the 0.6 s straggler landed


def test_on_result_is_optional():
    client = SlowClient()
    items = [(client, {"delay": 0.0}, {"id": "x", "sample_idx": 2})]
    results = asyncio.run(
        query_parallel(items, lambda r, m, s: {"id": m["id"], "s": s}, 1)
    )
    assert results == [{"id": "x", "s": 2}]
