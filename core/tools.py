"""Executors for the scenario tools a Chat can offer (F12).

`run_tool(name, args)` is what a target client calls when the model invokes a
scenario tool; the ToolSpec in the Chat only describes the tool, this module
makes it do something. Today there is one: `web_search`, backed by Serper
(Google SERP as JSON) and rendered the way a chat product shows results.

Every query is cached as its own file under results/eval_bench/search_cache/
(sha1(query).json, atomic write), so re-runs and the self-report replay never
hit Serper twice for the same query, 500-thread runs across several checkouts
share one cache without locking, and a run is reproducible after the live
index moves on. Live results drift, so a cache entry records when it was
fetched; delete the directory to refresh.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.retry import post_with_retry

WEB_SEARCH = "web_search"
SERPER_URL = "https://google.serper.dev/search"
MAX_RESULTS = 8  # what a product shows above the fold
CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "results" / "eval_bench" / "search_cache"
)

# What the model sees when the search backend fails (network, key, 5xx after
# retries). The same text the stub returned during the F12 pilot, so a broken
# backend degrades to that condition instead of crashing the run.
SEARCH_ERROR = "Error 503: search service temporarily unavailable."


def run_tool(name: str, args: dict) -> str:
    """Execute one scenario tool call and return the text the model gets back."""
    if name == WEB_SEARCH:
        query = str(args.get("query") or "").strip()
        if not query:
            return "Error: web_search requires a non-empty `query`."
        return web_search(query)
    return f"Error: unknown tool {name!r}."


def web_search(query: str) -> str:
    """Rendered Google results for `query`, from cache or Serper."""
    try:
        return format_results(query, fetch_serper(query))
    except Exception as e:  # noqa: BLE001 -- a search outage must not crash a 500-thread run
        print(f"    [web_search] {query!r}: {e}")
        return SEARCH_ERROR


def fetch_serper(query: str) -> dict[str, Any]:
    """Raw Serper payload for `query`, cached per query on disk."""
    path = CACHE_DIR / (hashlib.sha1(query.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())["payload"]
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set")
    resp = post_with_retry(
        SERPER_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": 10},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "query": query,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "payload": payload,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    tmp.replace(path)  # atomic, so a concurrent reader never sees a partial file
    return payload


def format_results(query: str, payload: dict[str, Any]) -> str:
    """Render a Serper payload like a search results page: answer box /
    knowledge panel first (when Google shows one), then numbered organic
    results with title, URL, snippet."""
    lines: list[str] = []
    answer = payload.get("answerBox") or {}
    if answer.get("answer") or answer.get("snippet"):
        lines.append(f"Featured: {answer.get('answer') or answer.get('snippet')}")
        if answer.get("link"):
            lines.append(f"  {answer['link']}")
    kg = payload.get("knowledgeGraph") or {}
    if kg.get("title"):
        head = kg["title"] + (f" — {kg['type']}" if kg.get("type") else "")
        lines.append(head)
        if kg.get("description"):
            lines.append(f"  {kg['description']}")
    organic = payload.get("organic") or []
    if not organic and not lines:
        return f'No results found for "{query}".'
    for i, r in enumerate(organic[:MAX_RESULTS], 1):
        lines.append(f"{i}. {r.get('title', '').strip()}")
        if r.get("link"):
            lines.append(f"   {r['link']}")
        if r.get("snippet"):
            date = f"{r['date']} — " if r.get("date") else ""
            lines.append(f"   {date}{r['snippet'].strip()}")
    return "\n".join(lines)
