"""One retry policy for every API client in the repo.

Two kinds of client exist:

- SDK clients (`openai`, `anthropic` packages) retry internally on 408/409/429/5xx
  and connection errors, with exponential backoff that honors Retry-After. They
  just need `max_retries=MAX_RETRIES` at construction.
- Raw `requests.post` clients (Anthropic think-tool, Gemini) get the same policy
  from `post_with_retry()`.

Why it matters: at MAX_CONCURRENT=500 a provider *will* 429 in bursts. Without
retries those bursts become silent `ERROR calling ...` holes in the results file
(see .claude/notes/selfreport-vs-cot-gap.md: 285/440 OpenRouter 429s at
concurrency 100 with the SDK's default of 2 retries).
"""

import random
import time
from typing import Any

import requests

# Retries after the first attempt. 8 retries with capped exponential backoff is
# ~1-2 min of waiting per call, enough to ride out a rate-limit burst.
MAX_RETRIES = 8

# 529 = Anthropic "overloaded"; the rest are the statuses the SDKs retry on.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 30.0
_RETRY_AFTER_CAP_S = 60.0


def backoff_seconds(attempt: int, retry_after: str | None = None) -> float:
    """Sleep before retry number `attempt` (0-based). Honors a numeric
    Retry-After header, else capped exponential backoff with jitter."""
    if retry_after:
        try:
            return min(float(retry_after), _RETRY_AFTER_CAP_S)
        except ValueError:
            pass
    return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2**attempt) * random.uniform(0.5, 1.5)


def post_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float = 300,
) -> requests.Response:
    """`requests.post` that retries connection errors and RETRY_STATUSES.

    Returns the last response, retryable or not, once retries are exhausted —
    the caller decides what a non-2xx means (raise_for_status, read the error
    body, ...). Re-raises the last connection error if no response ever came
    back.
    """
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=json, timeout=timeout)
        except requests.RequestException:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(backoff_seconds(attempt))
        else:
            if resp.status_code not in RETRY_STATUSES or attempt >= MAX_RETRIES:
                return resp
            time.sleep(backoff_seconds(attempt, resp.headers.get("retry-after")))
        attempt += 1
