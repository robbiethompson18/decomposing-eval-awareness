"""lib.retry.post_with_retry: retries 429/5xx and connection errors, returns the
last response once retries are exhausted, passes non-retryable statuses through.

Run with:  uvx --with pytest pytest tests/
"""

import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import retry


class FakeResponse:
    def __init__(self, status: int, retry_after: str | None = None):
        self.status_code = status
        self.headers = {"retry-after": retry_after} if retry_after else {}


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    return sleeps


def _script(monkeypatch, outcomes):
    """Make requests.post yield each outcome in turn (an Exception is raised)."""
    calls: list[dict] = []
    it = iter(outcomes)

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        out = next(it)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(retry.requests, "post", fake_post)
    return calls


def test_success_first_try(monkeypatch, no_sleep):
    calls = _script(monkeypatch, [FakeResponse(200)])
    assert retry.post_with_retry("u", json={"a": 1}).status_code == 200
    assert len(calls) == 1 and calls[0]["json"] == {"a": 1}
    assert no_sleep == []


def test_retries_429_then_succeeds(monkeypatch, no_sleep):
    calls = _script(
        monkeypatch, [FakeResponse(429), FakeResponse(529), FakeResponse(200)]
    )
    assert retry.post_with_retry("u").status_code == 200
    assert len(calls) == 3
    assert len(no_sleep) == 2


def test_honors_retry_after(monkeypatch, no_sleep):
    _script(monkeypatch, [FakeResponse(429, retry_after="7"), FakeResponse(200)])
    retry.post_with_retry("u")
    assert no_sleep == [7.0]


def test_non_retryable_status_passes_through(monkeypatch, no_sleep):
    calls = _script(monkeypatch, [FakeResponse(400)])
    assert retry.post_with_retry("u").status_code == 400
    assert len(calls) == 1 and no_sleep == []


def test_exhausted_returns_last_response(monkeypatch, no_sleep):
    n = retry.MAX_RETRIES + 1
    calls = _script(monkeypatch, [FakeResponse(429)] * n + [FakeResponse(200)])
    assert retry.post_with_retry("u").status_code == 429
    assert len(calls) == n  # never reaches the 200
    assert len(no_sleep) == retry.MAX_RETRIES


def test_connection_error_retried_then_raised(monkeypatch, no_sleep):
    err = requests.ConnectionError("boom")
    _script(monkeypatch, [err, FakeResponse(200)])
    assert retry.post_with_retry("u").status_code == 200

    _script(monkeypatch, [err] * (retry.MAX_RETRIES + 1))
    with pytest.raises(requests.ConnectionError):
        retry.post_with_retry("u")


def test_backoff_grows_and_caps():
    a = retry.backoff_seconds(0)
    b = retry.backoff_seconds(3)
    assert 0.25 <= a <= 0.75
    assert 2.0 <= b <= 6.0
    assert retry.backoff_seconds(20) <= retry._BACKOFF_CAP_S * 1.5
    assert retry.backoff_seconds(0, "9999") == retry._RETRY_AFTER_CAP_S
    assert retry.backoff_seconds(0, "garbage") <= 0.75
