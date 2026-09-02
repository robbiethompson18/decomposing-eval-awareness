"""core/tools.py: the live web_search executor (Serper) — rendering, per-query
cache, and failure behavior. No network: Serper is faked."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import tools
from core.tools import SEARCH_ERROR, format_results, run_tool, web_search

PAYLOAD = {
    "organic": [
        {
            "title": "KRUGER SAFARI COLLECTION - Hotel Reviews",
            "link": "https://www.tripadvisor.com/x",
            "snippet": "Kruger Safari Collection, Hoedspruit: See traveler reviews",
            "position": 1,
        },
        {
            "title": "Drive from Hoedspruit",
            "link": "https://www.facebook.com/y",
            "snippet": "How's the drive from hoedspruit to Paul Kruger gate?",
            "date": "10 months ago",
            "position": 2,
        },
    ]
}


def test_format_results_reads_like_a_results_page():
    text = format_results("kruger outfitters", PAYLOAD)
    assert text.splitlines() == [
        "1. KRUGER SAFARI COLLECTION - Hotel Reviews",
        "   https://www.tripadvisor.com/x",
        "   Kruger Safari Collection, Hoedspruit: See traveler reviews",
        "2. Drive from Hoedspruit",
        "   https://www.facebook.com/y",
        "   10 months ago — How's the drive from hoedspruit to Paul Kruger gate?",
    ]


def test_format_results_shows_answer_box_and_knowledge_panel_first():
    payload = {
        "answerBox": {"answer": "42", "link": "https://a"},
        "knowledgeGraph": {
            "title": "Intoxalock",
            "type": "Company",
            "description": "Ignition interlocks",
        },
        "organic": PAYLOAD["organic"][:1],
    }
    lines = format_results("q", payload).splitlines()
    assert lines[:4] == [
        "Featured: 42",
        "  https://a",
        "Intoxalock — Company",
        "  Ignition interlocks",
    ]
    assert lines[4].startswith("1. ")


def test_format_results_caps_at_max_results_and_handles_empty():
    many = {
        "organic": [
            {"title": f"t{i}", "link": f"https://{i}", "snippet": "s"}
            for i in range(20)
        ]
    }
    assert format_results("q", many).count("\n   https://") == tools.MAX_RESULTS
    assert (
        format_results("nothing here", {"organic": []})
        == 'No results found for "nothing here".'
    )


def test_fetch_serper_caches_per_query_and_never_refetches(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    calls: list[dict] = []

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return PAYLOAD

    def fake_post(url, *, headers, json, timeout):
        calls.append(json)
        return _Resp()

    monkeypatch.setattr(tools, "post_with_retry", fake_post)
    first = web_search("kruger outfitters")
    second = web_search("kruger outfitters")
    assert first == second == format_results("kruger outfitters", PAYLOAD)
    assert calls == [{"q": "kruger outfitters", "num": 10}]  # one live call
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["query"] == "kruger outfitters" and cached["payload"] == PAYLOAD
    assert "fetched_at" in cached


def test_search_failure_degrades_to_the_error_string(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert (
        web_search("anything") == SEARCH_ERROR
    )  # no key -> RuntimeError -> error text
    assert not list(tmp_path.glob("*.json"))  # failures are not cached


def test_run_tool_validates_name_and_query(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda q: f"results for {q}")
    assert run_tool("web_search", {"query": " cats "}) == "results for cats"
    assert run_tool("web_search", {}).startswith("Error: web_search requires")
    assert run_tool("run_code", {"code": "x"}).startswith("Error: unknown tool")
