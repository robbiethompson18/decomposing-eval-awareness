"""F12 (tool affordances): the stub web_search tool, its wire adapters, and the
target clients' tool-call dispatch + trace markers.

Run with:  PYTHONPATH=.venv/lib/python3.14/site-packages uvx --python .venv/bin/python --with pytest pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.target as target_mod
from core.target import THINK_TOOL_NAME, AnthropicApiClient, _handle_call, tool_calls_in
from eval_bench.render import WEB_SEARCH_TOOL, render
from lib.chat import chat_to_transcript_text, to_gemini_tool
from run.run_eval_bench import ALL_VARIED, build_configs, load_all_tasks

TASK = load_all_tasks()[0]["template"]
CONFIGS = dict(build_configs())
STUB = WEB_SEARCH_TOOL["stub_result"]


def _cold(cfg: dict) -> dict:
    return {**cfg, "F11": "cold"}  # no opener reply needed


def test_baseline_offers_the_search_tool_and_the_f12_ablation_removes_it():
    assert CONFIGS["baseline"]["F12"] == "search"
    assert CONFIGS["F12"] == {**CONFIGS["baseline"], "F12": "none"}
    assert CONFIGS[ALL_VARIED]["F12"] == "none"
    assert render(TASK, _cold(CONFIGS["baseline"]))["tools"] == [WEB_SEARCH_TOOL]
    assert "tools" not in render(TASK, _cold(CONFIGS["F12"]))
    assert "tools" not in render(TASK, CONFIGS[ALL_VARIED])


def test_judge_transcript_lists_offered_tools():
    with_tool = chat_to_transcript_text(render(TASK, _cold(CONFIGS["baseline"])))
    assert "[tools available]\nweb_search: " in with_tool
    without = chat_to_transcript_text(render(TASK, _cold(CONFIGS["F12"])))
    assert "[tools available]" not in without


def test_gemini_adapter_uppercases_nested_schema_types():
    decl = to_gemini_tool(
        WEB_SEARCH_TOOL["name"],
        WEB_SEARCH_TOOL["description"],
        WEB_SEARCH_TOOL["parameters"],
    )
    assert decl["parameters"]["type"] == "OBJECT"
    assert decl["parameters"]["properties"]["query"]["type"] == "STRING"
    assert decl["parameters"]["required"] == ["query"]
    # The source spec is untouched (lowercase for the other three providers).
    assert WEB_SEARCH_TOOL["parameters"]["type"] == "object"


def test_handle_call_routes_think_stub_and_unknown_tools():
    chat = render(TASK, _cold(CONFIGS["baseline"]))
    thoughts: list[str] = []
    assert _handle_call(chat, THINK_TOOL_NAME, {"thought": "hmm"}, thoughts) == "noted"
    assert _handle_call(chat, "web_search", {"query": "x"}, thoughts) == STUB
    assert _handle_call(chat, "run_code", {}, thoughts).startswith(
        "Error: unknown tool"
    )
    assert thoughts[0] == "hmm"
    joined = "\n\n".join(thoughts)
    assert tool_calls_in(f"<think>{joined}</think>ok") == [
        ("web_search", {"query": "x"}),
        ("run_code", {}),
    ]


def test_anthropic_loop_answers_a_search_call_with_the_stub(monkeypatch):
    chat = render(TASK, _cold(CONFIGS["baseline"]))
    sent: list[dict] = []
    replies = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "web_search",
                    "input": {"query": "latest cve"},
                }
            ]
        },
        {"content": [{"type": "text", "text": "Couldn't search, but..."}]},
    ]

    class _Resp:
        def __init__(self, body: dict) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._body

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> _Resp:
        sent.append(json)
        return _Resp(replies[len(sent) - 1])

    monkeypatch.setattr(target_mod, "post_with_retry", fake_post)
    client = AnthropicApiClient(
        {
            "ANTHROPIC_API_KEY": "k",
            "ANTHROPIC_MODEL": "m",
            "ANTHROPIC_API_BASE": "http://x",
        }
    )
    out = client.query_chat(chat)

    assert [t["name"] for t in sent[0]["tools"]] == ["think", "web_search"]
    assert sent[1]["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": STUB}],
    }
    assert out == (
        '<think>[tool call] web_search({"query": "latest cve"})\n'
        f"[tool result] {STUB}</think>Couldn't search, but..."
    )
    assert tool_calls_in(out) == [("web_search", {"query": "latest cve"})]
