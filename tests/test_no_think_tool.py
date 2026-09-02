"""THINK_TOOL=0: the no-harness "arm A" condition.

Run with:  uvx --with-requirements requirements.txt pytest tests/
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import target
from core.target import (
    THINK_TOOL_MANDATE,
    AnthropicApiClient,
    GeminiApiClient,
    OpenAITargetClient,
)
from lib.chat import Chat
from run.run_eval_bench import get_output_path

CHAT: Chat = {"system": None, "messages": [{"role": "user", "content": "hi"}]}


def _cfg(**extra):
    return {
        "ANTHROPIC_API_KEY": "k",
        "ANTHROPIC_MODEL": "claude-sonnet-4-5-20250929",
        "ANTHROPIC_API_BASE": "https://example.invalid",
        "OPENAI_API_KEY": "k",
        "OPENAI_TARGET_MODEL": "gpt-5.2",
        "GEMINI_API_KEY": "k",
        "GEMINI_MODEL": "gemini-3.5-flash",
        "GEMINI_THINKING_BUDGET": 0,
        "GEMINI_FORCE_FIRST_THINK": True,
        **extra,
    }


# --- output paths ----------------------------------------------------------


def test_output_path_suffixes_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr("run.run_eval_bench.RESULTS_DIR", tmp_path)
    (tmp_path / "responses_gpt-5_think_tool_20260901_000000.json").write_text("{}")
    (tmp_path / "responses_gpt-5_no_think_tool_armA_20260901_000000.json").write_text(
        "{}"
    )
    assert get_output_path("gpt-5", "_think_tool").name.startswith(
        "responses_gpt-5_think_tool_2026"
    )
    assert get_output_path("gpt-5", "_no_think_tool_armA").name.startswith(
        "responses_gpt-5_no_think_tool_armA_2026"
    )
    # An untagged no-think-tool run must not resume into the tagged file.
    assert "armA" not in get_output_path("gpt-5", "_no_think_tool").name
    # An unsuffixed (native) run must not resume into either of them.
    assert "think_tool" not in get_output_path("gpt-5", "").name


# --- clients: request shape with the think tool off ------------------------


def test_anthropic_no_think_tool_sends_bare_request(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(json)
        return SimpleNamespace(
            status_code=200,
            headers={},
            raise_for_status=lambda: None,
            json=lambda: {"content": [{"type": "text", "text": "hello"}]},
        )

    monkeypatch.setattr(target, "post_with_retry", fake_post)
    assert AnthropicApiClient(_cfg(THINK_TOOL=False)).query_chat(CHAT) == "hello"
    assert "tools" not in seen and "system" not in seen and "output_config" not in seen

    seen.clear()
    out = AnthropicApiClient(_cfg()).query_chat(CHAT)
    assert out == "<think></think>hello"
    assert seen["system"] == THINK_TOOL_MANDATE and seen["tools"][0]["name"] == "think"


def test_openai_no_think_tool_sends_bare_request():
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(type="message", content=[SimpleNamespace(text="hello")])
        return SimpleNamespace(id="r1", output=[msg])

    client = OpenAITargetClient(_cfg(THINK_TOOL=False))
    client.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    assert client.query_chat(CHAT) == "hello"
    assert (
        "tools" not in seen and "instructions" not in seen and "reasoning" not in seen
    )


def test_gemini_no_think_tool_sends_bare_request(monkeypatch):
    seen = {}

    def fake_post(url, json, **kw):
        seen.update(json)
        return SimpleNamespace(
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        )

    monkeypatch.setattr(target, "post_with_retry", fake_post)
    assert GeminiApiClient(_cfg(THINK_TOOL=False)).query_chat(CHAT) == "hello"
    assert "tools" not in seen and "systemInstruction" not in seen
    assert "toolConfig" not in seen and "thinkingConfig" not in seen["generationConfig"]
