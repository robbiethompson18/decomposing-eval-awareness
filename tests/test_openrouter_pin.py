"""OpenRouter provider pinning: every roster model is pinned, the pin goes out
on every request with fallbacks off, and the serving provider lands in the
sample record. No network: the OpenAI client is faked."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.target as target_mod
from core.target import OpenRouterApiClient, create_target_client
from lib.config import DEFAULT_OPENROUTER_CONFIGS, OPENROUTER_PROVIDERS
from lib.parallel import query_parallel
from scripts.gen_opener_replies import ROSTER

CHAT = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}


def _fake_openai(captured: list[dict], provider: str = "Z.AI"):
    """An `openai.OpenAI`-shaped stub whose chat.completions.create records kwargs."""
    msg = SimpleNamespace(content="answer", reasoning="why", tool_calls=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        provider=provider,
        usage=None,
    )

    def create(**kwargs):
        captured.append(kwargs)
        return resp

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_every_roster_openrouter_model_is_pinned():
    roster_keys = {key for backend, key in ROSTER if backend == "openrouter"}
    assert roster_keys <= set(OPENROUTER_PROVIDERS), roster_keys - set(
        OPENROUTER_PROVIDERS
    )
    assert set(OPENROUTER_PROVIDERS) <= set(DEFAULT_OPENROUTER_CONFIGS)


def test_pin_is_sent_with_fallbacks_off_and_provider_is_returned(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        target_mod, "create_openai_client", lambda *a, **k: _fake_openai(captured)
    )
    client = OpenRouterApiClient(
        {
            "OPENROUTER_MODEL": "z-ai/glm-5.3",
            "OPENROUTER_API_KEY": "k",
            "OPENROUTER_PROVIDER": "z-ai/fp8",
        }
    )
    text, meta = client.query_chat_meta(CHAT)
    assert captured[0]["extra_body"]["provider"] == {
        "order": ["z-ai/fp8"],
        "allow_fallbacks": False,
    }
    assert captured[0]["extra_body"]["reasoning"] == {"effort": "high"}
    assert text == "<think>why</think>answer"
    assert meta["provider"] == "Z.AI"


def test_unpinned_legacy_key_sends_no_provider_block(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        target_mod, "create_openai_client", lambda *a, **k: _fake_openai(captured)
    )
    OpenRouterApiClient(
        {"OPENROUTER_MODEL": "x", "OPENROUTER_API_KEY": "k"}
    ).query_chat(CHAT)
    assert "provider" not in captured[0]["extra_body"]


def test_factory_applies_the_config_pin_and_env_override(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_MODEL_KEY", "glm-5.3")
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    monkeypatch.setattr(
        target_mod, "create_openai_client", lambda *a, **k: _fake_openai([])
    )
    client, key = create_target_client("openrouter", {"OPENROUTER_API_KEY": "k"})
    assert key == "glm-5.3" and client.provider == OPENROUTER_PROVIDERS["glm-5.3"]
    assert "provider=z-ai/fp8" in capsys.readouterr().out

    monkeypatch.setenv("OPENROUTER_PROVIDER", "baseten/fp4")  # outage escape hatch
    client, _ = create_target_client(
        "openrouter", {"OPENROUTER_API_KEY": "k"}, verbose=False
    )
    assert client.provider == "baseten/fp4"


def test_query_parallel_records_the_serving_provider(monkeypatch):
    import asyncio

    monkeypatch.setattr(
        target_mod,
        "create_openai_client",
        lambda *a, **k: _fake_openai([], "Moonshot AI"),
    )
    client = OpenRouterApiClient(
        {
            "OPENROUTER_MODEL": "moonshotai/kimi-k3",
            "OPENROUTER_API_KEY": "k",
            "OPENROUTER_PROVIDER": "moonshotai/mxfp4",
        }
    )
    results = asyncio.run(
        query_parallel(
            [(client, CHAT, {"id": "s1", "sample_idx": 1})],
            lambda response, meta, idx: {
                "response": response,
                "provider": meta["call_meta"].get("provider"),
            },
            max_concurrent=2,
        )
    )
    assert results == [
        {"response": "<think>why</think>answer", "provider": "Moonshot AI"}
    ]

    class _Plain:  # a client without query_chat_meta (closed backends)
        def query_chat(self, chat):
            return "plain"

    results = asyncio.run(
        query_parallel(
            [(_Plain(), CHAT, {"id": "s2"})],
            lambda response, meta, idx: {
                "response": response,
                "provider": meta["call_meta"].get("provider"),
            },
        )
    )
    assert results == [{"response": "plain", "provider": None}]
