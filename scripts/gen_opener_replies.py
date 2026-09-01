"""Capture each roster model's real reply to the F11 opener, once, into
eval_bench/opener_replies.yaml (keyed [model_key][f9_level]).

Plain calls — same system prompt the run will use, no think tool, default
reasoning — so the reply is what the model naturally says to "are you there?".
Existing entries are kept unless MODELS names that model, so re-running for a
new model never churns replies Robbie already reviewed.

Usage:
  python scripts/gen_opener_replies.py                 # every roster model
  MODELS=kimi-k3,gpt-5.6-sol python scripts/gen_opener_replies.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_bench.render import OPENER, OPENER_REPLIES_PATH, system_prompt_for
from lib.config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_GEMINI_CONFIGS,
    DEFAULT_OPENAI_MODELS,
    DEFAULT_OPENROUTER_CONFIGS,
    get_api_keys_config,
)

# The 14-model target roster (CLAUDE.md), each on the backend the run uses.
ROSTER: list[tuple[str, str]] = [
    ("gemini_api", "gemini-3.1-pro"),
    ("gemini_api", "gemini-3.5-flash"),
    ("anthropic_api", "claude-sonnet-4.5"),
    ("anthropic_api", "claude-haiku-4.5"),
    ("openai_api", "gpt-5"),
    ("openai_api", "gpt-5.1"),
    ("openai_api", "gpt-5.2"),
    ("openai_api", "gpt-5.4"),
    ("openai_api", "gpt-5.6-sol"),
    ("openrouter", "qwen3.8-2.4t"),
    ("openrouter", "deepseek-v4-pro"),
    ("openrouter", "kimi-k3"),
    ("openrouter", "glm-5.3"),
    ("openrouter", "minimax-m3"),
]
F9_LEVELS = ("deployment", "bare")
MAX_TOKENS = 1024


def ask(backend: str, model_key: str, system: str, keys: dict, clients: dict) -> str:
    if backend == "anthropic_api":
        r = requests.post(
            keys["ANTHROPIC_API_BASE"].rstrip("/") + "/v1/messages",
            headers={
                "x-api-key": keys["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": DEFAULT_ANTHROPIC_MODELS[model_key],
                "system": system,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": OPENER}],
            },
            timeout=120,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    if backend == "openai_api":
        resp = clients["openai"].responses.create(
            model=DEFAULT_OPENAI_MODELS[model_key],
            instructions=system,
            input=OPENER,
            max_output_tokens=MAX_TOKENS,
        )
        return resp.output_text.strip()
    if backend == "gemini_api":
        model_id = DEFAULT_GEMINI_CONFIGS[model_key][0]
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={keys['GEMINI_API_KEY']}",
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": OPENER}]}],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS},
            },
            timeout=120,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if backend == "openrouter":
        model_id, temperature = DEFAULT_OPENROUTER_CONFIGS[model_key]
        kwargs = {"temperature": temperature} if temperature is not None else {}
        resp = clients["openrouter"].chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": OPENER},
            ],
            max_tokens=MAX_TOKENS,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    raise ValueError(backend)


def main() -> None:
    keys = get_api_keys_config()
    # Built once, up front: instantiating SDK clients inside the pool races the
    # SDK's lazy resource imports (a _ModuleLock deadlock on first use).
    clients = {
        "openai": OpenAI(api_key=keys["OPENAI_API_KEY"]),
        "openrouter": OpenAI(
            api_key=keys["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"
        ),
    }
    only = {m.strip() for m in os.getenv("MODELS", "").split(",") if m.strip()}
    roster = [(b, m) for b, m in ROSTER if not only or m in only]
    unknown = only - {m for _, m in ROSTER}
    if unknown:
        sys.exit(f"MODELS not in roster: {sorted(unknown)}")

    existing: dict = {}
    if OPENER_REPLIES_PATH.exists():
        existing = yaml.safe_load(OPENER_REPLIES_PATH.read_text()) or {}

    jobs = [(b, m, lvl) for b, m in roster for lvl in F9_LEVELS]

    def run(job: tuple[str, str, str]) -> tuple[str, str, str]:
        backend, model_key, level = job
        try:
            reply = ask(backend, model_key, system_prompt_for(level), keys, clients)
        except Exception as e:  # noqa: BLE001 -- report per-model, keep the rest
            reply = f"ERROR: {e}"
        print(f"[{model_key} / {level}] {reply[:120]!r}")
        return model_key, level, reply

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run, jobs))

    for model_key, level, reply in results:
        if reply.startswith("ERROR:"):
            continue  # keep whatever was there; the run raises if it's missing
        existing.setdefault(model_key, {})[level] = reply

    header = (
        "# Each roster model's real reply to the F11 opener "
        f"({OPENER!r}) under each F9 system prompt.\n"
        "# Captured by scripts/gen_opener_replies.py (plain call, no think tool),"
        " then reviewed by hand.\n"
        "# render() prepends [user: opener, assistant: reply] for F11=warm; a missing"
        " entry raises.\n"
    )
    ordered = {m: existing[m] for _, m in ROSTER if m in existing}
    ordered.update({m: v for m, v in existing.items() if m not in ordered})
    OPENER_REPLIES_PATH.write_text(
        header + yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, width=88)
    )
    errors = [(m, lvl) for m, lvl, r in results if r.startswith("ERROR:")]
    print(f"\nWrote {OPENER_REPLIES_PATH} ({len(results) - len(errors)} replies)")
    if errors:
        print(f"FAILED (not written): {errors}")


if __name__ == "__main__":
    main()
