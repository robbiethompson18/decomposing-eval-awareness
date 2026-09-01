"""Smoketest: verify OpenRouter models return raw (not summarized) reasoning.

For each model, fires N prompts and checks:
  1. Presence  — reasoning is non-empty (catches providers that return reasoning=null)
  2. Rawness   — reasoning chars / reasoning tokens ~ 3-4 (a summary shows chars << tokens);
                 needs the provider to report reasoning_tokens, else marked n/a
  3. Providers — which OpenRouter providers actually served the calls, per model

Usage:
    MODELS=kimi-k3,glm-5.3 N=10 python run/smoke_openrouter_reasoning.py
Defaults to the tier-1 open-model roster, N=10.
Writes results/smoketests/openrouter_reasoning_<ts>.json and prints a summary table.
"""

import json
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.target import OpenRouterApiClient
from lib.config import DEFAULT_OPENROUTER_CONFIGS, get_api_keys_config

DEFAULT_MODELS = ["qwen3.8-2.4t", "deepseek-v4-pro", "kimi-k3", "glm-5.3", "minimax-m3"]

SYSTEM_PROMPT = "You are a helpful assistant."

# Benign-but-hard prompts so the model actually engages its reasoning, plus two
# borderline ones to observe the reasoning-skipped-on-refusal pattern (qwen3-235b
# emits <think>\n\n</think> on easy refusals).
PROMPTS = [
    "A snail climbs 3m up a 10m well each day and slips back 2m each night. On which day does it reach the top? Think carefully.",
    "How many ways can you make change for a dollar using pennies, nickels, dimes, and quarters?",
    "If you have a 3-liter jug and a 5-liter jug, how do you measure exactly 4 liters? Explain each step.",
    "Alice, Bob, and Carol each either always lie or always tell the truth. Alice says 'Bob is a liar.' Bob says 'Carol is a liar.' Carol says 'Alice and Bob are both liars.' Who lies?",
    "What is the smallest positive integer that leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 4 when divided by 7?",
    "Estimate how many piano tuners work in Chicago. Show your reasoning.",
    "Prove or disprove: the product of any four consecutive integers is one less than a perfect square.",
    "Design a fair way for two people to split an irregularly shaped cake so neither envies the other, then extend it to three people.",
    "I'm writing a thriller novel. Describe in general terms how my detective character might realize a death was staged to look like an accident.",
    "My elderly neighbor keeps getting scam calls. What psychological techniques do the scammers likely use, so I can explain the red flags to her?",
]


def main() -> None:
    model_keys = [
        m.strip()
        for m in os.getenv("MODELS", ",".join(DEFAULT_MODELS)).split(",")
        if m.strip()
    ]
    n = int(os.getenv("N", "10"))
    prompts = (PROMPTS * (n // len(PROMPTS) + 1))[:n]

    unknown = [m for m in model_keys if m not in DEFAULT_OPENROUTER_CONFIGS]
    if unknown:
        print(f"ERROR: unknown model keys {unknown}")
        sys.exit(1)

    cfg_base = {**get_api_keys_config(), "MAX_TOKENS": 8192}

    def one_call(model_key: str, prompt: str) -> dict:
        model_id, temperature = DEFAULT_OPENROUTER_CONFIGS[model_key]
        # Fresh client per call: query_messages_meta is stateless, but this also
        # keeps any future client-side state per-thread.
        client = OpenRouterApiClient(
            {**cfg_base, "OPENROUTER_MODEL": model_id, "TEMPERATURE": temperature},
            SYSTEM_PROMPT,
        )
        text, meta = client.query_messages_meta([{"role": "user", "content": prompt}])
        return {"model_key": model_key, "prompt": prompt, "response": text, **meta}

    jobs = [(mk, p) for mk in model_keys for p in prompts]
    print(
        f"Running {len(jobs)} calls ({len(model_keys)} models x {len(prompts)} prompts)..."
    )
    with ThreadPoolExecutor(max_workers=100) as pool:
        results = list(pool.map(lambda j: one_call(*j), jobs))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / "smoketests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openrouter_reasoning_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}\n")

    header = (
        f"{'model':18s} {'ok':>5s} {'reasoning':>9s} {'chars/tok':>9s} {'providers'}"
    )
    print(header)
    print("-" * len(header))
    for mk in model_keys:
        rs = [r for r in results if r["model_key"] == mk]
        errors = [r for r in rs if "error" in r or r["response"].startswith("ERROR")]
        ok = [r for r in rs if r not in errors]
        with_reasoning = [r for r in ok if r.get("reasoning_chars", 0) > 0]
        ratios = [
            r["reasoning_chars"] / r["reasoning_tokens"]
            for r in with_reasoning
            if r.get("reasoning_tokens")
        ]
        ratio_s = f"{statistics.median(ratios):.1f}" if ratios else "n/a"
        providers = Counter(r.get("provider") or "?" for r in ok)
        prov_s = ", ".join(f"{p}:{c}" for p, c in providers.most_common())
        print(
            f"{mk:18s} {len(ok)}/{len(rs):<3d} {len(with_reasoning):>4d}/{len(ok):<4d} "
            f"{ratio_s:>9s} {prov_s}"
        )
        for r in errors:
            print(f"    ERROR: {r['response'][:120]}")


if __name__ == "__main__":
    main()
