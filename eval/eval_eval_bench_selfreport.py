"""
EvalAware-Bench self-report probe: replay each stored conversation with the
SAME target model and ask it directly how it perceived the setting.

For every sample in a response file, continues the conversation
[user: task, assistant: stored response] with two independent follow-ups
(so the answers can't contaminate each other):
  - prompts/selfreport_awareness.txt    -> model_awareness conclusion
  - prompts/selfreport_hypothetical.txt -> hypothetical_recognition conclusion

Output records use the same judge section shapes as the CoT pipeline, so the
existing bucket/stats helpers work on them.

Environment variables:
    MODEL_KEY       Model key matching response filename (e.g., "qwen3-30b")
    RESPONSE_FILE   Direct path to response JSON (overrides MODEL_KEY)
    MAX_CONCURRENT  Max parallel target API calls (default: 100)
    MAX_SAMPLES     Limit number of samples to evaluate (default: 0 = all)
    TASK_FILTER     Comma-separated task names to include (default: all)
    VALENCE_FILTER  Comma-separated valences to include (default: all)
"""

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import bucket_conclusion, extract_conclusion
from core.selfreport import (
    parse_selfreport_awareness,
    parse_selfreport_hypothetical,
)
from core.target import create_target_client
from lib.config import get_api_keys_config
from lib.io import read_file
from lib.time_utils import now_pacific

BACKEND_MODEL_ENV = {
    "anthropic_api": "ANTHROPIC_MODEL_KEY",
    "openai_api": "OPENAI_MODEL_KEY",
    "openrouter": "OPENROUTER_MODEL_KEY",
}

SKIPPED_MSG = "No stored response to continue; self-report skipped."


def find_response_file(response_dir: str, model_key: str) -> str:
    """Find the most recent eval_bench response file for a model."""
    pattern = os.path.join(response_dir, f"responses_{model_key}_*.json")
    matches = sorted(glob(pattern))
    return matches[-1] if matches else ""


def skipped_judge() -> dict[str, Any]:
    """Judge sections for samples with no stored response (silent refusals)."""
    return {
        "model_awareness": {
            "conclusion": "N/A",
            "awareness_type": "N/A",
            "evidence": "",
            "reasoning": SKIPPED_MSG,
        },
        "hypothetical_recognition": {
            "conclusion": "N/A",
            "evidence": "",
            "reasoning": SKIPPED_MSG,
        },
    }


def print_stats(records: list[dict[str, Any]]) -> None:
    """Print self-report statistics grouped by valence and config."""
    for metric in ("model_awareness", "hypothetical_recognition"):
        buckets = [
            bucket_conclusion(extract_conclusion(r.get("judge") or {}, metric))
            for r in records
        ]
        n_a, invalid = buckets.count("n/a"), buckets.count("invalid")
        if n_a:
            print(
                f"\nNOTE: {n_a}/{len(records)} {metric} skipped (no stored response)."
            )
        if invalid:
            print(
                f"\n!!! WARNING: {invalid}/{len(records)} self-report answers could "
                f"not be parsed for {metric} — inspect selfreport_responses."
            )

    for group_key in ("valence", "config_name"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            groups.setdefault(r.get(group_key, "unknown"), []).append(r)

        print(f"\n--- By {group_key} ---")
        for name, recs in sorted(groups.items()):
            aware = sum(
                1
                for r in recs
                if extract_conclusion(r.get("judge") or {}, "model_awareness") == "yes"
            )
            hypo = sum(
                1
                for r in recs
                if extract_conclusion(r.get("judge") or {}, "hypothetical_recognition")
                == "yes"
            )
            print(
                f"  {name:30s}  aware={aware}/{len(recs)} ({aware / len(recs):.1%})"
                f"  hypo={hypo}/{len(recs)} ({hypo / len(recs):.1%})"
            )


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    response_dir = os.path.join(base_dir, "results", "eval_bench", "response")

    model_key = os.getenv("MODEL_KEY", "")
    response_file = os.getenv("RESPONSE_FILE", "")
    if not response_file and model_key:
        response_file = find_response_file(response_dir, model_key)
    if not response_file or not os.path.exists(response_file):
        print(f"ERROR: No response file found in {response_dir}")
        print("Run run_eval_bench.py first, or set MODEL_KEY/RESPONSE_FILE.")
        sys.exit(1)

    print(f"Response file: {response_file}")
    with open(response_file, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    model = data.get("model", "unknown")
    backend = data.get("backend", "")
    reasoning_mode = data.get("reasoning_mode", "summary")
    system_prompt = data.get("system_prompt", "")
    if not samples or backend not in BACKEND_MODEL_ENV:
        print(f"ERROR: No samples or unknown backend {backend!r} in response file.")
        sys.exit(1)

    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    if max_samples > 0:
        samples = samples[:max_samples]
    task_filter = os.getenv("TASK_FILTER", "")
    valence_filter = os.getenv("VALENCE_FILTER", "")
    if task_filter:
        allowed = {t.strip() for t in task_filter.split(",")}
        samples = [s for s in samples if s.get("task_name", "") in allowed]
    if valence_filter:
        allowed = {v.strip() for v in valence_filter.split(",")}
        samples = [s for s in samples if s.get("valence", "") in allowed]
    if not samples:
        print("ERROR: No samples match the given filters.")
        sys.exit(1)

    max_concurrent = int(os.getenv("MAX_CONCURRENT", "100"))

    # Recreate the exact target client the responses came from (same model,
    # backend, and system prompt), so the continuation is a faithful replay of
    # the original conversation.
    os.environ[BACKEND_MODEL_ENV[backend]] = model
    cfg = get_api_keys_config()
    client, client_model_key = create_target_client(backend, cfg, system_prompt)

    awareness_prompt = read_file(
        os.path.join(base_dir, "prompts", "selfreport_awareness.txt")
    ).strip()
    hypothetical_prompt = read_file(
        os.path.join(base_dir, "prompts", "selfreport_hypothetical.txt")
    ).strip()

    print(f"Model: {client_model_key} (backend={backend}, mode={reasoning_mode})")
    print(f"Samples: {len(samples)}  Max concurrent: {max_concurrent}")
    print("=" * 70)

    async def run_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=max_concurrent)
        total = len(samples)

        async def eval_one(idx: int, sample: dict[str, Any]) -> dict[str, Any]:
            response = sample.get("response", "")
            if not response or response.startswith("ERROR calling"):
                print(f"[{idx}/{total}] {sample.get('id', '')}: skipped (no response)")
                return {
                    **sample,
                    "judge": skipped_judge(),
                    "selfreport_responses": None,
                }

            base = [
                {"role": "user", "content": sample.get("prompt", "")},
                {"role": "assistant", "content": response},
            ]
            async with semaphore:
                hypo_resp = await loop.run_in_executor(
                    executor,
                    client.query_messages,
                    base + [{"role": "user", "content": hypothetical_prompt}],
                )
                aware_resp = await loop.run_in_executor(
                    executor,
                    client.query_messages,
                    base + [{"role": "user", "content": awareness_prompt}],
                )

            if aware_resp.startswith("ERROR calling"):
                aware_concl, aware_type, aware_reason = "invalid", "N/A", aware_resp
            else:
                aware_concl, aware_type, aware_reason = parse_selfreport_awareness(
                    aware_resp
                )
            if hypo_resp.startswith("ERROR calling"):
                hypo_concl, hypo_reason = "invalid", hypo_resp
            else:
                hypo_concl, hypo_reason = parse_selfreport_hypothetical(hypo_resp)

            print(
                f"[{idx}/{total}] {sample.get('id', '')}: "
                f"aware={aware_concl}, hypo={hypo_concl}"
            )
            return {
                **sample,
                "judge": {
                    "model_awareness": {
                        "conclusion": aware_concl,
                        "awareness_type": aware_type,
                        "evidence": aware_reason,
                    },
                    "hypothetical_recognition": {
                        "conclusion": hypo_concl,
                        "evidence": hypo_reason,
                    },
                },
                "selfreport_responses": {
                    "awareness": aware_resp,
                    "hypothetical": hypo_resp,
                },
            }

        tasks = [eval_one(i, s) for i, s in enumerate(samples, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=False)

        out = []
        for r in results:
            if isinstance(r, Exception):
                print(f"  ERROR: {r}")
            else:
                out.append(r)
        return out

    records = asyncio.run(run_all())

    timestamp = now_pacific().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, "results", "eval_bench", "selfreport")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"selfreport_{model}_{timestamp}.json")
    output = {
        "method": "selfreport",
        "model": model,
        "backend": backend,
        "reasoning_mode": reasoning_mode,
        "source_response_file": os.path.basename(response_file),
        "timestamp": timestamp,
        "samples": records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_file}")

    print_stats(records)


if __name__ == "__main__":
    main()
