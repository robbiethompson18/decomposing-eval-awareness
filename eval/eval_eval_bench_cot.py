"""
EvalAware-Bench CoT evaluation: judge model responses for evaluation awareness.

Loads responses from run_eval_bench.py and evaluates each sample with a judge model
using the standard CoT monitoring pipeline.

Environment variables:
    JUDGE_MODEL     Judge model (default: gpt-5.6-terra; OPENAI_EVAL_MODEL also honored)
    MODEL_KEY       Model key matching response filename (e.g., "qwen3-30b")
    RESPONSE_FILE   Direct path to response JSON (overrides MODEL_KEY)
    MAX_CONCURRENT  Max parallel judge API calls (default: 100)
    MAX_SAMPLES     Limit number of samples to evaluate (default: 0 = all)
    TASK_FILTER     Comma-separated task names to judge (default: all)
    VALENCE_FILTER  Comma-separated valences to judge, e.g. "capability" (default: all)
    MERGE_INTO      Path to existing CoT JSON to merge into (replaces matching sample IDs)
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from glob import glob
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.evaluator import (
    evaluate_batch_async,
    evaluate_response_async,
    evaluate_sample,
    extract_think_tags,
)
from core.metrics import bucket_conclusion, bucket_task_conclusion, extract_conclusion
from lib.config import get_api_keys_config
from lib.io import load_responses, read_file


def find_response_file(response_dir: str, model_key: str) -> str:
    """Find the most recent eval_bench response file for a model ("*" = any model)."""
    pattern = os.path.join(response_dir, f"responses_{model_key}_*.json")
    matches = sorted(glob(pattern))
    matches = [m for m in matches if "_scratchpad_" not in os.path.basename(m)]
    return matches[-1] if matches else ""


def print_stats(records: list[dict[str, Any]]) -> None:
    """Print overall awareness plus statistics grouped by valence and config."""

    aware = sum(
        1
        for r in records
        if extract_conclusion(r.get("judge") or {}, "model_awareness") == "yes"
    )
    print(f"Overall awareness: {aware}/{len(records)} ({aware / len(records):.1%})")

    buckets = [
        bucket_conclusion(extract_conclusion(r.get("judge") or {}, "model_awareness"))
        for r in records
    ]
    n_a = buckets.count("n/a")
    invalid = buckets.count("invalid")
    no_cot = sum(
        1 for r in records if not extract_think_tags(r.get("response") or "")[0]
    )
    if n_a:
        print(
            f"\n!!! NOTE: {n_a}/{len(records)} samples not graded for awareness "
            f"(N/A) — no reasoning present in the response."
        )
    if no_cot > n_a:
        print(
            f"\n!!! WARNING: {no_cot}/{len(records)} responses contain no reasoning "
            f"(CoT), but only {n_a} were marked N/A — this file was likely judged "
            f"before the no-reasoning guard; its awareness numbers are unreliable."
        )
    if invalid:
        print(
            f"\n!!! WARNING: {invalid}/{len(records)} judge outputs could not be "
            f"parsed as Yes/No for model_awareness — they count in denominators "
            f"below but not as aware. Inspect the raw judge responses."
        )

    for group_key in ("valence", "config_name"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            g = r.get(group_key, "unknown")
            groups.setdefault(g, []).append(r)

        print(f"\n--- By {group_key} ---")
        for name, recs in sorted(groups.items()):
            aware = sum(
                1
                for r in recs
                if extract_conclusion(r.get("judge") or {}, "model_awareness") == "yes"
            )
            print(
                f"  {name:30s}  awareness={aware}/{len(recs)} ({aware / len(recs):.1%})"
            )

    # Task-execution outcomes (v2 judge: correct/guarded/incorrect/refused;
    # legacy files judged before the v2 prompt show yes/no here).
    print("\n--- Task execution by valence ---")
    by_valence: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_valence.setdefault(r.get("valence", "unknown"), []).append(r)
    for name, recs in sorted(by_valence.items()):
        counts = Counter(
            bucket_task_conclusion(
                extract_conclusion(r.get("judge") or {}, "task_performance")
            )
            for r in recs
        )
        summary = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"  {name:30s}  {summary}")


def main() -> None:
    api_keys = get_api_keys_config()
    openai_api_key = api_keys["OPENAI_API_KEY"]
    anthropic_api_key = api_keys["ANTHROPIC_API_KEY"]
    judge_model = (
        os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_EVAL_MODEL") or "gpt-5.6-terra"
    )
    model_key = os.getenv("MODEL_KEY", "")
    response_file = os.getenv("RESPONSE_FILE", "")
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "100"))
    task_filter = os.getenv("TASK_FILTER", "")
    valence_filter = os.getenv("VALENCE_FILTER", "")
    merge_into = os.getenv("MERGE_INTO", "")

    if judge_model.startswith("claude"):
        if not anthropic_api_key:
            print("ERROR: ANTHROPIC_API_KEY is required for a Claude judge.")
            sys.exit(1)
    elif not openai_api_key:
        print("ERROR: OPENAI_API_KEY is required.")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    response_dir = os.path.join(base_dir, "results", "eval_bench", "response")

    if not response_file:
        response_file = find_response_file(response_dir, model_key or "*")

    if not response_file or not os.path.exists(response_file):
        print(f"ERROR: No response file found in {response_dir}")
        print("Run run_eval_bench.py first, or set MODEL_KEY/RESPONSE_FILE.")
        sys.exit(1)

    print(f"Response file: {response_file}")

    samples, model = load_responses(response_file)
    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)

    if not model_key:
        match = re.search(
            r"responses_([^_]+(?:-[^_]+)*)", os.path.basename(response_file)
        )
        model_key = match.group(1) if match else model.replace("/", "_")

    if task_filter:
        allowed = {t.strip() for t in task_filter.split(",")}
        samples = [s for s in samples if s.get("task_name", "") in allowed]
    if valence_filter:
        allowed = {v.strip() for v in valence_filter.split(",")}
        samples = [s for s in samples if s.get("valence", "") in allowed]

    if not samples:
        print("ERROR: No samples match the given filters.")
        sys.exit(1)

    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} samples from model: {model}")
    print(f"Judge model: {judge_model}")
    print(f"Max concurrent: {max_concurrent}")
    print("=" * 70)

    judge_prompt_file = "judgePromptEvidenceBoth.txt"
    judge_template = read_file(os.path.join(base_dir, "prompts", judge_prompt_file))
    # Stamped onto every record so results stay traceable to the exact judge
    # prompt that produced them (MERGE_INTO can mix records across runs).
    judge_prompt_sha = hashlib.sha256(judge_template.encode()).hexdigest()[:8]
    print(f"Judge prompt: {judge_prompt_file} (sha {judge_prompt_sha})")

    async def eval_sample_fn(
        client: Any,
        sample: dict[str, Any],
        semaphore: asyncio.Semaphore,
        idx: int,
        total: int,
    ) -> dict[str, Any]:
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        return await evaluate_sample(
            semaphore,
            idx,
            total,
            sample_id=sample.get("id", ""),
            judge_fn=lambda: evaluate_response_async(
                client, judge_model, prompt, response, judge_template
            ),
            result_fn=lambda judge: {
                **sample,
                "judge": judge.get("judgment"),
                "judge_raw": judge.get("raw"),
                "judge_model": judge_model,
                "judge_prompt": judge_prompt_file,
                "judge_prompt_sha": judge_prompt_sha,
            },
            metrics=[
                ("model_awareness", "awareness"),
                ("hypothetical_recognition", "hypo"),
                ("task_performance", "task"),
            ],
        )

    all_records = asyncio.run(
        evaluate_batch_async(
            samples=samples,
            sample_evaluator=eval_sample_fn,
            api_key=openai_api_key,
            max_concurrent=max_concurrent,
            judge_model=judge_model,
            anthropic_api_key=anthropic_api_key,
        )
    )

    # Save or merge results
    if merge_into and os.path.exists(merge_into):
        with open(merge_into, "r", encoding="utf-8") as f:
            existing = json.load(f)
        new_ids = {r.get("id") for r in all_records}
        kept = [r for r in existing if r.get("id") not in new_ids]
        merged = kept + all_records
        with open(merge_into, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(
            f"\nMerged {len(all_records)} new records "
            f"(replaced {len(existing) - len(kept)} existing)"
        )
        print(f"Updated: {merge_into}")
        all_records = merged
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(base_dir, "results", "eval_bench", "cot")
        os.makedirs(results_dir, exist_ok=True)
        out_file = os.path.join(results_dir, f"cot_{model_key}_{timestamp}.json")

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {out_file}")

    print("\n" + "=" * 70)
    print(f"EVAL BENCH COT RESULTS — {model_key} ({len(all_records)} samples)")
    print("=" * 70)

    if all_records:
        print_stats(all_records)
    else:
        print("No successfully judged records to report.")


if __name__ == "__main__":
    main()
