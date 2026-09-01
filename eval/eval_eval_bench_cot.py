"""
EvalAware-Bench CoT evaluation: judge model responses for evaluation awareness.

Loads responses from run_eval_bench.py and evaluates each sample with a judge model
using the standard CoT monitoring pipeline.

Environment variables:
    JUDGE_MODEL          Primary judge (default: gpt-5.6-terra; OPENAI_EVAL_MODEL
                         also honored). claude-* routes to the Anthropic API.
    BACKUP_JUDGE_MODEL   Fallback judge for transcripts the primary refused / returned
                         unparseable JSON for (default: claude-sonnet-5). Only those
                         samples are re-graded; set "" to disable the fallback.
    MODEL_KEY            Model key matching response filename (e.g., "qwen3-30b")
    RESPONSE_FILE        Direct path to response JSON (overrides MODEL_KEY)
    MAX_CONCURRENT       Max parallel judge API calls (default: 100)
    MAX_SAMPLES          Limit number of samples to evaluate (default: 0 = all)
    TASK_FILTER          Comma-separated task names to judge (default: all)
    VALENCE_FILTER       Comma-separated valences to judge (default: all)
    MERGE_INTO           Path to existing CoT JSON to merge into (replaces matching IDs)

Default grader stack: gpt-5.6-terra primary (0 refusals, κ=0.91 vs gpt-5) +
claude-sonnet-5 backup for anything terra returns unparseable (awareness κ=1.0
vs terra on a 280-transcript check). Flip with JUDGE_MODEL=claude-sonnet-5
BACKUP_JUDGE_MODEL=gpt-5.6-terra to grade on Anthropic credits instead. See
.claude/notes/grader-selection-and-refusal-clause.md.
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
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
from lib.chat import chat_to_transcript_text
from lib.config import get_api_keys_config
from lib.io import load_responses, read_file
from lib.time_utils import now_pacific

# A verdict is valid if the judge returned a parseable model_awareness conclusion
# ("yes"/"no"/"n/a"). A refusal comes back as {"raw": <text>} with no
# model_awareness key, and a hard content-filter block leaves judge=None — both
# need the backup judge.
_VALID_CONCLUSIONS = {"yes", "no", "n/a"}


def needs_backup(record: dict[str, Any]) -> bool:
    """True if the primary judge failed to produce a usable verdict for this record."""
    judge = record.get("judge")
    if not isinstance(judge, dict):
        return True
    awareness = judge.get("model_awareness")
    if not isinstance(awareness, dict):
        return True
    return (
        str(awareness.get("conclusion", "")).strip().lower() not in _VALID_CONCLUSIONS
    )


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
    # Backup grader for transcripts the primary refuses / returns unparseable
    # JSON for; "" disables. Skipped if it equals the primary.
    backup_judge_model = os.getenv("BACKUP_JUDGE_MODEL", "claude-sonnet-5").strip()
    if backup_judge_model == judge_model:
        backup_judge_model = ""
    model_key = os.getenv("MODEL_KEY", "")
    response_file = os.getenv("RESPONSE_FILE", "")
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "100"))
    task_filter = os.getenv("TASK_FILTER", "")
    valence_filter = os.getenv("VALENCE_FILTER", "")
    merge_into = os.getenv("MERGE_INTO", "")

    for role, jm in (("primary", judge_model), ("backup", backup_judge_model)):
        if not jm:
            continue
        if jm.startswith("claude"):
            if not anthropic_api_key:
                print(
                    f"ERROR: ANTHROPIC_API_KEY required for Claude {role} judge {jm!r}."
                )
                sys.exit(1)
        elif not openai_api_key:
            print(f"ERROR: OPENAI_API_KEY required for OpenAI {role} judge {jm!r}.")
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

    samples, model, reasoning_mode = load_responses(response_file)
    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)
    if reasoning_mode == "none":
        print(
            "ERROR: this response file was generated with THINK_TOOL=0, so it has "
            "no CoT to judge. Use eval/eval_eval_bench_selfreport.py instead."
        )
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
    print(f"Backup judge: {backup_judge_model or '(disabled)'}")
    print(f"Max concurrent: {max_concurrent}")
    print("=" * 70)

    judge_prompt_file = "judgePromptEvidenceBoth.txt"
    judge_template = read_file(os.path.join(base_dir, "prompts", judge_prompt_file))
    # Stamped onto every record so results stay traceable to the exact judge
    # prompt that produced them (MERGE_INTO can mix records across runs).
    judge_prompt_sha = hashlib.sha256(judge_template.encode()).hexdigest()[:8]
    print(f"Judge prompt: {judge_prompt_file} (sha {judge_prompt_sha})")

    def make_eval_fn(active_judge: str):
        """Build a sample-evaluator bound to a specific judge model.

        result_fn does {**sample, ...}: on the backup pass `sample` is the primary
        record, so its stale judge_* fields are overwritten with the backup's.
        """

        async def eval_sample_fn(
            client: Any,
            sample: dict[str, Any],
            semaphore: asyncio.Semaphore,
            idx: int,
            total: int,
        ) -> dict[str, Any]:
            # The judge's {task} slot gets a readable transcript of the whole
            # scenario (system + prior turns + final request).
            # TODO(cleanup): drop the legacy flat-"prompt" fallback once every
            # pre-refactor response file is regenerated or retired — then this
            # can just be chat_to_transcript_text(sample["chat"]).
            chat = sample.get("chat")
            prompt = chat_to_transcript_text(chat) if chat else sample.get("prompt", "")
            response = sample.get("response", "")

            return await evaluate_sample(
                semaphore,
                idx,
                total,
                sample_id=sample.get("id", ""),
                judge_fn=lambda: evaluate_response_async(
                    client, active_judge, prompt, response, judge_template
                ),
                result_fn=lambda judge: {
                    **sample,
                    "judge": judge.get("judgment"),
                    "judge_raw": judge.get("raw"),
                    "judge_model": active_judge,
                    "judge_prompt": judge_prompt_file,
                    "judge_prompt_sha": judge_prompt_sha,
                },
                metrics=[
                    ("model_awareness", "awareness"),
                    ("hypothetical_recognition", "hypo"),
                    ("task_performance", "task"),
                ],
            )

        return eval_sample_fn

    def run_judge(judge: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            evaluate_batch_async(
                samples=batch,
                sample_evaluator=make_eval_fn(judge),
                api_key=openai_api_key,
                max_concurrent=max_concurrent,
                judge_model=judge,
                anthropic_api_key=anthropic_api_key,
            )
        )

    all_records = run_judge(judge_model, samples)

    # Backup pass: re-grade only the transcripts the primary judge refused or
    # returned unparseable JSON for (Claude judges refuse the most harmful items).
    if backup_judge_model:
        stalled = [r for r in all_records if needs_backup(r)]
        if not stalled:
            print(f"\nAll {len(all_records)} verdicts parseable; backup not needed.")
        else:
            print(
                f"\n{len(stalled)}/{len(all_records)} verdicts unparseable from "
                f"{judge_model}; re-grading those with backup {backup_judge_model}..."
            )
            redone = run_judge(backup_judge_model, stalled)
            for r in redone:
                r["judge_backup_used"] = True
            redone_ids = {r.get("id") for r in redone}
            all_records = [
                r for r in all_records if r.get("id") not in redone_ids
            ] + redone
            still_bad = sum(1 for r in redone if needs_backup(r))
            print(
                f"  backup resolved {len(redone) - still_bad}/{len(redone)} "
                f"({still_bad} still unparseable)"
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
        timestamp = now_pacific().strftime("%Y%m%d_%H%M%S")
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
