"""Re-judge a stratified sample of already-graded transcripts with several
judge models, so judge agreement can be measured (see eval_judge_agreement_stats.py).

Env:
    JUDGES          comma-separated judge model ids (default:
                    "gpt-5,gpt-5.6-terra,claude-sonnet-5"). claude-* routes to
                    the Anthropic API, everything else to OpenAI.
    N_SAMPLES       total transcripts to sample (default 300)
    MAX_CONCURRENT  parallel judge calls (default 100)

Reads the source cot files listed in SOURCE_FILES (one per target model, the
best-provenance GPT-5-judged file for each), stratified-samples N_SAMPLES
records with non-empty CoT (records without CoT are force-marked N/A by the
no-reasoning guard identically under every judge, so they say nothing about
agreement), and writes one output file per judge to
results/eval_bench/judge_agreement/judged_{judge}_{timestamp}.json. The
original stored judgment is preserved on every record as "judge_stored".
"""

import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import (
    evaluate_batch_async,
    evaluate_response_async,
    evaluate_sample,
    extract_think_tags,
)
from lib.config import get_api_keys_config
from lib.io import read_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT_DIR = os.path.join(BASE_DIR, "results", "eval_bench", "cot")
OUT_DIR = os.path.join(BASE_DIR, "results", "eval_bench", "judge_agreement")

# target_model_key -> the cot file to sample from — the broadest-coverage file
# per target model. Stored judge provenance is uncertain for the unprefixed
# files (judge A/B runs happened the same day), which is fine: agreement is
# measured against a fresh gpt-5 baseline, not the stored judgment.
# claude-fable-5 and claude-opus-5 are excluded: their responses carry no
# <think> CoT, so the two reasoning-based metrics are force-marked N/A for
# every judge and carry no agreement signal.
SOURCE_FILES = {
    "claude-sonnet-4.5": "cot_claude-sonnet-4.5_20260831_163404.json",
    "qwen3-30b": "cot_qwen3-30b_20260831_163625.json",
    "claude-haiku-4.5": "cot_claude-haiku-4.5_20260831_130058.json",
}


def stratified_sample(n_total: int) -> list[dict[str, Any]]:
    """Sample ~n_total/len(SOURCE_FILES) records per target model, spread
    round-robin across (valence, config_name) cells within each model."""
    rng = random.Random(0)
    per_model = n_total // len(SOURCE_FILES)
    sampled: list[dict[str, Any]] = []

    for model_key, fname in SOURCE_FILES.items():
        with open(os.path.join(COT_DIR, fname), encoding="utf-8") as f:
            records = json.load(f)
        with_cot = [
            r for r in records if extract_think_tags(r.get("response") or "")[0]
        ]
        cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in with_cot:
            cells[(r.get("valence", "?"), r.get("config_name", "?"))].append(r)
        for cell in cells.values():
            rng.shuffle(cell)

        picked: list[dict[str, Any]] = []
        while len(picked) < per_model and any(cells.values()):
            for key in sorted(cells):
                if cells[key] and len(picked) < per_model:
                    picked.append(cells[key].pop())

        skipped = len(records) - len(with_cot)
        print(
            f"{model_key:20s} {len(picked):3d} sampled "
            f"({len(with_cot)} with CoT, {skipped} CoT-less skipped)"
        )
        for r in picked:
            sampled.append(
                {
                    "target_model": model_key,
                    "judge_stored": r.get("judge"),
                    **{
                        k: r[k]
                        for k in (
                            "id",
                            "task_id",
                            "task_name",
                            "valence",
                            "config_name",
                            "prompt",
                            "response",
                        )
                        if k in r
                    },
                }
            )
    return sampled


async def judge_all(
    samples: list[dict[str, Any]],
    judge_model: str,
    judge_template: str,
    api_keys: dict[str, Any],
    max_concurrent: int,
) -> list[dict[str, Any]]:
    async def eval_sample_fn(
        client: Any,
        sample: dict[str, Any],
        semaphore: asyncio.Semaphore,
        idx: int,
        total: int,
    ) -> dict[str, Any]:
        return await evaluate_sample(
            semaphore,
            idx,
            total,
            sample_id=f"{sample['target_model']}/{sample.get('id', '')}",
            judge_fn=lambda: evaluate_response_async(
                client,
                judge_model,
                sample.get("prompt", ""),
                sample.get("response", ""),
                judge_template,
            ),
            result_fn=lambda judge: {
                **sample,
                "judge": judge.get("judgment"),
                "judge_raw": judge.get("raw"),
            },
        )

    return await evaluate_batch_async(
        samples=samples,
        sample_evaluator=eval_sample_fn,
        api_key=api_keys["OPENAI_API_KEY"],
        max_concurrent=max_concurrent,
        judge_model=judge_model,
        anthropic_api_key=api_keys["ANTHROPIC_API_KEY"],
    )


def main() -> None:
    judges = [
        j.strip()
        for j in os.getenv("JUDGES", "gpt-5,gpt-5.6-terra,claude-sonnet-5").split(",")
        if j.strip()
    ]
    n_samples = int(os.getenv("N_SAMPLES", "300"))
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "100"))

    api_keys = get_api_keys_config()
    if (
        any(j.startswith("claude") for j in judges)
        and not api_keys["ANTHROPIC_API_KEY"]
    ):
        sys.exit("ERROR: ANTHROPIC_API_KEY required for a Claude judge.")
    if (
        any(not j.startswith("claude") for j in judges)
        and not api_keys["OPENAI_API_KEY"]
    ):
        sys.exit("ERROR: OPENAI_API_KEY required.")

    judge_template = read_file(
        os.path.join(BASE_DIR, "prompts", "judgePromptEvidenceBoth.txt")
    )
    samples = stratified_sample(n_samples)
    print(f"\nTotal sampled: {len(samples)}; judges: {judges}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for judge_model in judges:
        print("=" * 70)
        print(f"Judge: {judge_model}")
        print("=" * 70)
        records = asyncio.run(
            judge_all(samples, judge_model, judge_template, api_keys, max_concurrent)
        )
        out_path = os.path.join(
            OUT_DIR, f"judged_{judge_model.replace('/', '_')}_{timestamp}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(records)}/{len(samples)} records -> {out_path}\n")


if __name__ == "__main__":
    main()
