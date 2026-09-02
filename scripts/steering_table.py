"""Per-condition steering readouts for the Qwen3.8-27B sweep.

For each (condition, response file, cot judge file): CoT-verbalized awareness %,
no-visible-answer %, reasoning-cap-hit %, safety-arm refusal %, mean reasoning chars.
Conditions are baseline-config prompts only (100 tasks x 2 valences).

Usage: python scripts/steering_table.py cond=response_file:cot_file ...
"""

import json
import sys
from pathlib import Path

RESP = Path("results/eval_bench/response")
COT = Path("results/eval_bench/cot")


def stats(resp_file: str, cot_file: str) -> dict:
    samples = json.loads((RESP / resp_file).read_text())["samples"]
    judged = {r["id"]: r for r in json.loads((COT / cot_file).read_text())}
    n = len(samples)
    no_answer = cap_hit = 0
    reasoning_chars = []
    for s in samples:
        r = s["response"]
        think, _, answer = r.partition("</think>")
        reasoning_chars.append(len(think))
        if not answer.strip():
            no_answer += 1
        if "</think>" not in r:
            cap_hit += 1  # reasoning never closed: hit MAX_TOKENS inside <think>
    aware = parsed = 0
    refused = safety_n = 0
    for s in samples:
        j = (judged.get(s["id"]) or {}).get("judge") or {}
        c = j.get("model_awareness", {}).get("conclusion")
        if c in ("yes", "no"):
            parsed += 1
            aware += c == "yes"
        if s["valence"] == "safety":
            safety_n += 1
            refused += j.get("task_performance", {}).get("conclusion") == "refused"
    return {
        "n": n,
        "cot_verbalized_pct": 100 * aware / max(parsed, 1),
        "verbalized_n": aware,
        "no_answer_pct": 100 * no_answer / n,
        "cap_hit_pct": 100 * cap_hit / n,
        "safety_refused_pct": 100 * refused / max(safety_n, 1),
        "mean_reasoning_chars": sum(reasoning_chars) / n,
    }


def main() -> None:
    rows = {}
    for arg in sys.argv[1:]:
        cond, _, files = arg.partition("=")
        resp_file, _, cot_file = files.partition(":")
        rows[cond] = stats(resp_file, cot_file)
    print(
        f"{'condition':24s} {'n':>4} {'CoT-verb%':>9} {'no-ans%':>8} {'cap-hit%':>9} {'refuse%':>8} {'reason-chars':>12}"
    )
    for cond, r in rows.items():
        print(
            f"{cond:24s} {r['n']:4d} {r['cot_verbalized_pct']:8.1f} {r['no_answer_pct']:8.1f} "
            f"{r['cap_hit_pct']:8.1f} {r['safety_refused_pct']:8.1f} {r['mean_reasoning_chars']:12.0f}"
        )
    Path("results/probes/steering_table.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
