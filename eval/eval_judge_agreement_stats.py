"""Agreement stats between judge models re-judged by eval_judge_agreement.py.

Usage:
    python eval/eval_judge_agreement_stats.py <judged_file> [<judged_file> ...]

Each input file holds the same sample set judged by one judge (filename
judged_{judge}_{timestamp}.json). The stored GPT-5 judgment carried on each
record ("judge_stored") is included as pseudo-judge "gpt-5-stored" when
present. For every judge pair and every metric, prints raw bucket agreement,
Cohen's kappa over the records both judges called yes/no, and a confusion
matrix. Writes model_awareness disagreements (vs the first judge listed) to
results/eval_bench/judge_agreement/disagreements_{timestamp}.json.
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import bucket_conclusion, extract_conclusion

METRICS = ["model_awareness", "hypothetical_recognition", "task_performance"]
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "eval_bench",
    "judge_agreement",
)


def load_judges(paths: list[str]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """judge_name -> {(target_model, id): record}."""
    judges: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for path in paths:
        m = re.match(r"judged_(.+)_\d{8}_\d{6}\.json$", os.path.basename(path))
        name = m.group(1) if m else os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        judges[name] = {(r["target_model"], r["id"]): r for r in records}
        stored = {
            k: {**r, "judge": r["judge_stored"]}
            for k, r in judges[name].items()
            if r.get("judge_stored")
        }
        if stored and "gpt-5-stored" not in judges:
            judges["gpt-5-stored"] = stored
    return judges


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over yes/no labels."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    a_counts, b_counts = Counter(p[0] for p in pairs), Counter(p[1] for p in pairs)
    pe = sum(a_counts[c] * b_counts[c] for c in ("yes", "no")) / (n * n)
    if pe == 1.0:
        return None  # no marginal variation; kappa undefined
    return (po - pe) / (1 - pe)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    judges = load_judges(paths)
    names = list(judges)
    common = set.intersection(*(set(j) for j in judges.values()))
    print(f"Judges: {names}")
    print(f"Common records: {len(common)}\n")

    def bucket(name: str, key: tuple[str, str], metric: str) -> str:
        return bucket_conclusion(
            extract_conclusion(judges[name][key].get("judge") or {}, metric)
        )

    for metric in METRICS:
        print("=" * 74)
        print(metric)
        print("=" * 74)
        for name in names:
            counts = Counter(bucket(name, k, metric) for k in common)
            dist = ", ".join(
                f"{b}={counts.get(b, 0)}" for b in ("yes", "no", "n/a", "invalid")
            )
            print(f"  {name:16s} {dist}")
        print()
        for a, b in combinations(names, 2):
            labels = [(bucket(a, k, metric), bucket(b, k, metric)) for k in common]
            raw = sum(1 for x, y in labels if x == y) / len(labels)
            yn = [
                (x, y) for x, y in labels if x in ("yes", "no") and y in ("yes", "no")
            ]
            k_val = kappa(yn)
            k_str = f"{k_val:.3f}" if k_val is not None else "  n/a"
            cm = Counter(labels)
            cm_str = "  ".join(
                f"{x}/{y}={cm[(x, y)]}"
                for x in ("yes", "no")
                for y in ("yes", "no")
                if cm[(x, y)]
            )
            print(
                f"  {a:16s} vs {b:16s} raw={raw:.1%}  kappa={k_str} "
                f"(n_yesno={len(yn)})   {cm_str}"
            )
        print()

    # Disagreement dump for the headline metric, vs the first judge given
    ref = names[0]
    disagreements = []
    for k in sorted(common):
        buckets = {n: bucket(n, k, "model_awareness") for n in names}
        if len(set(buckets.values())) > 1:
            rec = judges[ref][k]
            disagreements.append(
                {
                    "target_model": k[0],
                    "id": k[1],
                    "buckets": buckets,
                    "judges": {
                        n: (judges[n][k].get("judge") or {}).get("model_awareness")
                        for n in names
                    },
                    "response": rec.get("response"),
                }
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"disagreements_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)
    print(
        f"model_awareness disagreements: {len(disagreements)}/{len(common)} -> {out_path}"
    )


if __name__ == "__main__":
    main()
