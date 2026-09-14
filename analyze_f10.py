"""F10 (deployment date) baseline-vs-stale analysis.

Paired McNemar per model on self-reported eval-awareness, plus a pooled test.
Pairing key is (task_id, sample_idx): the same task+valence rendered under
baseline (real date) and under F10 (same time one year ago).
"""

import glob
import json
import os
import sys
from collections import defaultdict
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metrics import bucket_conclusion

METRIC = sys.argv[1] if len(sys.argv) > 1 else "model_awareness"
SR_DIR = "results_f10/eval_bench/selfreport"


def two_sided_exact_mcnemar(b: int, c: int) -> float:
    """Exact binomial McNemar: b = baseline-no/F10-yes, c = baseline-yes/F10-no."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load(path: str) -> tuple[str, dict]:
    with open(path) as f:
        d = json.load(f)
    pairs: dict = defaultdict(dict)
    n_raw = len(d["samples"])
    for s in d["samples"]:
        concl = bucket_conclusion(
            (s.get("judge", {}).get(METRIC) or {}).get("conclusion", "")
        )
        if concl not in ("yes", "no"):
            continue  # ungradeable: drop the whole pair below
        key = (s["task_id"], s.get("sample_idx", 1))
        pairs[key][s["config_name"]] = concl == "yes"
    return d["model"], pairs, n_raw


rows = []
tot_b = tot_c = tot_n = 0
tot_base = tot_f10 = 0
for path in sorted(glob.glob(os.path.join(SR_DIR, "selfreport_*.json"))):
    model, pairs, n_raw = load(path)
    b = c = n = base_yes = f10_yes = 0
    for v in pairs.values():
        if "baseline" not in v or "F10" not in v:
            continue
        n += 1
        base_yes += v["baseline"]
        f10_yes += v["F10"]
        if not v["baseline"] and v["F10"]:
            b += 1
        elif v["baseline"] and not v["F10"]:
            c += 1
    if not n:
        continue
    p = two_sided_exact_mcnemar(b, c)
    rows.append((model, n, 100 * base_yes / n, 100 * f10_yes / n, b, c, p, 200 * n / n_raw))
    tot_b += b
    tot_c += c
    tot_n += n
    tot_base += base_yes
    tot_f10 += f10_yes

hdr = f"{'model':<22}{'n':>5}{'grade%':>8}{'baseline%':>11}{'F10 stale%':>12}{'delta':>8}{'b':>5}{'c':>5}{'p':>10}"
print(f"\nSelf-reported eval-awareness ({METRIC}); paired by task\n")
print(hdr)
print("-" * len(hdr))
for m, n, bp, fp, b, c, p, g in sorted(rows, key=lambda r: r[3] - r[2], reverse=True):
    print(
        f"{m:<22}{n:>5}{g:>7.0f}%{bp:>10.1f}%{fp:>11.1f}%"
        f"{fp - bp:>+7.1f}{b:>5}{c:>5}{p:>10.2}"
    )
if tot_n:
    bp, fp = 100 * tot_base / tot_n, 100 * tot_f10 / tot_n
    p = two_sided_exact_mcnemar(tot_b, tot_c)
    print("-" * len(hdr))
    print(
        f"{'POOLED':<22}{tot_n:>5}{'':>8}{bp:>10.1f}%{fp:>11.1f}%{fp - bp:>+7.1f}"
        f"{tot_b:>5}{tot_c:>5}{p:>10.2}"
    )
    print("\ngrade% = share of the 200 task pairs gradeable in BOTH arms")
    print("b = baseline-no -> F10-yes (stale date induced awareness)")
    print("c = baseline-yes -> F10-no")
