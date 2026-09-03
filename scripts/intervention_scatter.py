"""Per-intervention scatter: self-report % vs vector d-units vs CoT-verbalized %, across models.

Interventions = the 17 OFAT cells (baseline, F1–F13, bare_harness, all_varied,
harmbench_verbatim; safety arm) + the ladder levels k=0..8 (both valences), per model,
from the JSONs written by scripts/probe_vs_verbalization.py with --train-acts. Vector
values are d-units ((proj − train chat mean) / train pooled SD) so models share a scale.
Writes a CSV of rows, R² for every pair (pooled and per model), and three PNG scatters.

Usage:
  python scripts/intervention_scatter.py \
      --model qwen3.8-27b=results/probes/exp1_qwen27b_L28.json:results/probes/exp2_ladder_qwen27b_L28.json \
      --model gpt-oss-20b=results/probes/gptoss/exp1_L12.json:results/probes/gptoss/ladder_L12.json \
      --vector framing --out-dir results/probes/scatter
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OFAT_ORDER = [
    "bare_harness",
    "baseline",
    *[f"F{i}" for i in range(1, 14)],
    "all_varied",
    "harmbench_verbatim",
]


def rows_for(model: str, exp1: str, ladder: str, vector: str) -> list[dict]:
    out = []
    for path, kind in ((exp1, "ofat"), (ladder, "ladder")):
        j = json.loads(Path(path).read_text())
        for g, r in j["per_group"].items():
            key = f"{vector}_d"
            if key not in r or r["sr_pct"] is None:
                continue
            out.append(
                {
                    "model": model,
                    "set": kind,
                    "intervention": g if kind == "ofat" else f"k={g}",
                    "n": r["n"],
                    "sr_pct": r["sr_pct"],
                    "cot_pct": r["cot_pct"]
                    if r["cot_pct"] is not None
                    else float("nan"),
                    "vec_d": r[key],
                }
            )
    return out


def r2(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", action="append", required=True, help="key=exp1.json:ladder.json"
    )
    ap.add_argument("--vector", default="framing", help="framing (PAV) or data (BDV)")
    ap.add_argument("--out-dir", default="results/probes/scatter")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.model:
        model, _, files = spec.partition("=")
        exp1, _, ladder = files.partition(":")
        rows += rows_for(model, exp1, ladder, args.vector)
    with open(out / f"interventions_{args.vector}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = {
        "sr_pct": "self-report %",
        "vec_d": f"{args.vector} vector (d-units)",
        "cot_pct": "CoT-verbalized %",
    }
    models = sorted({r["model"] for r in rows})
    stats = {}
    print(f"R² ({len(rows)} interventions, vector={args.vector})")
    for a, b in combinations(cols, 2):
        pooled = r2([r[a] for r in rows], [r[b] for r in rows])
        per = {
            m: r2(
                [r[a] for r in rows if r["model"] == m],
                [r[b] for r in rows if r["model"] == m],
            )
            for m in models
        }
        stats[f"{a}~{b}"] = {"pooled": pooled, **per}
        print(
            f"  {cols[a]} ~ {cols[b]}: pooled {pooled:.2f}  "
            + "  ".join(f"{m} {v:.2f}" for m, v in per.items())
        )

    markers = {"ofat": "o", "ladder": "s"}
    for a, b in combinations(cols, 2):
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, m in enumerate(models):
            for kind, mk in markers.items():
                sub = [r for r in rows if r["model"] == m and r["set"] == kind]
                ax.scatter(
                    [r[a] for r in sub],
                    [r[b] for r in sub],
                    marker=mk,
                    s=28,
                    color=f"C{i}",
                    alpha=0.8,
                    label=f"{m} {kind}",
                )
                for r in sub:
                    if r["set"] == "ofat" and r["intervention"] in (
                        "baseline",
                        "all_varied",
                        "bare_harness",
                        "harmbench_verbatim",
                    ):
                        ax.annotate(
                            r["intervention"], (r[a], r[b]), fontsize=6, alpha=0.7
                        )
        ax.set_xlabel(cols[a])
        ax.set_ylabel(cols[b])
        ax.set_title(f"per intervention, R² pooled {stats[f'{a}~{b}']['pooled']:.2f}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"scatter_{a}_vs_{b}_{args.vector}.png", dpi=150)
        plt.close(fig)
    (out / f"r2_{args.vector}.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
