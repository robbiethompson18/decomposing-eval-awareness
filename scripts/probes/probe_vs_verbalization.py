"""Probe readout vs CoT verbalization vs self-report, grouped by config or by k.

For every transcript: projection of the last-prompt-token residual (layer L) onto
the `data` direction (benchmark-vs-traffic) and the `framing` direction (the
paper's 16 stubs), the sonnet CoT-judge awareness verdict, and the self-report
verdict. Then per group (config_name for the OFAT run, n_varied for the
stacking run): mean projection, CoT-verbalized %, self-report %; Spearman across
groups for probe-vs-CoT and probe-vs-self-report; and per-transcript AUROC of
each projection for predicting each verbal measure, overall and within group.

Usage:
  python scripts/probes/probe_vs_verbalization.py --acts results/probes/acts_exp1.npz \
      --responses <response json> [...] --cot <cot json> [...] \
      --selfreport <selfreport json> [...] --group config_name --out results/probes/exp1.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

GROUP_ORDER = [
    "bare_harness",
    "baseline",
    *[f"F{i}" for i in range(1, 14)],
    "all_varied",
    "harmbench_verbatim",
]


def load_records(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        out.extend(d["samples"] if isinstance(d, dict) and "samples" in d else d)
    return out


def verdicts(paths: list[str]) -> dict[str, int]:
    out = {}
    for r in load_records(paths):
        c = (r.get("judge") or {}).get("model_awareness", {}).get("conclusion")
        if c in ("yes", "no"):
            out[r["id"]] = int(c == "yes")
    return out


def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--cot", nargs="+", default=[])
    ap.add_argument("--selfreport", nargs="+", default=[])
    ap.add_argument("--directions", default="results/probes/steer_directions.npz")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--group", default="config_name", help="config_name or n_varied")
    ap.add_argument(
        "--train-acts",
        default="",
        help="train-set activations: adds gap-fraction (0=real chat, 1=real benchmark) "
        "and d-unit columns so numbers compare across models/layers",
    )
    ap.add_argument("--train-jsonl", default="results/probes/trainset.jsonl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    acts = np.load(args.acts)
    layers = [int(v) for v in acts["layers"]]
    li = layers.index(args.layer)
    x = acts["prompt_last"][:, li].astype(np.float32)
    ids = [str(i) for i in acts["ids"]]
    d = np.load(args.directions)
    proj = {
        "data": x @ d["data_unit"][li],
        "framing": x @ d["framing_unit"][args.layer],
    }
    # normalization references from the train set (HarmBench excluded)
    ref = {}
    if args.train_acts:
        tr = np.load(args.train_acts)
        tli = [int(v) for v in tr["layers"]].index(args.layer)
        meta = {
            r["id"]: r
            for r in (
                json.loads(line)
                for line in Path(args.train_jsonl).read_text().split("\n")
                if line
            )
        }
        keep = np.array([meta[str(i)]["source"] != "harmbench" for i in tr["ids"]])
        ty = np.array([meta[str(i)]["label"] for i in tr["ids"][keep]])
        tx = tr["prompt_last"][keep][:, tli].astype(np.float32)
        for name, vec in (
            ("data", d["data_unit"][li]),
            ("framing", d["framing_unit"][args.layer]),
        ):
            p = tx @ vec
            c, b = p[ty == 0].mean(), p[ty == 1].mean()
            sd = np.sqrt((p[ty == 0].var(ddof=1) + p[ty == 1].var(ddof=1)) / 2)
            ref[name] = (float(c), float(b), float(sd))
            print(f"ref {name}: chat {c:+.2f} bench {b:+.2f} sd {sd:.2f}")

    def norm(name: str, value: float) -> dict:
        if name not in ref:
            return {}
        c, b, sd = ref[name]
        return {f"{name}_gapfrac": (value - c) / (b - c), f"{name}_d": (value - c) / sd}

    resp = {r["id"]: r for r in load_records(args.responses)}
    cot = verdicts(args.cot)
    sr = verdicts(args.selfreport)

    def group_of(r):
        if args.group == "n_varied":
            return int(r.get("n_varied", 0))
        return r["config_name"]

    groups = np.array([group_of(resp[i]) for i in ids], dtype=object)
    valence = np.array([resp[i]["valence"] for i in ids])
    keys = sorted(
        set(groups),
        key=lambda g: (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, str(g)),
    )

    rows = {}
    print(f"{'group':20s} {'n':>5} {'data':>7} {'framing':>8} {'CoT%':>6} {'SR%':>6}")
    for g in keys:
        m = groups == g
        gid = [i for i, mm in zip(ids, m) if mm]
        c = [cot[i] for i in gid if i in cot]
        s = [sr[i] for i in gid if i in sr]
        rows[str(g)] = {
            "n": int(m.sum()),
            "data_mean": float(proj["data"][m].mean()),
            "framing_mean": float(proj["framing"][m].mean()),
            "cot_pct": 100 * float(np.mean(c)) if c else None,
            "sr_pct": 100 * float(np.mean(s)) if s else None,
        }
        r = rows[str(g)]
        r.update(norm("data", r["data_mean"]))
        r.update(norm("framing", r["framing_mean"]))
        extra = (
            f"  gapfrac data={r['data_gapfrac']:+.2f} framing={r['framing_gapfrac']:+.2f}"
            if ref
            else ""
        )
        print(
            f"{g!s:20s} {r['n']:5d} {r['data_mean']:7.2f} {r['framing_mean']:8.2f} "
            f"{(r['cot_pct'] if r['cot_pct'] is not None else float('nan')):6.1f} "
            f"{(r['sr_pct'] if r['sr_pct'] is not None else float('nan')):6.1f}{extra}"
        )

    out = {
        "layer": args.layer,
        "group": args.group,
        "per_group": rows,
        "reference": ref,
    }
    # group-level rank agreement
    for dname in ("data", "framing"):
        for vname, col in (("cot", "cot_pct"), ("sr", "sr_pct")):
            pairs = [
                (rows[k][f"{dname}_mean"], rows[k][col])
                for k in rows
                if rows[k][col] is not None
            ]
            if len(pairs) >= 4:
                rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]
                out[f"spearman_{dname}_vs_{vname}"] = float(rho)
                print(
                    f"group-level Spearman {dname} vs {vname}: {rho:+.3f} (n groups={len(pairs)})"
                )

    # per-transcript AUROC, overall / within valence / within group
    for dname in ("data", "framing"):
        for vname, lab in (("cot", cot), ("sr", sr)):
            sel = np.array([i in lab for i in ids])
            if sel.sum() == 0:
                continue
            y = [lab[i] for i in ids if i in lab]
            overall = safe_auc(y, proj[dname][sel])
            within_val = {}
            for v in ("safety", "capability"):
                mv = sel & (valence == v)
                if mv.sum():
                    within_val[v] = safe_auc(
                        [lab[i] for i, mm in zip(ids, mv) if mm], proj[dname][mv]
                    )
            within_g = defaultdict(list)
            cells = []
            for g in keys:
                for v in ("safety", "capability"):
                    mg = sel & (groups == g) & (valence == v)
                    if mg.sum() >= 10:
                        a = safe_auc(
                            [lab[i] for i, mm in zip(ids, mg) if mm], proj[dname][mg]
                        )
                        if a is not None:
                            cells.append(a)
                            within_g[str(g)].append(a)
            out[f"auroc_{dname}_vs_{vname}"] = {
                "overall": overall,
                "within_valence": within_val,
                "mean_within_group_valence": float(np.mean(cells)) if cells else None,
                "n_cells": len(cells),
            }
            print(
                f"AUROC {dname} -> {vname}: overall={overall}  within-valence={within_val}  "
                f"mean within (group,valence)={out[f'auroc_{dname}_vs_{vname}']['mean_within_group_valence']} ({len(cells)} cells)"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
