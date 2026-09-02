"""Head-to-head: the paper's 16-stub framing direction vs our benchmark-vs-traffic
direction, same estimator (diff of means), on

  A. the natural train set (real benchmarks vs real traffic): the framing direction
     never saw these prompts; ours is scored source-grouped out-of-fold.
  B. eval_bench baseline (all factors realistic) vs all_varied, and each single
     factor vs baseline. Neither direction has seen eval_bench.

Inputs: results/probes/{acts_train,acts_eval}.npz, trainset.jsonl, steer_directions.npz
(framing_unit at every layer) and the eval_bench response file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ORDER = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "all_varied",
    "harmbench_verbatim",
]


def dom(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-acts", default="results/probes/acts_train.npz")
    ap.add_argument("--train-jsonl", default="results/probes/trainset.jsonl")
    ap.add_argument("--eval-acts", default="results/probes/acts_eval.npz")
    ap.add_argument("--eval-responses", required=True)
    ap.add_argument("--directions", default="results/probes/steer_directions.npz")
    ap.add_argument("--exclude-sources", default="harmbench")
    ap.add_argument("--out", default="results/probes/compare_directions.json")
    args = ap.parse_args()

    tr = np.load(args.train_acts)
    meta = {}
    for line in Path(args.train_jsonl).read_text().split("\n"):
        if line:
            r = json.loads(line)
            meta[r["id"]] = r
    drop = set(filter(None, args.exclude_sources.split(",")))
    keep = np.array([meta[str(i)]["source"] not in drop for i in tr["ids"]])
    ids = [str(i) for i in tr["ids"][keep]]
    y = np.array([meta[i]["label"] for i in ids])
    src = np.array([meta[i]["source"] for i in ids])
    layers = [int(v) for v in tr["layers"]]
    framing = np.load(args.directions)["framing_unit"]  # (n_layers, hidden)

    ev = np.load(args.eval_acts)
    data = json.loads(Path(args.eval_responses).read_text())
    by_id = {r["id"]: r for r in data["samples"]}
    cfg = np.array([by_id[str(i)]["config_name"] for i in ev["ids"]])
    present = [c for c in ORDER if c in set(cfg)]

    rows = []
    print(f"{'layer':>5} | natural held-out AUROC   | eval_bench: baseline vs ...")
    print(
        f"{'':>5} | framing   ours(OOF)      | "
        + "  ".join(f"{c[:6]:>6}" for c in present)
    )
    print(
        f"{'':>5} |                          | " + "  ".join("fr/our" for _ in present)
    )
    for li, layer in enumerate(layers):
        x = tr["prompt_last"][keep][:, li].astype(np.float32)
        s_fr = x @ framing[layer]
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(x, y, src):
            oof[b] = x[b] @ dom(x[a], y[a])
        auc_fr = roc_auc_score(y, s_fr)
        auc_ours = roc_auc_score(y, oof)

        ex = ev["prompt_last"][:, li].astype(np.float32)
        d_ours = dom(x, y)
        e_fr, e_ours = ex @ framing[layer], ex @ d_ours
        base = cfg == "baseline"
        per_cfg = {}
        for c in present:
            m = base | (cfg == c)
            yy = (cfg[m] == c).astype(int)
            per_cfg[c] = (roc_auc_score(yy, e_fr[m]), roc_auc_score(yy, e_ours[m]))
        rows.append(
            {
                "layer": layer,
                "natural_auroc_framing": float(auc_fr),
                "natural_auroc_ours_oof": float(auc_ours),
                "eval_bench_vs_baseline": {
                    c: {"framing": float(a), "ours": float(b)}
                    for c, (a, b) in per_cfg.items()
                },
            }
        )
        print(
            f"L{layer:>3} | {auc_fr:.3f}     {auc_ours:.3f}          | "
            + "  ".join(f"{a:.2f}/{b:.2f}" for a, b in per_cfg.values())
        )

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
