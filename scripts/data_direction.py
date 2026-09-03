"""Benchmark-vs-traffic direction (BENCH_DIFF_VECTOR) from a model's train-set activations.

diff of means, eval minus deploy, per probed layer, HarmBench excluded (eval_bench's
100 tasks are HarmBench behaviors). Also the per-layer source-grouped held-out AUROC
of that direction, so the readout/steering layer can be picked per model.

Usage: python scripts/data_direction.py --acts acts_train_<model>.npz \
           --train-jsonl results/probes/trainset.jsonl --out data_direction_<model>.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def dom(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--exclude-sources", default="harmbench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr = np.load(args.acts)
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
    x_all = tr["prompt_last"][keep].astype(np.float32)

    direction, norms, aucs = [], [], []
    print("layer  held-out AUROC  in-sample d")
    for li, layer in enumerate(layers):
        x = x_all[:, li]
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(x, y, src):
            oof[b] = x[b] @ dom(x[a], y[a])
        d = dom(x, y)
        p = x @ d
        sd = np.sqrt((p[y == 1].var(ddof=1) + p[y == 0].var(ddof=1)) / 2)
        direction.append(d)
        norms.append(float(np.linalg.norm(x, axis=1).mean()))
        aucs.append(float(roc_auc_score(y, oof)))
        print(
            f"L{layer:3d}   {aucs[-1]:.3f}          {(p[y == 1].mean() - p[y == 0].mean()) / sd:.2f}"
        )
    np.savez(
        args.out,
        layers=np.array(layers),
        direction=np.stack(direction),
        mean_norm=np.array(norms),
        heldout_auroc=np.array(aucs),
    )
    best = layers[int(np.argmax(aucs))]
    print(f"best held-out layer: L{best}; wrote {args.out}")


if __name__ == "__main__":
    main()
