"""Linear eval-awareness probe on pooled residual-stream features.

Design (see the probe/qwen27b branch discussion, 2026-09-01):
  labels    from the PROMPT CONFIG, not the judge. deploy=0: `baseline`.
            eval=1: `all_varied` + `harmbench_verbatim`. Single-factor cells
            (F1..F8) are never trained on; they are scored for dose-response.
  features  one (position, layer) slice of the extract_activations.py .npz
  probes    (a) sklearn LogisticRegression, L2, standardized, balanced,
                C picked by inner CV; (b) diff-of-means direction
                (mass-mean probing, Marks & Tegmark 2023) as a robustness check
  splits    GroupKFold by task_id so a task's safety/capability twins and its
            configs never straddle train/test

Readouts:
  1. held-out eval-vs-deploy AUROC per (position, layer) -> pick the best
  2. mean probe logit per config (F1..F8 dose-response, baseline vs eval)
  3. AUROC of the probe logit for predicting the CoT judge's *verbalized*
     awareness label (the representation-vs-verbalization gap)

Usage:
  python scripts/train_awareness_probe.py --acts acts_qwen27b.npz \
      --responses results/eval_bench/response/responses_qwen3.8-27b_probe_<ts>.json \
      --cot results/eval_bench/cot/cot_qwen3.8-27b_<ts>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEPLOY = {"baseline"}
EVAL = {"all_varied", "harmbench_verbatim"}
C_GRID = [0.001, 0.01, 0.1, 1.0]


def fit_lr(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold logits from a grouped 5-fold LR with an inner C sweep."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        best_c, best_auc = C_GRID[0], -1.0
        for c in C_GRID:
            inner = np.zeros(len(tr))
            for itr, ite in GroupKFold(3).split(x[tr], y[tr], groups[tr]):
                m = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
                ).fit(x[tr][itr], y[tr][itr])
                inner[ite] = m.decision_function(x[tr][ite])
            auc = roc_auc_score(y[tr], inner)
            if auc > best_auc:
                best_c, best_auc = c, auc
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000),
        ).fit(x[tr], y[tr])
        oof[te] = m.decision_function(x[te])
    return oof


def fit_dom(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold diff-of-means projections (unit direction, centered)."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        d = x[tr][y[tr] == 1].mean(0) - x[tr][y[tr] == 0].mean(0)
        d /= np.linalg.norm(d) + 1e-8
        mid = (x[tr][y[tr] == 1].mean(0) + x[tr][y[tr] == 0].mean(0)) / 2
        oof[te] = (x[te] - mid) @ d
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--cot", default="")
    ap.add_argument(
        "--positions", default="prompt_last,prompt_mean,cot_mean,answer_mean"
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    acts = np.load(args.acts)
    ids = list(acts["ids"])
    layers = list(acts["layers"])
    data = json.loads(Path(args.responses).read_text())
    by_id = {r["id"]: r for r in (data["samples"] if isinstance(data, dict) else data)}
    verbal = {}
    if args.cot:
        for r in json.loads(Path(args.cot).read_text()):
            c = (r.get("judge") or {}).get("model_awareness", {}).get("conclusion")
            if c in ("yes", "no"):
                verbal[r["id"]] = int(c == "yes")

    cfg = np.array([by_id[i]["config_name"] for i in ids])
    task = np.array([by_id[i]["task_name"] for i in ids])
    train_mask = np.isin(cfg, list(DEPLOY | EVAL))
    y = np.isin(cfg, list(EVAL)).astype(int)
    print(
        f"{len(ids)} records; train set {train_mask.sum()} "
        f"(eval={y[train_mask].sum()}, deploy={(1 - y[train_mask]).sum()}); "
        f"verbalized labels for {sum(i in verbal for i in ids)}"
    )

    # 1. sweep (position, layer): held-out AUROC on the train configs
    results = []
    for pos in args.positions.split(","):
        feats = acts[pos].astype(np.float32)
        for li, layer in enumerate(layers):
            x = feats[:, li]
            if not np.any(x):
                continue
            oof_lr = fit_lr(x[train_mask], y[train_mask], task[train_mask])
            oof_dom = fit_dom(x[train_mask], y[train_mask], task[train_mask])
            results.append(
                {
                    "position": pos,
                    "layer": int(layer),
                    "auroc_lr": roc_auc_score(y[train_mask], oof_lr),
                    "auroc_dom": roc_auc_score(y[train_mask], oof_dom),
                }
            )
            print(
                f"{pos:12s} L{int(layer):2d}  AUROC lr={results[-1]['auroc_lr']:.3f} "
                f"dom={results[-1]['auroc_dom']:.3f}",
                flush=True,
            )
    best = max(results, key=lambda r: r["auroc_lr"])
    print(f"\nBEST: {best}")

    # 2+3. refit the best probe on all train configs, score everything
    pos, li = best["position"], layers.index(best["layer"])
    x = acts[pos].astype(np.float32)[:, li]
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000),
    ).fit(x[train_mask], y[train_mask])
    logit = m.decision_function(x)
    # train-config rows get their out-of-fold logit so the table isn't in-sample
    oof = np.full(len(ids), np.nan)
    oof[train_mask] = fit_lr(x[train_mask], y[train_mask], task[train_mask])
    score = np.where(train_mask, oof, logit)

    print("\nMean probe logit by config (>0 = eval-like; train configs out-of-fold):")
    per_cfg = {}
    for c in sorted(set(cfg)):
        sel = cfg == c
        per_cfg[c] = {
            "n": int(sel.sum()),
            "mean_logit": float(score[sel].mean()),
            "frac_eval": float((score[sel] > 0).mean()),
        }
        print(
            f"  {c:20s} n={sel.sum():4d}  mean={score[sel].mean():+.2f}  "
            f"frac>0={(score[sel] > 0).mean():.2f}"
        )

    gap = None
    if verbal:
        vsel = np.array([i in verbal for i in ids])
        vy = np.array([verbal[i] for i in ids if i in verbal])
        if 0 < vy.sum() < len(vy):
            auc_v = roc_auc_score(vy, score[vsel])
            r = np.corrcoef(vy, score[vsel])[0, 1]
            gap = {
                "n": int(vsel.sum()),
                "n_verbalized": int(vy.sum()),
                "auroc": auc_v,
                "r": r,
            }
            print(
                f"\nProbe vs VERBALIZED awareness (sonnet judge): n={vsel.sum()} "
                f"verbalized={vy.sum()}  AUROC={auc_v:.3f}  point-biserial r={r:.3f}"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "sweep": results,
                    "best": best,
                    "per_config": per_cfg,
                    "verbal_gap": gap,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
