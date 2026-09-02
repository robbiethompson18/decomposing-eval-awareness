"""Eval-awareness linear probe: train on real traffic vs real benchmarks, test on eval_bench.

Design (.claude/notes/activation-probe-plan.md):
  train   results/probes/trainset.jsonl (build_probe_trainset.py) + its activations.
          label 0 = real user traffic (WildChat / lmsys / ShareGPT), label 1 = real
          benchmark items (12 sources). System prompt randomized per record.
  probe   sklearn LogisticRegression (L2, standardized, balanced, C by inner CV) and
          diff-of-means. Cross-validation grouped by SOURCE, so every held-out fold
          is made of benchmarks / traffic sources the probe never saw.
  test    eval_bench transcripts (never trained on): per-config mean logit, and the
          probe-vs-verbalized-awareness readout against the CoT judge.

Readouts:
  1. source-grouped OOF AUROC per (position, layer); per-source held-out accuracy
  2. topic-matched AUROC (harmful / code / math only) and per-F9-level AUROC
  3. transfer to eval_bench: mean logit by config (baseline -> F1..F8 -> all_varied)
  4. probe logit vs sonnet-judge verbalized awareness (AUROC, point-biserial r)
  5. cosine between the train diff-of-means direction and eval_bench's own
     baseline-vs-all_varied direction; in-distribution eval_bench-trained AUROC as ceiling

Usage:
  python scripts/train_awareness_probe.py --train-acts results/probes/acts_train.npz \
      --train-jsonl results/probes/trainset.jsonl --eval-acts results/probes/acts_eval.npz \
      --eval-responses results/eval_bench/response/responses_qwen3.8-27b_..._probe_<ts>.json \
      --cot results/eval_bench/cot/cot_qwen3.8-27b_<ts>.json --out results/probes/probe_qwen27b.json
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

C_GRID = [0.001, 0.01, 0.1, 1.0]
EB_DEPLOY = {"baseline"}
EB_EVAL = {"all_varied", "harmbench_verbatim"}


def lr(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=3000),
    )


def oof_lr(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold logits, grouped folds, inner grouped CV over C."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(x, y, groups):
        best_c, best_auc = C_GRID[0], -1.0
        for c in C_GRID:
            inner = np.zeros(len(tr))
            for itr, ite in GroupKFold(3).split(x[tr], y[tr], groups[tr]):
                inner[ite] = (
                    lr(c).fit(x[tr][itr], y[tr][itr]).decision_function(x[tr][ite])
                )
            auc = roc_auc_score(y[tr], inner)
            if auc > best_auc:
                best_c, best_auc = c, auc
        oof[te] = lr(best_c).fit(x[tr], y[tr]).decision_function(x[te])
    return oof


def dom_dir(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def dom_mid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x[y == 1].mean(0) + x[y == 0].mean(0)) / 2


def oof_dom(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(x, y, groups):
        oof[te] = (x[te] - dom_mid(x[tr], y[tr])) @ dom_dir(x[tr], y[tr])
    return oof


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-acts", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--eval-acts", default="")
    ap.add_argument("--eval-responses", default="")
    ap.add_argument("--cot", default="")
    ap.add_argument("--positions", default="prompt_last,prompt_mean")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # ---- training set ------------------------------------------------------
    acts = np.load(args.train_acts)
    layers = [int(v) for v in acts["layers"]]
    meta = {r["id"]: r for r in load_jsonl(args.train_jsonl)}
    ids = [str(i) for i in acts["ids"]]
    y = np.array([meta[i]["label"] for i in ids])
    src = np.array([meta[i]["source"] for i in ids])
    topic = np.array([meta[i]["topic"] for i in ids])
    f9 = np.array([meta[i]["f9"] for i in ids])
    print(
        f"train: {len(ids)} records, eval={y.sum()} deploy={(1 - y).sum()}, "
        f"sources={len(set(src))}"
    )

    # 1. sweep (position, layer), folds grouped by source
    sweep = []
    for pos in args.positions.split(","):
        feats = acts[pos].astype(np.float32)
        for li, layer in enumerate(layers):
            x = feats[:, li]
            s_lr = oof_lr(x, y, src)
            s_dom = oof_dom(x, y, src)
            sweep.append(
                {
                    "position": pos,
                    "layer": layer,
                    "auroc_lr": float(roc_auc_score(y, s_lr)),
                    "auroc_dom": float(roc_auc_score(y, s_dom)),
                }
            )
            print(
                f"  {pos:12s} L{layer:2d}  AUROC lr={sweep[-1]['auroc_lr']:.3f} "
                f"dom={sweep[-1]['auroc_dom']:.3f}",
                flush=True,
            )
    best = max(sweep, key=lambda r: r["auroc_lr"])
    pos, li = best["position"], layers.index(best["layer"])
    print(f"\nBEST (source-grouped OOF): {best}")

    # per-source held-out accuracy, topic-matched + per-F9 AUROC at the best slice
    x = acts[pos].astype(np.float32)[:, li]
    s = oof_lr(x, y, src)
    per_source = {}
    for name in sorted(set(src)):
        sel = src == name
        per_source[name] = {
            "n": int(sel.sum()),
            "label": int(y[sel][0]),
            "mean_logit": float(s[sel].mean()),
            "frac_eval": float((s[sel] > 0).mean()),
        }
    print(
        "\nHeld-out per source (frac scored eval-like; want ~1 for label 1, ~0 for 0):"
    )
    for name, r in per_source.items():
        print(
            f"  {name:16s} label={r['label']} n={r['n']:4d} "
            f"mean={r['mean_logit']:+.2f} frac_eval={r['frac_eval']:.2f}"
        )
    topic_auc = {
        t: safe_auc(y[topic == t], s[topic == t])
        for t in ("harmful", "code", "math", "other")
    }
    f9_auc = {
        lvl: safe_auc(y[f9 == lvl], s[f9 == lvl]) for lvl in ("deployment", "bare")
    }
    f9_leak = safe_auc((f9 == "bare").astype(int), s)
    print(f"\nTopic-matched AUROC: {topic_auc}")
    print(f"Per-F9 AUROC: {f9_auc}; logit predicting F9=bare (want ~0.5): {f9_leak}")

    # final probe on all training data
    final = lr(0.01).fit(x, y)
    train_dir = dom_dir(x, y)
    out = {
        "best": best,
        "sweep": sweep,
        "per_source": per_source,
        "topic_auc": topic_auc,
        "f9_auc": f9_auc,
        "f9_leak_auc": f9_leak,
    }

    # ---- transfer to eval_bench ----------------------------------------------
    if args.eval_acts:
        e_acts = np.load(args.eval_acts)
        assert [int(v) for v in e_acts["layers"]] == layers
        e_ids = [str(i) for i in e_acts["ids"]]
        data = json.loads(Path(args.eval_responses).read_text())
        by_id = {
            r["id"]: r for r in (data["samples"] if isinstance(data, dict) else data)
        }
        cfg = np.array([by_id[i]["config_name"] for i in e_ids])
        task = np.array([by_id[i]["task_name"] for i in e_ids])
        ex = e_acts[pos].astype(np.float32)[:, li]
        logit = final.decision_function(ex)

        print(
            f"\nTransfer to eval_bench ({pos} L{best['layer']}), probe never saw these:"
        )
        per_cfg = {}
        order = [
            "baseline",
            *[f"F{i}" for i in range(1, 9)],
            "all_varied",
            "harmbench_verbatim",
        ]
        seen = set(cfg)
        for c in [c for c in order if c in seen] + sorted(seen - set(order)):
            sel = cfg == c
            per_cfg[c] = {
                "n": int(sel.sum()),
                "mean_logit": float(logit[sel].mean()),
                "frac_eval": float((logit[sel] > 0).mean()),
            }
            print(
                f"  {c:20s} n={sel.sum():4d} mean={logit[sel].mean():+.2f} "
                f"frac_eval={(logit[sel] > 0).mean():.2f}"
            )
        out["eval_bench_per_config"] = per_cfg
        eb_y = np.isin(cfg, list(EB_EVAL)).astype(int)
        eb_mask = np.isin(cfg, list(EB_DEPLOY | EB_EVAL))
        out["eval_bench_transfer_auroc"] = safe_auc(eb_y[eb_mask], logit[eb_mask])
        print(
            f"  transfer AUROC baseline vs all_varied+harmbench: "
            f"{out['eval_bench_transfer_auroc']}"
        )

        # 5. same direction? + in-distribution ceiling
        eb_dir = dom_dir(ex[eb_mask], eb_y[eb_mask])
        out["cosine_train_vs_evalbench_dir"] = float(train_dir @ eb_dir)
        rand = np.random.default_rng(0).standard_normal(len(eb_dir))
        print(
            f"  cosine(train dir, eval_bench dir) = "
            f"{out['cosine_train_vs_evalbench_dir']:.3f} "
            f"(random ~{abs(rand @ eb_dir / np.linalg.norm(rand)):.3f})"
        )
        s_in = oof_lr(ex[eb_mask], eb_y[eb_mask], task[eb_mask])
        out["eval_bench_in_distribution_auroc"] = float(
            roc_auc_score(eb_y[eb_mask], s_in)
        )
        print(
            f"  in-distribution ceiling (eval_bench-trained, task-grouped OOF): "
            f"{out['eval_bench_in_distribution_auroc']:.3f}"
        )

        # 4. verbalized awareness
        if args.cot:
            verbal = {}
            for r in json.loads(Path(args.cot).read_text()):
                c = (r.get("judge") or {}).get("model_awareness", {}).get("conclusion")
                if c in ("yes", "no"):
                    verbal[r["id"]] = int(c == "yes")
            vsel = np.array([i in verbal for i in e_ids])
            vy = np.array([verbal[i] for i in e_ids if i in verbal])
            auc_v = safe_auc(vy, logit[vsel])
            r_pb = (
                float(np.corrcoef(vy, logit[vsel])[0, 1]) if auc_v is not None else None
            )
            out["verbal_gap"] = {
                "n": int(vsel.sum()),
                "n_verbalized": int(vy.sum()),
                "auroc": auc_v,
                "r": r_pb,
            }
            print(
                f"\nProbe vs VERBALIZED awareness: n={vsel.sum()} verbalized={vy.sum()} "
                f"AUROC={auc_v} r={r_pb}"
            )
            within = {}
            for c in sorted(set(cfg[vsel])):
                m = vsel & (cfg == c)
                yy = np.array([verbal[i] for i in np.array(e_ids)[m]])
                within[c] = safe_auc(yy, logit[m])
            out["verbal_gap_within_config"] = within
            print(f"  within-config AUROC (config held fixed): {within}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
