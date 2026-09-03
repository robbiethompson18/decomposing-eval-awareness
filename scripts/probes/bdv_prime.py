"""BDV' / BDV'': the benchmark-vs-chat direction read at the last response token
(BDV') or the last think token (BDV'') instead of the last prompt token (BDV).

Inputs (all with the same 17 probed layers):
  --train-acts   activations of the 3,600 train prompts WITH the model's own
                 responses (extract_activations.py on trainset_responses_*.json)
  --eval-acts    OFAT 2,100-transcript activations (for Cohen's d vs CoT / self-report)
  --stack-acts   stacking half-A activations (for the k ladder)
  --directions   steer_directions.npz (PAV = framing_unit, for the comparison rows)

Per position P in {prompt_last, think_last, response_last}:
  1. BDV_P per layer = mean(bench) - mean(chat), HarmBench excluded; source-grouped
     5-fold out-of-fold AUROC per layer + in-sample AUROC/d at L28
  2. cosine(BDV_prompt_last, BDV_P) per layer
  3. at L28: gap-fraction ladder over k, Pearson r vs k / CoT% / self-report%,
     within-k AUROC vs self-report and CoT
  4. OFAT: Cohen's d (aware - not) vs CoT judge and vs self-report, pooled + per valence

Each readout uses the SAME token position the direction was built from.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

POSITIONS = ["prompt_last", "think_last", "response_last"]
NAMES = {"prompt_last": "BDV", "think_last": "BDV''", "response_last": "BDV'"}


def dom(x, y):
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / s)


def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def verdicts(path):
    d = json.loads(Path(path).read_text())
    recs = (
        d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
    )
    out = {}
    for r in recs:
        c = (r.get("judge") or {}).get("model_awareness", {}).get("conclusion")
        if c in ("yes", "no"):
            out[r["id"]] = int(c == "yes")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-acts", required=True)
    ap.add_argument("--train-jsonl", default="results/probes/trainset.jsonl")
    ap.add_argument("--eval-acts", required=True)
    ap.add_argument("--eval-responses", required=True)
    ap.add_argument("--eval-cot", required=True)
    ap.add_argument("--eval-selfreport", required=True)
    ap.add_argument("--stack-acts", required=True)
    ap.add_argument("--stack-responses", required=True)
    ap.add_argument("--stack-cot", required=True)
    ap.add_argument("--stack-selfreport", required=True)
    ap.add_argument("--directions", default="results/probes/steer_directions.npz")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default="results/probes/bdv_prime_qwen27b.json")
    args = ap.parse_args()

    tr = np.load(args.train_acts)
    layers = [int(v) for v in tr["layers"]]
    li = layers.index(args.layer)
    meta = {}
    for line in Path(args.train_jsonl).read_text().split("\n"):
        if line:
            r = json.loads(line)
            meta[r["id"]] = r
    tids = [str(i) for i in tr["ids"]]
    keep = np.array([meta[i]["source"] != "harmbench" for i in tids])
    y = np.array([meta[i]["label"] for i in tids])[keep]
    src = np.array([meta[i]["source"] for i in tids])[keep]
    pav = np.load(args.directions)["framing_unit"]

    # ---- 1 + 2: directions, held-out AUROC per layer, cosine to prompt-last BDV
    dirs = {}  # position -> (n_layers_probed, hidden)
    out = {"layer": args.layer, "positions": {}}
    print(
        f"{'layer':>5} | "
        + " | ".join(f"{NAMES[p]:>7} OOF" for p in POSITIONS)
        + " | cos(BDV,BDV') cos(BDV,BDV'')"
    )
    for p in POSITIONS:
        dirs[p] = np.stack(
            [dom(tr[p][keep][:, j].astype(np.float32), y) for j in range(len(layers))]
        )
    for j, layer in enumerate(layers):
        row = f"L{layer:>3} | "
        for p in POSITIONS:
            x = tr[p][keep][:, j].astype(np.float32)
            oof = np.zeros(len(y))
            for a, b in GroupKFold(5).split(x, y, src):
                oof[b] = x[b] @ dom(x[a], y[a])
            auc = roc_auc_score(y, oof)
            out["positions"].setdefault(p, {}).setdefault("oof_auroc_by_layer", {})[
                str(layer)
            ] = float(auc)
            row += f"{auc:10.3f} | "
        cos1 = float(dirs["prompt_last"][j] @ dirs["response_last"][j])
        cos2 = float(dirs["prompt_last"][j] @ dirs["think_last"][j])
        out.setdefault("cosine_by_layer", {})[str(layer)] = {
            "BDV_vs_BDVprime": cos1,
            "BDV_vs_BDVpp": cos2,
        }
        print(row + f"{cos1:+.3f}        {cos2:+.3f}")

    # reference means at L28 for gap fraction, in-sample d
    ref = {}
    for p in POSITIONS:
        proj = tr[p][keep][:, li].astype(np.float32) @ dirs[p][li]
        ref[p] = (float(proj[y == 0].mean()), float(proj[y == 1].mean()))
        out["positions"][p]["insample_L"] = {
            "auroc": float(roc_auc_score(y, proj)),
            "d": cohen_d(proj[y == 1], proj[y == 0]),
            "chat_mean": ref[p][0],
            "bench_mean": ref[p][1],
        }
    proj_pav = tr["prompt_last"][keep][:, li].astype(np.float32) @ pav[args.layer]
    ref["PAV"] = (float(proj_pav[y == 0].mean()), float(proj_pav[y == 1].mean()))

    def gap(p, v):
        c, b = ref[p]
        return (v - c) / (b - c)

    # ---- 3: ladder on stacking half A
    st = np.load(args.stack_acts)
    sids = [str(i) for i in st["ids"]]
    sresp = {
        r["id"]: r
        for r in json.loads(Path(args.stack_responses).read_text())["samples"]
    }
    k = np.array([int(sresp[i].get("n_varied", 0)) for i in sids])
    scot, ssr = verdicts(args.stack_cot), verdicts(args.stack_selfreport)
    ks = np.arange(9)

    def rate(lab, m):
        v = [lab[i] for i, mm in zip(sids, m) if mm and i in lab]
        return 100 * float(np.mean(v)) if v else np.nan

    cot_k = np.array([rate(scot, k == kk) for kk in ks])
    sr_k = np.array([rate(ssr, k == kk) for kk in ks])
    series = {p: st[p][:, li].astype(np.float32) @ dirs[p][li] for p in POSITIONS}
    series["PAV"] = st["prompt_last"][:, li].astype(np.float32) @ pav[args.layer]
    print(
        f"\nLadder at L{args.layer} (gap fraction, 0 = real chat, 1 = real benchmark); CoT% / SR% per k:"
    )
    print("k   " + "  ".join(f"{NAMES.get(p, p):>7}" for p in series) + "   CoT%   SR%")
    out["ladder"] = {}
    for p, s in series.items():
        g = np.array([gap(p, s[k == kk].mean()) for kk in ks])
        within_sr = [
            safe_auc(
                [ssr[i] for i, mm in zip(sids, k == kk) if mm and i in ssr],
                s[(k == kk) & np.array([i in ssr for i in sids])],
            )
            for kk in ks
        ]
        within_cot = [
            safe_auc(
                [scot[i] for i, mm in zip(sids, k == kk) if mm and i in scot],
                s[(k == kk) & np.array([i in scot for i in sids])],
            )
            for kk in ks
        ]
        out["ladder"][NAMES.get(p, p)] = {
            "gap_fraction_by_k": [float(v) for v in g],
            "r_k": float(pearsonr(g, ks)[0]),
            "r_cot": float(pearsonr(g, cot_k)[0]),
            "r_sr": float(pearsonr(g, sr_k)[0]),
            "within_k_auroc_sr": within_sr,
            "within_k_auroc_cot": within_cot,
            "mean_within_k_auroc_sr": float(
                np.mean([v for v in within_sr if v is not None])
            ),
            "mean_within_k_auroc_cot": float(
                np.mean([v for v in within_cot if v is not None])
            ),
        }
    for kk in ks:
        print(
            f"{kk}   "
            + "  ".join(
                f"{out['ladder'][NAMES.get(p, p)]['gap_fraction_by_k'][kk]:+7.2f}"
                for p in series
            )
            + f"   {cot_k[kk]:5.1f}  {sr_k[kk]:5.1f}"
        )
    for p in series:
        L = out["ladder"][NAMES.get(p, p)]
        print(
            f"{NAMES.get(p, p):>7}: r(k)={L['r_k']:+.2f} r(CoT)={L['r_cot']:+.2f} r(SR)={L['r_sr']:+.2f} | within-k AUROC vs SR {L['mean_within_k_auroc_sr']:.2f}, vs CoT {L['mean_within_k_auroc_cot']:.2f}"
        )

    # ---- 4: Cohen's d on the OFAT set
    ev = np.load(args.eval_acts)
    eids = [str(i) for i in ev["ids"]]
    eresp = {
        r["id"]: r for r in json.loads(Path(args.eval_responses).read_text())["samples"]
    }
    val = np.array([eresp[i]["valence"] for i in eids])
    ecot, esr = verdicts(args.eval_cot), verdicts(args.eval_selfreport)
    eseries = {p: ev[p][:, li].astype(np.float32) @ dirs[p][li] for p in POSITIONS}
    eseries["PAV"] = ev["prompt_last"][:, li].astype(np.float32) @ pav[args.layer]
    print("\nOFAT Cohen's d (aware - not), pooled (safety, capability):")
    out["cohen_d"] = {}
    for p, s in eseries.items():
        row = {}
        for lab_name, lab in (("cot", ecot), ("sr", esr)):
            m = np.array([i in lab for i in eids])
            yy = np.array([lab[i] for i in eids if i in lab])
            d_all = cohen_d(s[m][yy == 1], s[m][yy == 0])
            per = {}
            for v in ("safety", "capability"):
                mv = m & (val == v)
                y2 = np.array([lab[i] for i, mm in zip(eids, mv) if mm])
                per[v] = (
                    cohen_d(s[mv][y2 == 1], s[mv][y2 == 0])
                    if 0 < y2.sum() < len(y2)
                    else None
                )
            row[lab_name] = {"d": d_all, **per}
        out["cohen_d"][NAMES.get(p, p)] = row
        print(
            f"{NAMES.get(p, p):>7}: vs CoT {row['cot']['d']:+.2f} ({row['cot']['safety']:+.2f}, {row['cot']['capability']:+.2f}) | vs SR {row['sr']['d']:+.2f} ({row['sr']['safety']:+.2f}, {row['sr']['capability']:+.2f})"
        )

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
