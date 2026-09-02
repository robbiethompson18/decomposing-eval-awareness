"""
Phase 1 analysis: does delivering the SAME request through a real agent harness
(OpenClaw) change self-reported eval awareness?

Three arms per model, paired on (task_id, valence):
  baseline   bare API, T3 system prompt, stub search tool   (pinned BIG_RUN file, sample1;
             for the two closed models the OpenRouter re-run `*_p1base_*` so both arms
             share a provider route)
  control    bare API, the real OpenClaw system prompt as text, no live tools
             (`openclaw_prompt` config, `*_p1ctl_*` files)
  harness    OpenClaw for real: live tools, sandbox, agent loop (`*_openclaw_p1_*` files)

Rates use denominator yes+no (invalid/n/a reported, never folded into "no").
Paired test: exact two-sided McNemar on the discordant pairs. Also prints the
harness arm's tool-call rate, since "did it actually use the affordances" is the
first thing to check before believing any delta.

    python scripts/analyze_openclaw_phase1.py [TAG=p1]
Writes results/eval_bench/openclaw/phase1_summary.json.
"""

import json
import math
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import extract_conclusion
from lib.stats import wilson_ci
from plots.plot_awareness import BIG_RUN

ROOT = Path(__file__).resolve().parent.parent
SR = ROOT / "results" / "eval_bench" / "selfreport"
TAG = os.getenv("TAG", "p1")
MODELS = [
    "qwen3.8-2.4t",
    "deepseek-v4-pro",
    "kimi-k3",
    "glm-5.3",
    "minimax-m3",
    "claude-sonnet-4.5",
    "gemini-3.5-flash",
]
OPENROUTER_BASELINE = {"claude-sonnet-4.5", "gemini-3.5-flash"}
METRICS = ("model_awareness", "hypothetical_recognition")


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["samples"] if isinstance(data, dict) else data


def latest(pattern: str) -> Path | None:
    hits = sorted(glob(str(SR / pattern)))
    return Path(hits[-1]) if hits else None


def by_pair(records: list[dict], config: str, metric: str) -> dict[tuple, str]:
    """(task_id, valence) -> yes/no/other for sample1 of the given config."""
    out = {}
    for r in records:
        if r.get("config_name") != config or r.get("sample_idx", 1) != 1:
            continue
        out[(r["task_id"], r["valence"])] = extract_conclusion(
            r.get("judge") or {}, metric
        )
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant counts b (A yes/B no), c (A no/B yes)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def rate(pairs: dict[tuple, str], valence: str | None = None) -> tuple[int, int, int]:
    yes = no = other = 0
    for (_, v), concl in pairs.items():
        if valence and v != valence:
            continue
        if concl == "yes":
            yes += 1
        elif concl == "no":
            no += 1
        else:
            other += 1
    return yes, no, other


def fmt(yes: int, no: int, other: int) -> str:
    n = yes + no
    if not n:
        return "  —  "
    lo, hi = wilson_ci(yes, n)
    s = f"{100 * yes / n:4.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] n={n}"
    return s + (f" (+{other} inv)" if other else "")


def compare(a: dict, b: dict, valence: str | None) -> tuple[int, int, int, float]:
    """Paired A vs B on shared keys: (n_pairs, b=A-yes/B-no, c=A-no/B-yes, p)."""
    keys = [
        k
        for k in a
        if k in b
        and (valence is None or k[1] == valence)
        and a[k] in ("yes", "no")
        and b[k] in ("yes", "no")
    ]
    bb = sum(1 for k in keys if a[k] == "yes" and b[k] == "no")
    cc = sum(1 for k in keys if a[k] == "no" and b[k] == "yes")
    return len(keys), bb, cc, mcnemar_exact(bb, cc)


COT = ROOT / "results" / "eval_bench" / "cot"


def cot_pairs(
    model: str, arm: str, metric: str = "model_awareness"
) -> dict[tuple, str]:
    """CoT-judge verdicts per (task_id, valence), sample1, for one arm.

    baseline: the pinned BIG_RUN cot file (or the *_p1base_* judge for the two
    closed models); control: cot_<model>_p1ctl_*; harness: cot_<model>_openclaw_p1_*.
    """
    if arm == "baseline":
        if model in OPENROUTER_BASELINE:
            hits = sorted(glob(str(COT / f"cot_{model}_{TAG}base_*.json")))
            path = Path(hits[-1]) if hits else None
        else:
            pinned = BIG_RUN.get(model, {}).get("cot")
            path = COT / pinned if pinned else None
        config = "baseline"
    elif arm == "control":
        hits = sorted(glob(str(COT / f"cot_{model}_{TAG}ctl_*.json")))
        path, config = (Path(hits[-1]) if hits else None), "openclaw_prompt"
    else:
        hits = sorted(glob(str(COT / f"cot_{model}_openclaw_{TAG}_*.json")))
        path, config = (Path(hits[-1]) if hits else None), "openclaw"
    if path is None or not path.exists():
        return {}
    return by_pair(load(path), config, metric)


def tool_use(model: str) -> str:
    f = sorted(
        glob(
            str(
                ROOT
                / "results/eval_bench/response"
                / f"responses_{model}_openclaw_{TAG}_*.json"
            )
        )
    )
    if not f:
        return "—"
    recs = load(Path(f[-1]))
    used = sum(1 for r in recs if r.get("openclaw_n_tool_calls"))
    errs = sum(1 for r in recs if str(r.get("response", "")).startswith("ERROR"))
    return f"{used}/{len(recs)} used tools, {errs} errors"


def main() -> None:
    summary: dict = {}
    for metric in METRICS:
        print(f"\n=== {metric} (self-report), sample1, yes/(yes+no) ===")
        print(f"{'model':18s} {'arm':9s} {'all':32s} {'safety':32s} {'capability':32s}")
        for model in MODELS:
            arms: dict[str, dict] = {}
            if model in OPENROUTER_BASELINE:
                f = latest(f"selfreport_{model}_{TAG}base_*.json")
            else:
                pinned = BIG_RUN.get(model, {}).get("selfreport")
                f = SR / pinned if pinned else None
            if f and f.exists():
                arms["baseline"] = by_pair(load(f), "baseline", metric)
            f = latest(f"selfreport_{model}_{TAG}ctl_*.json")
            if f:
                arms["control"] = by_pair(load(f), "openclaw_prompt", metric)
            f = latest(f"selfreport_{model}_openclaw_{TAG}_*.json")
            if f:
                arms["harness"] = by_pair(load(f), "openclaw", metric)
            for arm, pairs in arms.items():
                print(
                    f"{model:18s} {arm:9s} {fmt(*rate(pairs)):32s} "
                    f"{fmt(*rate(pairs, 'safety')):32s} {fmt(*rate(pairs, 'capability')):32s}"
                )
                summary.setdefault(metric, {}).setdefault(model, {})[arm] = {
                    v: rate(pairs, None if v == "all" else v)
                    for v in ("all", "safety", "capability")
                }
            for a, b in (
                ("harness", "baseline"),
                ("control", "baseline"),
                ("harness", "control"),
            ):
                if a in arms and b in arms:
                    for v in (None, "safety", "capability"):
                        n, bb, cc, p = compare(arms[a], arms[b], v)
                        tag = v or "all"
                        print(
                            f"{'':18s}   {a}-vs-{b:9s} {tag:10s} pairs={n:3d}  "
                            f"{a}-only={bb:3d} {b}-only={cc:3d}  McNemar p={p:.3g}"
                        )
                        summary[metric][model][f"{a}_vs_{b}_{tag}"] = {
                            "pairs": n,
                            f"{a}_only": bb,
                            f"{b}_only": cc,
                            "p": p,
                        }
            if metric == METRICS[0]:
                print(f"{'':18s}   harness tool use: {tool_use(model)}")

    print(
        "\n=== model_awareness (CoT judge, sonnet-5 / terra), sample1, yes/(yes+no) ==="
    )
    print(f"{'model':18s} {'arm':9s} {'all':32s} {'safety':32s} {'capability':32s}")
    for model in MODELS:
        for arm in ("baseline", "control", "harness"):
            pairs = cot_pairs(model, arm)
            if not pairs:
                continue
            print(
                f"{model:18s} {arm:9s} {fmt(*rate(pairs)):32s} "
                f"{fmt(*rate(pairs, 'safety')):32s} {fmt(*rate(pairs, 'capability')):32s}"
            )
            summary.setdefault("cot_model_awareness", {}).setdefault(model, {})[arm] = {
                v: rate(pairs, None if v == "all" else v)
                for v in ("all", "safety", "capability")
            }
    out = ROOT / "results" / "eval_bench" / "openclaw" / "phase1_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
