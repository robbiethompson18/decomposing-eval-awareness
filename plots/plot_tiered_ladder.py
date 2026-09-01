"""
Monitoring-information ladder plots: for each metric, how the detected rate
climbs across what a monitor can see —

    output-only  ->  output + CoT  ->  self-report (ask the model)

One slope-line per model (colored by provider), for three metrics:
  awareness     model_awareness      (does the model know it's being evaluated)
  realistic     hypothetical_recognition (does it recognize a constructed scenario)
  is_guarded    task_performance == "guarded"  (deliberate softening; NO self-report
                stage — task compliance isn't a self-report probe, so 2 points only)

output / output+CoT come from the tiered grader (results/eval_bench/cot_tiered/);
the self-report stage from results/eval_bench/selfreport/. Files are pinned in
PINS (newest per model at write time) — not newest-wins globbing — so a later
canary run can't silently swap in. Rates are yes/(yes+no); n/a and invalid are
excluded from the denominator (never folded into "no"), pooled across all configs
(the three stages share the same transcript set, so the within-transcript climb
is unaffected by the config mix).

Writes three PNGs to OUT_DIR (default results/eval_bench/plots).
"""

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_meta import MODEL_META
from plots.style import PROVIDER_COLOR, color_legend, new_fig, save

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results", "eval_bench")
OUT_DIR = os.getenv("OUT_DIR", os.path.join(RES, "plots"))

# 13 graded models (gemini-3.1-pro still generating, excluded).
MODELS = [
    "claude-sonnet-4.5", "claude-haiku-4.5", "gemini-3.5-flash",
    "qwen3.8-2.4t", "deepseek-v4-pro", "kimi-k3", "glm-5.3", "minimax-m3",
    "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.6-sol",
]


def newest(pattern: str) -> str | None:
    hits = [p for p in sorted(glob.glob(pattern)) if "_scratchpad_" not in p]
    return hits[-1] if hits else None


def pin(model: str) -> dict[str, str | None]:
    return {
        "tiered": newest(os.path.join(RES, "cot_tiered", f"tiered_{model}_2026*.json")),
        "selfreport": newest(
            os.path.join(RES, "selfreport", f"selfreport_{model}_2026*.json")
        ),
    }


def _yes_rate(concls: list[str]) -> float | None:
    c = Counter(concls)
    n = c["yes"] + c["no"]
    return c["yes"] / n * 100 if n else None


def _concl(judge: dict, metric: str) -> str:
    return str(((judge or {}).get(metric) or {}).get("conclusion", "?")).lower()


def tiered_rate(records: list[dict], tier: str, metric: str) -> float | None:
    if metric == "task_performance":  # is_guarded: guarded / all deliberate outcomes
        buckets = ("correct", "guarded", "incorrect", "refused")
        vals = [_concl((r.get("judgments") or {}).get(tier), metric) for r in records]
        n = sum(v in buckets for v in vals)
        return sum(v == "guarded" for v in vals) / n * 100 if n else None
    return _yes_rate(
        [_concl((r.get("judgments") or {}).get(tier), metric) for r in records]
    )


def selfreport_rate(records: list[dict], metric: str) -> float | None:
    return _yes_rate([_concl(r.get("judge"), metric) for r in records])


# (chart name, metric, has self-report stage)
CHARTS = [
    ("awareness", "model_awareness", True),
    ("realistic", "hypothetical_recognition", True),
    ("is_guarded", "task_performance", False),
]
TITLES = {
    "awareness": "Evaluation awareness — what the monitor catches",
    "realistic": "Recognizes constructed scenario — what the monitor catches",
    "is_guarded": "Deliberately guarded output — what the monitor catches",
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pins = {m: pin(m) for m in MODELS}
    data = {}  # model -> {"tiered": records, "selfreport": records}
    for m, p in pins.items():
        if not p["tiered"]:
            print(f"  skip {m}: no tiered file", file=sys.stderr)
            continue
        with open(p["tiered"]) as f:
            d = {"tiered": json.load(f)["samples"]}
        if p["selfreport"]:
            with open(p["selfreport"]) as f:
                d["selfreport"] = json.load(f).get("samples", [])
        data[m] = d

    for name, metric, has_sr in CHARTS:
        stages = ["output-only", "output + CoT"] + (["self-report"] if has_sr else [])
        xs = list(range(len(stages)))
        fig, ax = new_fig(w=8.5, h=6.0)
        used_providers = {}
        label_pts = []  # (y, model) for right-end labels
        for m, d in data.items():
            prov = MODEL_META.get(m, {}).get("provider", "Open-weight")
            color = PROVIDER_COLOR.get(prov, "#888")
            used_providers[prov] = color
            ys = [
                tiered_rate(d["tiered"], "output", metric),
                tiered_rate(d["tiered"], "output_cot", metric),
            ]
            if has_sr:
                ys.append(
                    selfreport_rate(d.get("selfreport", []), metric)
                    if d.get("selfreport")
                    else None
                )
            # drop trailing None (e.g. OpenAI missing a self-report file)
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not valid:
                continue
            vx, vy = zip(*valid)
            ax.plot(vx, vy, "-o", color=color, lw=2, ms=7, alpha=0.9,
                    markeredgecolor="white", markeredgewidth=0.8)
            label_pts.append((vx[-1], vy[-1], m, color))

        # direct-label each line at its right end, dodged to avoid overlap
        label_pts.sort(key=lambda t: t[1])
        last_y = -1e9
        for x_end, y_end, m, color in label_pts:
            y_lab = max(y_end, last_y + 2.0)
            last_y = y_lab
            ax.annotate(MODEL_META.get(m, {}).get("display", m),
                        (x_end, y_end), (x_end + 0.06, y_lab),
                        fontsize=7.5, color=color, va="center",
                        annotation_clip=False)

        ax.set_xticks(xs)
        ax.set_xticklabels(stages, fontsize=10)
        ax.set_xlim(-0.15, xs[-1] + 1.15)
        ax.set_ylim(bottom=-1)
        ax.set_ylabel("detected rate  (yes / yes+no, %)", fontsize=10)
        ax.set_title(TITLES[name], fontsize=12, loc="left")
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
        color_legend(ax, used_providers, marker=True, loc="upper left")
        save(fig, OUT_DIR, f"ladder_{name}.png")


if __name__ == "__main__":
    main()
