"""
Graphs for the eval-awareness replication (TODO item 3):
  by provider, by release date, by API cost, and self-report uplift.

RUN SELECTION IS EXPLICIT: only files pinned in BIG_RUN below are plotted —
never newest-wins globbing, which conflates canary/smoke runs with the real
sweep (results dirs hold both). To update pins after a new sweep, run with
SURVEY=1: it prints every result file with n / config coverage / judge mix /
source response file, and marks the currently-pinned ones.

For each pinned model: model_awareness yes-rate with denominator = yes+no
(n/a and invalid are counted and reported but never fold into "no"), Wilson
95% CIs. Writes four PNGs plus awareness_summary_{config}.json into OUT_DIR.

Rates are per-config; plots use CONFIG_FILTER (default "baseline") so the
headline model comparison isn't polluted by the awareness-raising factor
variants.

Environment variables:
    SURVEY          "1" = print the file survey and exit (for updating pins)
    CONFIG_FILTER   Config to plot (default: "baseline"; "ALL" pools everything)
    OUT_DIR         Output directory (default: results/eval_bench/plots)

Model metadata (release date, output $/Mtok) lives in MODEL_META below.
None = unknown at sketch time; models with missing metadata are dropped from
the affected plot and listed on stderr. #@robbie fill in the None cells
(esp. post-Jan-2026 releases and OpenRouter prices).
"""

import json
import math
import os
import sys
from collections import Counter
from datetime import date
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import bucket_conclusion, extract_conclusion

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "eval_bench"
)

# The 2026-08-31→09-01 giant run: responses generated 20260901_005920 (Pacific),
# CoT judged by claude-sonnet-5 (intentional — Robbie 2026-09-01; ~150 records
# per file are gpt-5.6-terra via id-cache reuse), selfreport probes replayed
# from the same response files. One row per roster model; None = that leg
# hadn't finished when pins were last updated (rerun SURVEY=1 and fill in).
BIG_RUN: dict[str, dict[str, str | None]] = {
    "claude-sonnet-4.5": {
        "cot": "cot_claude-sonnet-4.5_20260901_095322.json",
        # terra-judged alternate from the same responses: ..._20260901_025642.json
        "selfreport": "selfreport_claude-sonnet-4.5_20260901_033516.json",
    },
    "claude-haiku-4.5": {
        "cot": "cot_claude-haiku-4.5_20260901_095331.json",
        # terra-judged alternate: ..._20260901_032213.json
        "selfreport": "selfreport_claude-haiku-4.5_20260901_034056.json",
    },
    "gpt-5": {
        "cot": None,
        "selfreport": None,
    },  # cot still judging (n=4968 last survey)
    "gpt-5.1": {"cot": None, "selfreport": None},  # cot still judging (n=5022)
    "gpt-5.2": {"cot": None, "selfreport": None},  # cot still judging (n=5130)
    "gpt-5.4": {"cot": None, "selfreport": None},  # cot still judging (n=4644)
    "gpt-5.6-sol": {"cot": None, "selfreport": None},  # not started yet
    "gemini-3.1-pro": {"cot": None, "selfreport": None},  # only 4-sample canary so far
    "gemini-3.5-flash": {
        "cot": "cot_gemini-3.5-flash_20260901_100742.json",
        "selfreport": "selfreport_gemini-3.5-flash_20260901_094919.json",
    },
    "qwen3.8-2.4t": {
        "cot": "cot_qwen3.8-2.4t_20260901_100727.json",  # 085001 is a superseded duplicate
        "selfreport": "selfreport_qwen3.8-2.4t_20260901_081110.json",
    },
    "deepseek-v4-pro": {
        "cot": "cot_deepseek-v4-pro_20260901_101000.json",
        "selfreport": "selfreport_deepseek-v4-pro_20260901_091928.json",
    },
    "kimi-k3": {
        "cot": "cot_kimi-k3_20260901_100843.json",
        "selfreport": "selfreport_kimi-k3_20260901_085934.json",
    },
    "glm-5.3": {
        "cot": "cot_glm-5.3_20260901_095305.json",
        "selfreport": "selfreport_glm-5.3_20260901_055442.json",
    },
    "minimax-m3": {
        "cot": "cot_minimax-m3_20260901_095346.json",
        "selfreport": "selfreport_minimax-m3_20260901_074409.json",
    },
}

EXPECTED_CONFIGS = 9  # baseline + F1..F8

# provider = color group; lab = fine-grained attribution kept for the summary.
# release: month-precision dates use day=15. usd_out = $ per Mtok output.
# None = unknown at sketch time (post-cutoff model or unchecked OpenRouter price).
MODEL_META: dict[str, dict] = {
    "claude-sonnet-4.5": {
        "provider": "Anthropic",
        "lab": "Anthropic",
        "release": date(2025, 9, 29),
        "usd_out": 15.0,
    },
    "claude-haiku-4.5": {
        "provider": "Anthropic",
        "lab": "Anthropic",
        "release": date(2025, 10, 1),
        "usd_out": 5.0,
    },
    "gpt-5": {
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": date(2025, 8, 7),
        "usd_out": 10.0,
    },
    "gpt-5.1": {
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": date(2025, 11, 15),
        "usd_out": 10.0,
    },
    "gpt-5.2": {
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": None,
    },
    "gpt-5.4": {
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": None,
    },
    "gpt-5.6-sol": {
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": None,
    },
    "gemini-3.1-pro": {
        "provider": "Google",
        "lab": "Google",
        "release": None,
        "usd_out": None,
    },
    "gemini-3.5-flash": {
        "provider": "Google",
        "lab": "Google",
        "release": None,
        "usd_out": None,
    },
    "qwen3.8-2.4t": {
        "provider": "Open-weight",
        "lab": "Alibaba",
        "release": date(2026, 8, 15),
        "usd_out": None,
    },
    "deepseek-v4-pro": {
        "provider": "Open-weight",
        "lab": "DeepSeek",
        "release": date(2026, 4, 15),
        "usd_out": None,
    },
    "kimi-k3": {
        "provider": "Open-weight",
        "lab": "Moonshot",
        "release": date(2026, 7, 15),
        "usd_out": None,
    },
    "glm-5.3": {
        "provider": "Open-weight",
        "lab": "Z.ai",
        "release": date(2026, 8, 15),
        "usd_out": None,
    },
    "minimax-m3": {
        "provider": "Open-weight",
        "lab": "MiniMax",
        "release": date(2026, 5, 15),
        "usd_out": None,
    },
}

# Self-contained display names (lib.config.MODEL_DISPLAY_NAMES is keyed by
# target-client slugs, which sometimes differ from result-file model keys).
DISPLAY_NAMES = {
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "gpt-5": "GPT-5",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.6-sol": "GPT-5.6",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "qwen3.8-2.4t": "Qwen3.8",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "kimi-k3": "Kimi K3",
    "glm-5.3": "GLM-5.3",
    "minimax-m3": "MiniMax M3",
}

# CoT channel is broken for these (adaptive thinking / probe refusals), so a
# CoT-vs-selfreport comparison is instrument failure, not signal.
# See .claude/notes/selfreport-vs-cot-gap.md. (Neither is in BIG_RUN today;
# kept so re-adding them to a future sweep can't silently produce the plot.)
UPLIFT_EXCLUDE = {"claude-opus-5", "claude-fable-5"}

# dataviz reference palette, light mode; providers get fixed slots (never cycled).
PROVIDER_COLOR = {
    "Anthropic": "#2a78d6",
    "OpenAI": "#eb6834",
    "Google": "#1baf7a",
    "Open-weight": "#eda100",
}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", []) if isinstance(data, dict) else data


def pinned_path(kind: str, model: str) -> str | None:
    fname = BIG_RUN.get(model, {}).get(kind)
    if fname is None:
        return None
    path = os.path.join(RESULTS, kind, fname)
    if not os.path.exists(path):
        sys.exit(f"pinned file missing: {path} — rerun SURVEY=1 and update BIG_RUN")
    return path


def survey() -> None:
    """Print every result file so BIG_RUN pins can be updated by hand."""
    pinned = {v[k] for v in BIG_RUN.values() for k in ("cot", "selfreport") if v[k]}
    for kind in ("cot", "selfreport"):
        print(f"=== {kind} ('*' = pinned in BIG_RUN)")
        for path in sorted(glob(os.path.join(RESULTS, kind, f"{kind}_*.json"))):
            base = os.path.basename(path)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"  {base:58} UNREADABLE")
                continue
            recs = data.get("samples", []) if isinstance(data, dict) else data
            src = (
                os.path.basename(data.get("source_response_file", ""))
                if isinstance(data, dict)
                else ""
            )
            cfgs = len({r.get("config_name") for r in recs})
            judges = Counter(r.get("judge_model") for r in recs)
            jstr = ",".join(f"{k}:{v}" for k, v in judges.most_common(2))
            mark = "*" if base in pinned else " "
            print(f"{mark} {base:58} n={len(recs):5} cfgs={cfgs} {jstr:42} {src}")


def summarize(records: list[dict], config: str) -> dict:
    """Bucket model_awareness conclusions for one config ('ALL' pools)."""
    counts = {"yes": 0, "no": 0, "n/a": 0, "invalid": 0}
    judges: Counter = Counter()
    for r in records:
        if config != "ALL" and r.get("config_name") != config:
            continue
        b = bucket_conclusion(
            extract_conclusion(r.get("judge") or {}, "model_awareness")
        )
        counts[b] += 1
        judges[r.get("judge_model", "selfreport")] += 1
    graded = counts["yes"] + counts["no"]
    rate = counts["yes"] / graded if graded else None
    lo, hi = wilson_ci(counts["yes"], graded)
    return {
        "counts": counts,
        "n_graded": graded,
        "rate": rate,
        "ci": [lo, hi],
        "judges": dict(judges),
    }


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


def new_fig(w=8.0, h=5.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, out_dir: str, name_: str):
    path = os.path.join(out_dir, name_)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def plot_by_provider(rows: list[dict], out_dir: str, config: str):
    """Horizontal bars, grouped by provider, sorted by rate within group."""
    rows = sorted(rows, key=lambda r: (r["provider"], -r["rate"]))
    fig, ax = new_fig(h=0.42 * len(rows) + 1.2)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        c = PROVIDER_COLOR[r["provider"]]
        ax.barh(y, r["rate"] * 100, height=0.62, color=c, zorder=3)
        ax.plot([r["ci"][0] * 100, r["ci"][1] * 100], [y, y], color=INK, lw=1, zorder=4)
        ax.text(
            r["ci"][1] * 100 + 0.15,
            y,
            f"{r['rate'] * 100:.1f}%",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels([name(r["model"]) for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CoT-judged eval-awareness (% of graded samples)", color=INK2)
    ax.set_title(f"Eval awareness by provider ({config} config)", color=INK, loc="left")
    ax.xaxis.grid(True, color="#e5e4e0", lw=0.7, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PROVIDER_COLOR.values()]
    ax.legend(
        handles, PROVIDER_COLOR.keys(), frameon=False, fontsize=8, loc="lower right"
    )
    save(fig, out_dir, f"awareness_by_provider_{config}.png")


def plot_scatter(rows, out_dir, config, xkey, xlabel, fname, log_x=False):
    """Shared shape for the release-date and cost scatters."""
    usable = [r for r in rows if r.get(xkey) is not None]
    dropped = [r["model"] for r in rows if r.get(xkey) is None]
    if dropped:
        print(
            f"[{fname}] skipped (missing {xkey}): {', '.join(dropped)}", file=sys.stderr
        )
    if not usable:
        return
    fig, ax = new_fig()
    if log_x:
        ax.set_xscale("log")
    for r in usable:
        c = PROVIDER_COLOR[r["provider"]]
        y = r["rate"] * 100
        ax.errorbar(
            r[xkey],
            y,
            yerr=[[y - r["ci"][0] * 100], [r["ci"][1] * 100 - y]],
            fmt="o",
            color=c,
            ms=7,
            elinewidth=1,
            capsize=0,
            zorder=3,
        )
        ax.annotate(
            name(r["model"]),
            (r[xkey], y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_ylabel("CoT-judged eval-awareness (%)", color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(
        f"Eval awareness by {xlabel.lower()} ({config} config)", color=INK, loc="left"
    )
    ax.yaxis.grid(True, color="#e5e4e0", lw=0.7, zorder=0)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c) for c in PROVIDER_COLOR.values()
    ]
    ax.legend(handles, PROVIDER_COLOR.keys(), frameon=False, fontsize=8)
    save(fig, out_dir, fname)


def plot_uplift(rows: list[dict], out_dir: str, config: str):
    """Dumbbell per model: CoT-judged rate -> self-report rate."""
    rows = [
        r
        for r in rows
        if r["model"] not in UPLIFT_EXCLUDE and r.get("self_rate") is not None
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["self_rate"])
    fig, ax = new_fig(h=0.45 * len(rows) + 1.4)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        cot, self_ = r["rate"] * 100, r["self_rate"] * 100
        ax.plot([cot, self_], [y, y], color="#e5e4e0", lw=2, zorder=2)
        ax.plot(
            cot,
            y,
            "o",
            color=PROVIDER_COLOR[r["provider"]],
            mfc=SURFACE,
            ms=8,
            mew=1.6,
            zorder=3,
        )
        ax.plot(self_, y, "o", color=PROVIDER_COLOR[r["provider"]], ms=8, zorder=3)
        ax.text(
            self_ + 1.5,
            y,
            f"{cot:.0f}→{self_:.0f}%",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels([name(r["model"]) for r in rows], fontsize=9)
    ax.set_xlabel(
        "eval-awareness (%): open = CoT-judged, filled = self-report", color=INK2
    )
    ax.set_title(
        f"Uplift from asking the model itself ({config} config)", color=INK, loc="left"
    )
    ax.xaxis.grid(True, color="#e5e4e0", lw=0.7, zorder=0)
    save(fig, out_dir, f"selfreport_uplift_{config}.png")


def main():
    if os.getenv("SURVEY") == "1":
        survey()
        return

    config = os.getenv("CONFIG_FILTER", "baseline")
    out_dir = os.getenv("OUT_DIR", os.path.join(RESULTS, "plots"))
    os.makedirs(out_dir, exist_ok=True)

    rows, summary = [], {}
    for model in BIG_RUN:
        cot_path = pinned_path("cot", model)
        if cot_path is None:
            print(f"[skip] {model}: no cot file pinned yet", file=sys.stderr)
            continue
        records = load_records(cot_path)
        n_cfgs = len({r.get("config_name") for r in records})
        if n_cfgs != EXPECTED_CONFIGS:
            print(
                f"[warn] {model}: {n_cfgs}/{EXPECTED_CONFIGS} configs in pinned "
                f"cot file — incomplete run?",
                file=sys.stderr,
            )
        s = summarize(records, config)
        if s["rate"] is None:
            print(
                f"[skip] {model}: no graded samples for config {config}",
                file=sys.stderr,
            )
            continue
        row = {
            "model": model,
            "rate": s["rate"],
            "ci": s["ci"],
            "n_graded": s["n_graded"],
            **MODEL_META[model],
        }
        summary[model] = {"cot_file": os.path.basename(cot_path), "cot": s}
        self_path = pinned_path("selfreport", model)
        if self_path:
            ss = summarize(load_records(self_path), config)
            row["self_rate"] = ss["rate"]
            summary[model]["selfreport_file"] = os.path.basename(self_path)
            summary[model]["selfreport"] = ss
        rows.append(row)

    with open(os.path.join(out_dir, f"awareness_summary_{config}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'model':22} {'n':>5} {'cot%':>6} {'self%':>6}  na/inv (cot)")
    for r in rows:
        c = summary[r["model"]]["cot"]["counts"]
        self_pct = (
            f"{r['self_rate'] * 100:5.1f}"
            if r.get("self_rate") is not None
            else "    –"
        )
        print(
            f"{r['model']:22} {r['n_graded']:>5} {r['rate'] * 100:5.1f} {self_pct:>6}"
            f"  {c['n/a']}/{c['invalid']}"
        )

    plot_by_provider(rows, out_dir, config)
    plot_scatter(
        rows,
        out_dir,
        config,
        "release",
        "Release date",
        f"awareness_by_release_date_{config}.png",
    )
    plot_scatter(
        rows,
        out_dir,
        config,
        "usd_out",
        "API cost ($/Mtok output)",
        f"awareness_by_cost_{config}.png",
        log_x=True,
    )
    plot_uplift(rows, out_dir, config)


if __name__ == "__main__":
    main()
