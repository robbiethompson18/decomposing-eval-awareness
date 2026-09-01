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

The provider/release-date/cost charts plot ONE probe, chosen by PROBE
(default "selfreport"; "cot" for the CoT-monitoring rate). Models missing
the chosen probe's file drop out with a stderr note. The uplift chart always
needs both probes. Rates are per-config; CONFIG_FILTER (default "baseline")
keeps the headline comparison free of the awareness-raising factor variants.

Environment variables:
    SURVEY          "1" = print the file survey and exit (for updating pins)
    PROBE           "selfreport" (default) or "cot"
    CONFIG_FILTER   Config to plot (default: "baseline"; "ALL" pools everything)
    OUT_DIR         Output directory (default: results/eval_bench/plots)

Model metadata (provider, release date, output $/Mtok, display names) lives
in lib/model_meta.py.
"""

import json
import os
import sys
from collections import Counter
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import bucket_conclusion, extract_conclusion
from lib.model_meta import MODEL_META, display_name
from lib.stats import wilson_ci
from plots.style import (
    GRID,
    INK,
    INK2,
    PROVIDER_COLOR,
    SURFACE,
    new_fig,
    provider_legend,
    save,
)

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
    # gpt cot files are final but short of 5400 (failed samples); no gpt
    # selfreport leg has run yet as of 2026-09-01 11:15am.
    "gpt-5": {"cot": "cot_gpt-5_20260901_102007.json", "selfreport": None},
    "gpt-5.1": {"cot": "cot_gpt-5.1_20260901_101924.json", "selfreport": None},
    "gpt-5.2": {"cot": "cot_gpt-5.2_20260901_102115.json", "selfreport": None},
    "gpt-5.4": {"cot": "cot_gpt-5.4_20260901_102144.json", "selfreport": None},
    "gpt-5.6-sol": {"cot": "cot_gpt-5.6-sol_20260901_102920.json", "selfreport": None},
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

PROBE_TITLE = {"selfreport": "Self-reported", "cot": "CoT-judged"}
PROBE_XLABEL = {
    "selfreport": "self-reported eval-awareness (% of graded samples)",
    "cot": "CoT-judged eval-awareness (% of graded samples)",
}

# CoT channel is broken for these (adaptive thinking / probe refusals), so a
# CoT-vs-selfreport comparison is instrument failure, not signal.
# See .claude/notes/selfreport-vs-cot-gap.md. (Neither is in BIG_RUN today;
# kept so re-adding them to a future sweep can't silently produce the plot.)
UPLIFT_EXCLUDE = {"claude-opus-5", "claude-fable-5"}


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


def plot_by_provider(rows: list[dict], out_dir: str, probe: str, config: str):
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
    ax.set_yticklabels([display_name(r["model"]) for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(PROBE_XLABEL[probe], color=INK2)
    ax.set_title(
        f"{PROBE_TITLE[probe]} eval awareness by provider ({config} config)",
        color=INK,
        loc="left",
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    provider_legend(ax, loc="lower right")
    save(fig, out_dir, f"awareness_by_provider_{probe}_{config}.png")


def plot_scatter(rows, out_dir, probe, config, xkey, xlabel, fname, log_x=False):
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
            display_name(r["model"]),
            (r[xkey], y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_ylabel(PROBE_XLABEL[probe], color=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_title(
        f"{PROBE_TITLE[probe]} eval awareness by {xlabel.lower()} ({config} config)",
        color=INK,
        loc="left",
    )
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    provider_legend(ax, marker=True)
    save(fig, out_dir, fname)


def plot_uplift(rows: list[dict], out_dir: str, config: str):
    """Dumbbell per model: CoT-judged rate -> self-report rate (probe-independent)."""
    rows = [
        r
        for r in rows
        if r["model"] not in UPLIFT_EXCLUDE
        and r.get("cot_rate") is not None
        and r.get("self_rate") is not None
    ]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["self_rate"])
    fig, ax = new_fig(h=0.45 * len(rows) + 1.4)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        cot, self_ = r["cot_rate"] * 100, r["self_rate"] * 100
        ax.plot([cot, self_], [y, y], color=GRID, lw=2, zorder=2)
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
    ax.set_yticklabels([display_name(r["model"]) for r in rows], fontsize=9)
    ax.set_xlabel(
        "eval-awareness (%): open = CoT-judged, filled = self-report", color=INK2
    )
    ax.set_title(
        f"Uplift from asking the model itself ({config} config)", color=INK, loc="left"
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    save(fig, out_dir, f"selfreport_uplift_{config}.png")


def main():
    if os.getenv("SURVEY") == "1":
        survey()
        return

    probe = os.getenv("PROBE", "selfreport")
    if probe not in PROBE_TITLE:
        sys.exit(f"PROBE must be one of {sorted(PROBE_TITLE)}, got {probe!r}")
    config = os.getenv("CONFIG_FILTER", "baseline")
    out_dir = os.getenv("OUT_DIR", os.path.join(RESULTS, "plots"))
    os.makedirs(out_dir, exist_ok=True)

    rows, summary = [], {}
    for model in BIG_RUN:
        stats = {}
        for kind in ("cot", "selfreport"):
            path = pinned_path(kind, model)
            if path is None:
                continue
            records = load_records(path)
            if kind == "cot":
                n_cfgs = len({r.get("config_name") for r in records})
                if n_cfgs != EXPECTED_CONFIGS:
                    print(
                        f"[warn] {model}: {n_cfgs}/{EXPECTED_CONFIGS} configs in "
                        f"pinned cot file — incomplete run?",
                        file=sys.stderr,
                    )
            s = summarize(records, config)
            stats[kind] = s
            summary.setdefault(model, {})[f"{kind}_file"] = os.path.basename(path)
            summary[model][kind] = s
        if not stats:
            print(f"[skip] {model}: no files pinned yet", file=sys.stderr)
            continue
        primary = stats.get(probe)
        if primary is None or primary["rate"] is None:
            print(
                f"[warn] {model}: no {probe} data for config {config} — "
                f"dropped from single-probe charts",
                file=sys.stderr,
            )
        row = {
            "model": model,
            "rate": primary["rate"] if primary else None,
            "ci": primary["ci"] if primary else None,
            "n_graded": primary["n_graded"] if primary else 0,
            "cot_rate": stats.get("cot", {}).get("rate"),
            "self_rate": stats.get("selfreport", {}).get("rate"),
            **MODEL_META[model],
        }
        rows.append(row)

    with open(os.path.join(out_dir, f"awareness_summary_{config}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'model':22} {'cot%':>6} {'self%':>6}")
    for r in rows:
        cot_pct = (
            f"{r['cot_rate'] * 100:5.1f}" if r["cot_rate"] is not None else "    –"
        )
        self_pct = (
            f"{r['self_rate'] * 100:5.1f}" if r["self_rate"] is not None else "    –"
        )
        print(f"{r['model']:22} {cot_pct:>6} {self_pct:>6}")

    plotted = [r for r in rows if r["rate"] is not None]
    plot_by_provider(plotted, out_dir, probe, config)
    plot_scatter(
        plotted,
        out_dir,
        probe,
        config,
        "release",
        "Release date",
        f"awareness_by_release_date_{probe}_{config}.png",
    )
    plot_scatter(
        plotted,
        out_dir,
        probe,
        config,
        "usd_out",
        "API cost ($/Mtok output)",
        f"awareness_by_cost_{probe}_{config}.png",
        log_x=True,
    )
    plot_uplift(rows, out_dir, config)


if __name__ == "__main__":
    main()
