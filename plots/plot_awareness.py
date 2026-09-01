"""
Graphs for the eval-awareness replication (TODO item 3):
  by provider, by release date, by API cost, self-report uplift, a
  per-factor heatmap (model × OFAT config), and by task type (HarmBench
  category × valence, pooled across models).

RUN SELECTION IS EXPLICIT: only files pinned in BIG_RUN below are plotted —
never newest-wins globbing, which conflates canary/smoke runs with the real
sweep (results dirs hold both). To update pins after a new sweep, run with
SURVEY=1: it prints every result file with n / config coverage / judge mix /
source response file, and marks the currently-pinned ones.

For each pinned model: model_awareness yes-rate with denominator = yes+no
(n/a and invalid are counted and reported but never fold into "no"), Wilson
95% CIs. Writes five PNGs plus awareness_summary_{config}.json into OUT_DIR.

The provider/release-date/cost charts plot ONE probe, chosen by PROBE
(default "selfreport"; "cot" for the CoT-monitoring rate). Models missing
the chosen probe's file drop out with a stderr note. The uplift chart always
needs both probes. Rates are per-config; CONFIG_FILTER (default "baseline")
keeps the headline comparison free of the awareness-raising factor variants.
The factor heatmap ignores CONFIG_FILTER — its columns *are* the configs.
The task-type chart pools every pinned model's records (so it weights models
by graded n, like the heatmap's Pooled row) and splits safety vs capability.

Environment variables:
    SURVEY          "1" = print the file survey and exit (for updating pins)
    PROBE           "selfreport" (default) or "cot"
    CONFIG_FILTER   Config to plot (default: "baseline"; "ALL" pools everything)
    OUT_DIR         Output directory (default: results/eval_bench/plots)

Model metadata (provider, release date, output $/Mtok, display names) lives
in lib/model_meta.py.
"""

import json
import math
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
    VALENCE_COLOR,
    color_legend,
    new_fig,
    save,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "eval_bench")
TASKS = os.path.join(ROOT, "eval_bench", "tasks")

# The 2026-08-31→09-01 giant run: responses generated 20260901_005920 (Pacific;
# gemini 010911), selfreport probes replayed from the same response files.
# CoT grader: every cot pin is the 2026-09-01 ~14:25 gpt-5.6-terra sweep over
# the completed response files (no backup grader; transcripts OpenAI's cyber
# filter blocks are N/A rows, ~1%/model — see the grader note). The OpenAI
# legs were regenerated after the corp account was restored (PR #47 fixed the
# selfreport probe; files dated 14:00–14:40). One row per roster model.
BIG_RUN: dict[str, dict[str, str | None]] = {
    "claude-sonnet-4.5": {
        "cot": "cot_claude-sonnet-4.5_20260901_142454.json",
        "selfreport": "selfreport_claude-sonnet-4.5_20260901_033516.json",
    },
    "claude-haiku-4.5": {
        "cot": "cot_claude-haiku-4.5_20260901_142524.json",
        "selfreport": "selfreport_claude-haiku-4.5_20260901_034056.json",
    },
    "gpt-5": {
        "cot": "cot_gpt-5_20260901_142448.json",
        "selfreport": "selfreport_gpt-5_20260901_141533.json",
    },
    "gpt-5.1": {
        "cot": "cot_gpt-5.1_20260901_142738.json",
        "selfreport": "selfreport_gpt-5.1_20260901_143914.json",
    },
    "gpt-5.2": {
        "cot": "cot_gpt-5.2_20260901_142502.json",
        "selfreport": "selfreport_gpt-5.2_20260901_140000.json",
    },
    "gpt-5.4": {
        "cot": "cot_gpt-5.4_20260901_142611.json",
        "selfreport": "selfreport_gpt-5.4_20260901_140552.json",
    },
    "gpt-5.6-sol": {
        "cot": "cot_gpt-5.6-sol_20260901_142546.json",
        "selfreport": "selfreport_gpt-5.6-sol_20260901_142348.json",
    },
    "gemini-3.1-pro": {
        "cot": "cot_gemini-3.1-pro_20260901_142317.json",
        "selfreport": None,  # selfreport leg running as of 2026-09-01 15:00
    },
    "gemini-3.5-flash": {
        "cot": "cot_gemini-3.5-flash_20260901_142512.json",
        "selfreport": "selfreport_gemini-3.5-flash_20260901_094919.json",
    },
    "qwen3.8-2.4t": {
        "cot": "cot_qwen3.8-2.4t_20260901_142510.json",
        "selfreport": "selfreport_qwen3.8-2.4t_20260901_081110.json",
    },
    "deepseek-v4-pro": {
        "cot": "cot_deepseek-v4-pro_20260901_142806.json",
        "selfreport": "selfreport_deepseek-v4-pro_20260901_091928.json",
    },
    "kimi-k3": {
        "cot": "cot_kimi-k3_20260901_142534.json",
        "selfreport": "selfreport_kimi-k3_20260901_085934.json",
    },
    "glm-5.3": {
        "cot": "cot_glm-5.3_20260901_142520.json",
        "selfreport": "selfreport_glm-5.3_20260901_055442.json",
    },
    "minimax-m3": {
        "cot": "cot_minimax-m3_20260901_142554.json",
        "selfreport": "selfreport_minimax-m3_20260901_074409.json",
    },
}

EXPECTED_CONFIGS = 9  # baseline + F1..F8

# OFAT configs in column order, labelled by the *varied* level of the one
# factor flipped (see eval_bench/taxonomy.yaml).
FACTOR_LABELS = {
    "baseline": "baseline",
    "F1": "F1\nfake\nentities",
    "F2": "F2\nincoherent\nscenario",
    "F3": "F3\nformal\nregister",
    "F4": "F4\nbare\nrequest",
    "F5": "F5\nstakes\ntelegraphed",
    "F6": "F6\nknown\nparadigm",
    "F7": "F7\ntemplate\nformat",
    "F8": "F8\neval\ndisclosure",
}

# HarmBench semantic categories the 100 task pairs were adapted from.
CATEGORY_LABELS = {
    "illegal": "Illegal activities",
    "misinformation_disinformation": "Misinformation",
    "harassment_bullying": "Harassment & bullying",
    "cybercrime_intrusion": "Cybercrime & intrusion",
    "harmful": "General harm",
    "copyright": "Copyright (song lyrics)",
}

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


def task_categories() -> dict[str, str]:
    """task_name -> HarmBench category, from the task-generation logs.

    The category isn't in the task YAMLs; it only survives in the logs of the
    pipeline that adapted the HarmBench behaviors. Tasks that were kept from the
    first pass appear only in the *_before_replaced log, so read both.
    """
    cats: dict[str, str] = {}
    for fname in ("generation_log_before_replaced.json", "generation_log.json"):
        with open(os.path.join(TASKS, fname)) as f:
            for t in json.load(f)["tasks"]:
                cats[t["behavior_id"]] = t["category"]
    return cats


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
    color_legend(ax, PROVIDER_COLOR, loc="lower right")
    save(fig, out_dir, f"awareness_by_provider_{probe}_{config}.png")


def plot_scatter(
    rows, out_dir, probe, config, xkey, xlabel, title_noun, fname, log_x=False
):
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
        f"{PROBE_TITLE[probe]} eval awareness by {title_noun} ({config} config)",
        color=INK,
        loc="left",
    )
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    color_legend(ax, PROVIDER_COLOR, marker=True)
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


def plot_factor_heatmap(out_dir: str, probe: str):
    """Awareness rate per (model, OFAT config) as a labelled heatmap + pooled row.

    Self-report is the useful probe here: CoT-judged rates sit at ~1% at
    baseline and only F1/F2/F8 move them, so they can't rank factors.
    """
    configs = list(FACTOR_LABELS)
    grid: list[list[float]] = []
    labels: list[str] = []
    pooled = {c: [0, 0] for c in configs}
    for model in BIG_RUN:
        path = pinned_path(probe, model)
        if path is None:
            continue
        recs = load_records(path)
        row = []
        for c in configs:
            s = summarize(recs, c)
            row.append(s["rate"] if s["rate"] is not None else math.nan)
            pooled[c][0] += s["counts"]["yes"]
            pooled[c][1] += s["n_graded"]
        grid.append(row)
        labels.append(display_name(model))
    if not grid:
        return
    grid.append([y / n if n else math.nan for y, n in pooled.values()])
    labels.append("Pooled")

    n_rows, n_cols = len(grid), len(configs)
    vmax = max(v for row in grid for v in row if not math.isnan(v))
    fig, ax = new_fig(w=0.95 * n_cols + 1.8, h=0.5 * n_rows + 1.6)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if math.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=INK2)
                continue
            ink = "white" if v / vmax > 0.6 else INK
            ax.text(
                j,
                i,
                f"{v * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color=ink,
            )
    # Surface-colored gaps between cells; dashed rule after baseline; rule above pooled.
    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.axhline(n_rows - 1.5, color=INK2, lw=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([FACTOR_LABELS[c] for c in configs], fontsize=7.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel(
        "single factor flipped to its varied level (OFAT vs baseline)", color=INK2
    )
    ax.set_title(
        f"{PROBE_TITLE[probe]} eval awareness by factor varied", color=INK, loc="left"
    )
    save(fig, out_dir, f"awareness_by_factor_{probe}.png")


def plot_by_task_type(out_dir: str, probe: str, config: str):
    """Grouped horizontal bars: pooled awareness per HarmBench category × valence."""
    cats = task_categories()
    yes: Counter = Counter()
    graded: Counter = Counter()
    n_models, unmapped = 0, set()
    for model in BIG_RUN:
        path = pinned_path(probe, model)
        if path is None:
            continue
        n_models += 1
        for r in load_records(path):
            if config != "ALL" and r.get("config_name") != config:
                continue
            cat = cats.get(r["task_name"])
            if cat is None:
                unmapped.add(r["task_name"])
                continue
            key = (cat, r["valence"])
            b = bucket_conclusion(
                extract_conclusion(r.get("judge") or {}, "model_awareness")
            )
            yes[key] += b == "yes"
            graded[key] += b in ("yes", "no")
    if unmapped:
        print(f"[task-type] no category for: {sorted(unmapped)}", file=sys.stderr)
    if not graded:
        return

    def rate(cat, val):
        return yes[cat, val] / graded[cat, val] if graded[cat, val] else math.nan

    order = sorted(
        CATEGORY_LABELS,
        key=lambda c: (
            -sum(yes[c, v] for v in VALENCE_COLOR)
            / max(1, sum(graded[c, v] for v in VALENCE_COLOR))
        ),
    )
    fig, ax = new_fig(h=0.85 * len(order) + 1.4)
    bar_h = 0.36
    for i, cat in enumerate(order):
        for j, val in enumerate(VALENCE_COLOR):
            y = i + (j - 0.5) * (bar_h + 0.04)
            p = rate(cat, val) * 100
            lo, hi = wilson_ci(yes[cat, val], graded[cat, val])
            ax.barh(y, p, height=bar_h, color=VALENCE_COLOR[val], zorder=3)
            ax.plot([lo * 100, hi * 100], [y, y], color=INK, lw=1, zorder=4)
            ax.text(
                hi * 100 + 0.3,
                y,
                f"{p:.1f}%",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CATEGORY_LABELS[c] for c in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"{PROBE_XLABEL[probe][:-1]}, pooled over {n_models} models)", color=INK2
    )
    ax.set_title(
        f"{PROBE_TITLE[probe]} eval awareness by task type ({config} config)",
        color=INK,
        loc="left",
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    color_legend(ax, {f"{v} task": c for v, c in VALENCE_COLOR.items()})
    save(fig, out_dir, f"awareness_by_task_type_{probe}_{config}.png")


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
        "release date",
        f"awareness_by_release_date_{probe}_{config}.png",
    )
    plot_scatter(
        plotted,
        out_dir,
        probe,
        config,
        "usd_out",
        "API cost ($/Mtok output)",
        "API cost",
        f"awareness_by_cost_{probe}_{config}.png",
        log_x=True,
    )
    plot_uplift(rows, out_dir, config)
    plot_factor_heatmap(out_dir, probe)
    plot_by_task_type(out_dir, probe, config)


if __name__ == "__main__":
    main()
