"""
Factor-stacking heatmap: replication of the paper's Figure 7 (arXiv:2605.23055).

Two panels (safety tasks, capability tasks) of model × number-of-factors-varied
(0..8), each cell the awareness rate over every graded sample at that k. Input
is CoT-judged files produced by eval/eval_eval_bench_cot.py from STACKING=1
response files (run/run_eval_bench.py).

Matched-type by default, like the paper: a sample counts as aware only when the
judge's awareness_type equals the task's valence (safety awareness on a safety
task). MATCHED=0 counts any awareness type. Denominator = yes + no verdicts
(n/a and invalid are excluded, never folded into "no").

RUN SELECTION IS EXPLICIT (same rule as plot_awareness.py): only files pinned in
STACKING_RUN are plotted, unless STACKING_FILES lists cot files directly (the
model key is read from the filename, cot_<model>_...). The k of each record is
its stored n_varied, or, for files written before that field existed, the count
of F1–F8 at their varied level in factor_config.

Environment variables:
    STACKING_FILES  Comma-separated cot JSON paths (overrides STACKING_RUN pins)
    MATCHED         "1" (default) matched-type awareness; "0" any type
    OUT_DIR         Output directory (default: results/eval_bench/plots)
"""

import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt

from core.metrics import bucket_conclusion, extract_conclusion
from lib.model_meta import display_name
from plots.style import INK, INK2, SURFACE, style_axes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "eval_bench")

# Pins: model_key -> cot file (basename under results/eval_bench/cot) from a
# STACKING=1 run. Row order here is the row order of the figure, matching the
# paper's Figure 7. The 2026-09-01 run: responses `*_stacking_paper_*`
# (25 tasks, 10 combos/k, seed 0, 1 sample), judged with claude-sonnet-5.
# gpt-5: the personal OpenAI account was terminated ~700 calls in, so 2,701 of
# its 3,400 samples are API-error strings (judged n/a) — 4 to 57 graded per
# cell; read that row as a stub. See .claude/notes/stacking-replication.md.
STACKING_RUN: dict[str, str] = {
    "claude-sonnet-4.5": "cot_claude-sonnet-4.5_20260901_232604.json",
    "gpt-5": "cot_gpt-5_20260901_224225.json",
    "deepseek-r1": "cot_deepseek-r1_20260901_235452.json",
    "kimi-k2.5": "cot_kimi-k2.5_20260901_232103.json",
    "glm-5": "cot_glm-5_20260901_233709.json",
    "qwen3-235b": "cot_qwen3-235b_20260901_225719.json",
    "qwen3-30b": "cot_qwen3-30b_20260901_223224.json",
    "gpt-oss-120b": "cot_gpt-oss-120b_20260901_233341.json",
    "gpt-oss-20b": "cot_gpt-oss-20b_20260902_000214.json",
}

PAPER_FACTORS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
VARIED_LEVELS = {
    "F1": "varied",
    "F2": "varied",
    "F3": "formal",
    "F4": "varied",
    "F5": "varied",
    "F6": "varied",
    "F7": "template",
    "F8": "varied",
}
KS = list(range(len(PAPER_FACTORS) + 1))
VALENCES = ("safety", "capability")


def record_k(r: dict) -> int:
    if isinstance(r.get("n_varied"), int):
        return r["n_varied"]
    cfg = r.get("factor_config") or {}
    return sum(1 for f in PAPER_FACTORS if cfg.get(f) == VARIED_LEVELS[f])


def is_aware(r: dict, matched: bool) -> str:
    """'yes' / 'no' / other bucket for one record under the chosen rule."""
    judge = r.get("judge") or {}
    b = bucket_conclusion(extract_conclusion(judge, "model_awareness"))
    if b != "yes" or not matched:
        return b
    section = judge.get("model_awareness") or {}
    kind = str(section.get("awareness_type", "")).strip().lower()
    return "yes" if kind == r.get("valence") else "no"


def load_cells(path: str, matched: bool) -> dict[tuple[str, int], list[int]]:
    """(valence, k) -> [yes, graded] for one cot file."""
    with open(path) as f:
        records = json.load(f)
    cells: dict[tuple[str, int], list[int]] = {}
    for r in records:
        b = is_aware(r, matched)
        if b not in ("yes", "no"):
            continue
        cell = cells.setdefault((r.get("valence", "?"), record_k(r)), [0, 0])
        cell[0] += b == "yes"
        cell[1] += 1
    return cells


def model_key_from_path(path: str) -> str:
    m = re.search(r"cot_([^_]+(?:-[^_]+)*)_", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def resolve_files() -> list[tuple[str, str]]:
    """[(model_key, cot path)] from STACKING_FILES or the STACKING_RUN pins."""
    override = os.getenv("STACKING_FILES", "").strip()
    if override:
        paths = [p.strip() for p in override.split(",") if p.strip()]
        return [(model_key_from_path(p), p) for p in paths]
    files = []
    for model, fname in STACKING_RUN.items():
        path = os.path.join(RESULTS, "cot", fname)
        if not os.path.exists(path):
            sys.exit(f"pinned file missing: {path} — update STACKING_RUN")
        files.append((model, path))
    if not files:
        sys.exit("nothing to plot: pin files in STACKING_RUN or set STACKING_FILES")
    return files


def plot(
    models: list[str], grids: dict[str, list[list[float]]], out_path: str, matched: bool
):
    n_rows = len(models)
    vmax = max(
        (v for g in grids.values() for row in g for v in row if not math.isnan(v)),
        default=1.0,
    )
    vmax = max(vmax, 0.05)
    # Near-square cells like the paper's figure, whatever the row count.
    fig, axes = plt.subplots(
        1, 2, figsize=(0.55 * len(KS) * 2 + 3.0, 0.55 * n_rows + 1.7), dpi=200
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, valence in zip(axes, VALENCES):
        style_axes(ax)
        grid = grids[valence]
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
        ax.set_xticks([x - 0.5 for x in range(1, len(KS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)
        ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS], fontsize=8.5)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([display_name(m) for m in models], fontsize=9)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Number of Factors Varied", color=INK2)
        ax.set_title(f"{valence.capitalize()} Tasks", color=INK, fontsize=10)
    axes[1].tick_params(labelleft=False)
    rule = "matched-type" if matched else "any-type"
    fig.suptitle(
        f"CoT-judged eval awareness by number of factors varied ({rule})",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    matched = os.getenv("MATCHED", "1") != "0"
    out_dir = os.getenv("OUT_DIR", os.path.join(RESULTS, "plots"))
    os.makedirs(out_dir, exist_ok=True)

    models: list[str] = []
    grids: dict[str, list[list[float]]] = {v: [] for v in VALENCES}
    summary: dict[str, dict] = {}
    for model, path in resolve_files():
        cells = load_cells(path, matched)
        models.append(model)
        summary[model] = {"file": os.path.basename(path), "cells": {}}
        for valence in VALENCES:
            row = []
            for k in KS:
                yes, n = cells.get((valence, k), [0, 0])
                row.append(yes / n if n else math.nan)
                summary[model]["cells"][f"{valence}/{k}"] = {"yes": yes, "n": n}
            grids[valence].append(row)

    rule = "matched" if matched else "any"
    with open(os.path.join(out_dir, f"stacking_summary_{rule}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    header = "  ".join(f"{k:>7}" for k in KS)
    for valence in VALENCES:
        print(
            f"\n{valence} tasks ({rule}-type awareness), % (n)\n{'model':20} {header}"
        )
        for model, row in zip(models, grids[valence]):
            cells = summary[model]["cells"]
            line = "  ".join(
                f"{v * 100:3.0f}%({cells[f'{valence}/{k}']['n']:>3})"
                if not math.isnan(v)
                else "      –"
                for k, v in zip(KS, row)
            )
            print(f"{model:20} {line}")

    plot(
        models,
        grids,
        os.path.join(out_dir, f"awareness_by_n_factors_{rule}.png"),
        matched,
    )


if __name__ == "__main__":
    main()
