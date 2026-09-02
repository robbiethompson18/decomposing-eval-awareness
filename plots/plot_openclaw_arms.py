"""
Self-reported eval awareness under three delivery arms of the SAME request:
bare API (baseline), the real OpenClaw system prompt as text (prompt-only
control), and OpenClaw for real (live tools, sandbox, agent loop).

Reads results/eval_bench/openclaw/phase1_summary.json, written by
scripts/analyze_openclaw_phase1.py, and draws one heatmap per valence (rows =
models + pooled, columns = arms) in the same visual language as the factor
heatmap in plot_awareness.py. Denominator = yes+no; n printed under each cell.

    python plots/plot_openclaw_arms.py   [OUT_DIR=results/eval_bench/plots]
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.model_meta import display_name
from plots.style import INK, INK2, SURFACE, new_fig, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "eval_bench", "openclaw", "phase1_summary.json")
ARMS = [
    ("baseline", "bare API\n(T3 prompt,\nstub tool)"),
    ("control", "OpenClaw prompt\nas text\n(no live tools)"),
    ("harness", "OpenClaw\nfor real\n(tools, sandbox)"),
]
METRIC_TITLE = {
    "model_awareness": "Self-reported eval awareness",
    "hypothetical_recognition": "Self-reported hypothetical recognition",
}


def main() -> None:
    out_dir = os.getenv("OUT_DIR", os.path.join(ROOT, "results", "eval_bench", "plots"))
    os.makedirs(out_dir, exist_ok=True)
    with open(SUMMARY) as f:
        summary = json.load(f)
    for metric, per_model in summary.items():
        for valence in ("safety", "capability", "all"):
            grid, ns, labels = [], [], []
            pooled = {arm: [0, 0] for arm, _ in ARMS}
            for model, arms in per_model.items():
                row, nrow = [], []
                for arm, _ in ARMS:
                    yes, no, _other = arms.get(arm, {}).get(valence, (0, 0, 0))
                    n = yes + no
                    row.append(yes / n if n else math.nan)
                    nrow.append(n)
                    pooled[arm][0] += yes
                    pooled[arm][1] += n
                if all(math.isnan(v) for v in row):
                    continue
                grid.append(row)
                ns.append(nrow)
                labels.append(display_name(model))
            if not grid:
                continue
            grid.append([y / n if n else math.nan for y, n in pooled.values()])
            ns.append([n for _, n in pooled.values()])
            labels.append("Pooled")

            n_rows, n_cols = len(grid), len(ARMS)
            vmax = max(v for row in grid for v in row if not math.isnan(v)) or 1
            fig, ax = new_fig(w=1.6 * n_cols + 2.2, h=0.55 * n_rows + 1.8)
            ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
            for i, row in enumerate(grid):
                for j, v in enumerate(row):
                    if math.isnan(v):
                        ax.text(j, i, "–", ha="center", va="center", color=INK2)
                        continue
                    ink = "white" if v / vmax > 0.6 else INK
                    ax.text(
                        j,
                        i - 0.12,
                        f"{v * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=ink,
                    )
                    ax.text(
                        j,
                        i + 0.24,
                        f"n={ns[i][j]}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=ink,
                    )
            ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
            ax.grid(which="minor", color=SURFACE, lw=2)
            ax.tick_params(which="minor", length=0)
            ax.axhline(n_rows - 1.5, color=INK2, lw=1)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([lbl for _, lbl in ARMS], fontsize=8)
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(labels, fontsize=9)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            ax.set_xlabel("how the same request was delivered", color=INK2)
            ax.set_title(
                f"{METRIC_TITLE.get(metric, metric)} — {valence} tasks",
                color=INK,
                loc="left",
            )
            save(fig, out_dir, f"openclaw_arms_{metric}_{valence}.png")


if __name__ == "__main__":
    main()
