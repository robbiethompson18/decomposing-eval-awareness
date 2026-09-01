"""Shared matplotlib styling for repo plots: palette, axes chrome, figure I/O.

Palette is the dataviz reference categorical set (light mode); providers get
fixed color slots — never cycled, so a provider keeps its hue across charts.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROVIDER_COLOR = {
    "Anthropic": "#2a78d6",
    "OpenAI": "#eb6834",
    "Google": "#1baf7a",
    "Open-weight": "#eda100",
}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#e5e4e0"


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


def save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def provider_legend(ax, marker: bool = False, **kwargs):
    """Legend with one entry per provider, matching the fixed color slots."""
    if marker:
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=c)
            for c in PROVIDER_COLOR.values()
        ]
    else:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c) for c in PROVIDER_COLOR.values()
        ]
    ax.legend(handles, PROVIDER_COLOR.keys(), frameon=False, fontsize=8, **kwargs)
