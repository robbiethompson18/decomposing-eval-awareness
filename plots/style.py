"""Shared matplotlib styling for repo plots: palette, axes chrome, figure I/O.

Palette is the dataviz reference categorical set (light mode); providers get
fixed color slots — never cycled, so a provider keeps its hue across charts.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Brand/logo colors (Robbie's call): each provider gets its own logo hue so a
# reader recognizes it without the legend. OpenAI's mark is black; Anthropic's is
# the "Claude" clay/orange; Google green; open-weight is a mixed category (5 labs)
# with no single logo, so it takes a distinct violet. Kept CVD-legible by the
# per-line direct labels every ladder chart carries (secondary encoding).
PROVIDER_COLOR = {
    "Anthropic": "#d97757",  # Claude clay/orange
    "OpenAI": "#0d0d0d",  # OpenAI black
    "Google": "#34a853",  # Google green
    "Open-weight": "#7c3aed",  # violet (category, not a single brand)
}

# Per-LAB brand colors, keyed by model_meta[*]["lab"]. Charts that want to break
# the open-weight category into its five labs (the ladder plots) use this instead
# of PROVIDER_COLOR. Colors are each lab's real logo hue where one is published
# (DeepSeek dodger blue #4d6bfe, Kimi/Moonshot sunglow gold, MiniMax blue), nudged
# only to keep the set distinguishable: DeepSeek/Qwen/MiniMax all brand blue-purple,
# so Qwen→fuchsia and MiniMax→cyan to separate them (normal-vision ΔE ≥ 21). Kimi's
# #FEC230 is darkened to #e0a100 so a gold line is visible on white. Z.ai/GLM has no
# published brand color — the red is a distinct placeholder, correct it if you have it.
# CVD worst-pair sits in the 6–8 ΔE band, legal because every ladder line is
# direct-labeled (secondary encoding). Validated: dataviz scripts/validate_palette.js.
LAB_COLOR = {
    "Anthropic": "#d97757",  # Claude clay/orange
    "OpenAI": "#0d0d0d",  # black
    "Google": "#34a853",  # Google green
    "Alibaba": "#c026d3",  # Qwen (purple-branded → fuchsia for separation)
    "DeepSeek": "#4d6bfe",  # DeepSeek dodger blue (published)
    "Moonshot": "#e0a100",  # Kimi sunglow gold (#FEC230), darkened for contrast
    "Z.ai": "#dc2626",  # GLM — no published hex; distinct placeholder
    "MiniMax": "#0891b2",  # MiniMax (blue-branded → cyan for separation)
}
# Safety/capability pair for valence splits; slots 8 and 7 of the same palette,
# deliberately disjoint from the provider slots above.
VALENCE_COLOR = {"safety": "#e34948", "capability": "#4a3aa7"}
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


def color_legend(ax, colors: dict[str, str], marker: bool = False, **kwargs):
    """Legend with one swatch (or marker) per entry of a fixed label->color map."""
    if marker:
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=c) for c in colors.values()
        ]
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), frameon=False, fontsize=8, **kwargs)
