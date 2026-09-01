"""The `harmbench_verbatim` anchor config: the original HarmBench behavior, as-is.

Every eval_bench task directory is named after the HarmBench BehaviorID it was
adapted from (all 100 match verbatim). The factor grid (baseline + F1..F8)
never reproduces the original benchmark text: `realistic` levels *add*
naturalism HarmBench never had (casual first person, backstory, named
entities) and `varied` levels *add* artificiality it never had (DAN wrappers,
placeholder entities, false premises, eval suffix). Verbatim HarmBench is a
third point — bare and blunt — so it gets its own config rather than a grid
cell. See .claude/notes/harmbench-anchor-configs.md for the scoring.

The user message follows HarmBench's own DirectRequest baseline
(baselines/direct_request/direct_request.py): the Behavior string alone, or
"{ContextString}\\n\\n---\\n\\n{Behavior}" for contextual behaviors. Safety
arm only: HarmBench has no benign twins.
"""

import csv
from pathlib import Path

from eval_bench.render import DEFAULT_SYSTEM_PROMPT
from lib.chat import Chat

HARMBENCH_VERBATIM = "harmbench_verbatim"

# 100-row subset of HarmBench's harmbench_behaviors_text_all.csv (MIT license,
# github.com/centerforaisafety/HarmBench), filtered to the BehaviorIDs that
# have an eval_bench task directory. Vendored so runs are reproducible offline.
SUBSET_CSV = Path(__file__).resolve().parent / "harmbench_subset.csv"

# Where verbatim HarmBench sits on the taxonomy, so downstream code that indexes
# factor_config by F1..F8 keeps working. Two values are off the grid on purpose:
# F1 is "n/a" when the behavior names no entities at all (the 53 "standard"
# behaviors), and F3 is "stem" — the bare-imperative benchmark register that the
# taxonomy's definition calls a cue but neither of its levels (casual/formal)
# implements. F4 (no surrounding context) and F5 (names the harm outright) are
# the two varied levels HarmBench genuinely has.
_FACTOR_VECTOR = {
    "F1": "n/a",
    "F2": "realistic",
    "F3": "stem",
    "F4": "varied",
    "F5": "varied",
    "F6": "realistic",
    "F7": "prose",
    "F8": "realistic",
}


def load_harmbench_subset() -> dict[str, dict[str, str]]:
    """BehaviorID -> row for the 100 vendored HarmBench behaviors."""
    with open(SUBSET_CSV, newline="", encoding="utf-8") as f:
        return {r["BehaviorID"]: r for r in csv.DictReader(f)}


def harmbench_verbatim_text(row: dict[str, str]) -> str:
    """The user message exactly as HarmBench's DirectRequest baseline sends it."""
    behavior = row["Behavior"].strip()
    context = (row.get("ContextString") or "").strip()
    return f"{context}\n\n---\n\n{behavior}" if context else behavior


def harmbench_verbatim_chat(row: dict[str, str]) -> Chat:
    """Verbatim behavior under the same baseline system prompt as every grid cell."""
    return {
        "system": DEFAULT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": harmbench_verbatim_text(row)}],
    }


def harmbench_factor_config(row: dict[str, str]) -> dict[str, str]:
    """Taxonomy position of this verbatim behavior (see _FACTOR_VECTOR)."""
    cfg = dict(_FACTOR_VECTOR)
    if (row.get("ContextString") or "").strip():
        # Contextual behaviors carry HarmBench's own dossier of real-looking
        # entities (the baseline grid cell reuses these same entities).
        cfg["F1"] = "realistic"
    return cfg
