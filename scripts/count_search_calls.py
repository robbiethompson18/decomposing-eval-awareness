"""How often did models reach for the F12 web_search stub?

Reads response files and prints, per model x config x valence, how many
samples made at least one web_search call and how many calls in total, then a
few example queries per model. Rows for configs without the tool (baseline,
F1..F11, ...) should be zero — a nonzero there means the model called a tool it
was never offered.

Usage:
  python scripts/count_search_calls.py results/eval_bench/response/responses_kimi-k3_*.json
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.target import tool_calls_in

SEARCH_TOOL = "web_search"
EXAMPLES_PER_MODEL = 5


def main(paths: list[str]) -> None:
    # (model, config, valence) -> [n_samples, n_with_call, n_calls]
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    queries: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", os.path.basename(path))
        for s in data["samples"]:
            calls = [
                args
                for name, args in tool_calls_in(s.get("response", ""))
                if name == SEARCH_TOOL
            ]
            row = counts[(model, s["config_name"], s["valence"])]
            row[0] += 1
            row[1] += bool(calls)
            row[2] += len(calls)
            queries[model] += [str(a.get("query", a)) for a in calls]

    print(
        f"{'model':<20}{'config':<20}{'valence':<12}{'n':>6}{'with_call':>11}{'calls':>7}"
    )
    for (model, cfg, valence), (n, n_with, n_calls) in sorted(counts.items()):
        print(f"{model:<20}{cfg:<20}{valence:<12}{n:>6}{n_with:>11}{n_calls:>7}")

    for model, qs in sorted(queries.items()):
        print(f"\nExample {SEARCH_TOOL} queries, {model} ({len(qs)} total):")
        for q in qs[:EXAMPLES_PER_MODEL]:
            print(f"  - {q}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
