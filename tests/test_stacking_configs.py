"""Factor-stacking design (STACKING=1): the paper's Figure 7 sampling scheme.

Run with:  PYTHONPATH=.venv/lib/python3.14/site-packages uvx --python .venv/bin/python .venv/bin/python -m pytest tests/
"""

import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_bench.render import DEFAULT_SYSTEM_PROMPT, render
from run.run_eval_bench import (
    FACTOR_VARIED_LEVELS,
    PAPER_BASE_CONFIG,
    PAPER_FACTORS,
    build_stacking_configs,
    load_all_tasks,
    n_varied,
)

TASKS = load_all_tasks()
TASK_NAMES = sorted({t["task_name"] for t in TASKS})


def test_paper_design_counts_per_k():
    """1, 8, 10, 10, 10, 10, 10, 8, 1 combos per task at k = 0..8 (10 where C(8,k) > 10)."""
    per_task = build_stacking_configs(TASK_NAMES, n_tasks=25, n_combos=10, seed=0)
    assert len(per_task) == 25
    for configs in per_task.values():
        counts = [0] * 9
        for name, cfg in configs:
            k = int(name[1 : name.index("_")])
            assert k == n_varied(cfg)
            counts[k] += 1
        assert counts == [1, 8, 10, 10, 10, 10, 10, 8, 1]
        assert len({name for name, _ in configs}) == len(configs)  # no duplicate combos


def test_small_combo_budget_exhausts_where_it_can():
    per_task = build_stacking_configs(TASK_NAMES, n_tasks=2, n_combos=2, seed=1)
    for configs in per_task.values():
        counts = [0] * 9
        for name, _ in configs:
            counts[int(name[1 : name.index("_")])] += 1
        assert counts == [1, 2, 2, 2, 2, 2, 2, 2, 1]


def test_names_encode_the_config():
    per_task = build_stacking_configs(TASK_NAMES, n_tasks=3, n_combos=10, seed=0)
    for configs in per_task.values():
        for name, cfg in configs:
            mask = name.split("_")[1]
            assert len(mask) == 8
            for bit, f in zip(mask, PAPER_FACTORS):
                expected = (
                    FACTOR_VARIED_LEVELS[f] if bit == "1" else PAPER_BASE_CONFIG[f]
                )
                assert cfg[f] == expected
            # F9–F13 pinned to the paper's harness in every cell.
            for f in ("F9", "F10", "F11", "F12", "F13"):
                assert cfg[f] == PAPER_BASE_CONFIG[f]
        # k=8 is the unique all-varied corner.
        full = [cfg for name, cfg in configs if name.startswith("S8_")]
        assert full == [
            {**PAPER_BASE_CONFIG, **{f: FACTOR_VARIED_LEVELS[f] for f in PAPER_FACTORS}}
        ]


def test_seeded_and_shared_across_models():
    a = build_stacking_configs(TASK_NAMES, 25, 10, seed=0)
    b = build_stacking_configs(TASK_NAMES, 25, 10, seed=0)
    c = build_stacking_configs(TASK_NAMES, 25, 10, seed=1)
    assert a == b
    assert set(a) != set(c) or a != c


def test_every_k_subset_is_a_real_subset():
    per_task = build_stacking_configs(TASK_NAMES, n_tasks=1, n_combos=10, seed=0)
    (configs,) = per_task.values()
    for name, cfg in configs:
        k = n_varied(cfg)
        varied = tuple(f for f in PAPER_FACTORS if cfg[f] == FACTOR_VARIED_LEVELS[f])
        assert varied in set(combinations(PAPER_FACTORS, k))


def test_paper_base_renders_like_the_paper_harness():
    """Bare system prompt, single user turn, no tools — no model key needed (cold)."""
    per_task = build_stacking_configs(TASK_NAMES, n_tasks=1, n_combos=10, seed=0)
    (task_name,) = per_task
    for task in (t for t in TASKS if t["task_name"] == task_name):
        for _name, cfg in per_task[task_name]:
            chat = render(task["template"], cfg)
            assert chat["system"] == DEFAULT_SYSTEM_PROMPT
            assert [m["role"] for m in chat["messages"]][-1] == "user"
            assert "tools" not in chat
