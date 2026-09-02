"""F13 (typos): every shell of every task carries its {TYPO_*} placeholders,
each backed by a valid F13 slot, and the typos/clean levels render as
expected in every F7/F3/F4 shell combination."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_bench.render import render
from eval_bench.typos import (
    SHELL_KEYS,
    TYPO_SLOT_PREFIX,
    placeholders,
    required_typos,
    validate_typo_pair,
)
from run.run_eval_bench import ALL_VARIED, build_configs, load_all_tasks

TASKS = load_all_tasks()
CONFIGS = dict(build_configs())


def _cold(cfg):
    return {**cfg, "F11": "cold"}  # no opener reply needed


def test_validator_contract():
    assert validate_typo_pair("honestly", "honeslty") == []
    assert validate_typo_pair("don't", "dont") == []
    assert validate_typo_pair("Honestly", "honeslty")  # capitalized
    assert validate_typo_pair("honestly", "honestly")  # no edit
    assert validate_typo_pair("honestly", "hnsty")  # too big an edit
    assert validate_typo_pair("the one", "teh one")  # not a bare word


def test_configs():
    assert CONFIGS["baseline"]["F13"] == "typos"
    assert CONFIGS["F13"] == {**CONFIGS["baseline"], "F13": "clean"}
    assert CONFIGS[ALL_VARIED]["F13"] == "clean"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["task_id"])
def test_every_shell_carries_valid_typo_slots(task):
    template = task["template"]
    typo_slots = {
        name: slot
        for name, slot in template["slots"].items()
        if name.startswith(TYPO_SLOT_PREFIX)
    }
    assert typo_slots, "no TYPO slots"
    for name, slot in typo_slots.items():
        assert slot["factor"] == "F13", name
        assert validate_typo_pair(slot["clean"], slot["typos"]) == [], name
    for f7, f3, f4 in SHELL_KEYS:
        shell = template["shells"][f7][f3][f4]
        found = [p.strip("{}") for p in placeholders(shell)]
        assert set(found) <= set(typo_slots), (f7, f3, f4, found)
        clean_shell = shell
        for name in found:
            clean_shell = clean_shell.replace(
                "{" + name + "}", typo_slots[name]["clean"]
            )
        assert len(found) == required_typos(clean_shell), (f7, f3, f4, found)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["task_id"])
def test_typos_level_differs_from_clean_in_exactly_the_typo_words(task):
    template = task["template"]
    for f7, f3, f4 in SHELL_KEYS:
        levels = {"F7": f7[3:], "F3": f3[3:], "F4": f4[3:]}
        typos_msg = render(template, _cold({**CONFIGS["baseline"], **levels}))
        clean_msg = render(template, _cold({**CONFIGS["F13"], **levels}))
        a = typos_msg["messages"][-1]["content"].split()
        b = clean_msg["messages"][-1]["content"].split()
        assert len(a) == len(b), (f7, f3, f4)
        n_expected = len(placeholders(template["shells"][f7][f3][f4]))
        assert sum(x != y for x, y in zip(a, b)) == n_expected, (f7, f3, f4)
