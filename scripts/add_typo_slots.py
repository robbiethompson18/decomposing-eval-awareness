"""Author the F13 `{TYPO_*}` slots for eval_bench tasks, in three steps.

python scripts/add_typo_slots.py extract OUT_DIR N_BATCHES [SEED.json]
    For every task with no TYPO slots yet, write OUT_DIR/batch_<i>.json:
    {task_id: {"shells": {"F7_x/F3_x/F4_x": clean text, ... 8 shells},
               "pool": {word: typo}   # seeded from SEED.json if given
               "uncovered": [shell keys with < 2 pool words present]}}
    The authoring job is to grow each pool until `uncovered` is empty.
python scripts/add_typo_slots.py check POOLS.json ...
    Validate {task_id: {word: typo}} pool files: every pair is a small edit
    of a lowercase word and every shell has >= 2 pool words present as whole
    tokens (eval_bench/typos.py contract). Touches nothing.
python scripts/add_typo_slots.py apply POOLS.json ...
    Same validation, then edit each task YAML in place: the first two pool
    words present in each shell become {TYPO_WORD} placeholders and the
    matching slots are inserted under `slots:`. Edits are made on the raw
    text (a yaml round-trip rewraps most files) and verified by re-parsing.
"""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_bench.typos import (
    N_TYPOS,
    SHELL_KEYS,
    TYPO_SLOT_PREFIX,
    required_typos,
    slot_name,
    validate_typo_pair,
    word_pattern,
)

TASKS_DIR = Path(__file__).resolve().parent.parent / "eval_bench" / "tasks"
MIN_TOKEN_GAP = 5  # keep a shell's two typos apart


def task_files() -> dict[str, Path]:
    return {
        yaml.safe_load(p.read_text())["task_id"]: p
        for p in sorted(TASKS_DIR.glob("*/*.yaml"))
    }


def shell_text(template: dict, key: tuple[str, str, str]) -> str:
    f7, f3, f4 = key
    return template["shells"][f7][f3][f4]


def has_typo_slots(template: dict) -> bool:
    return any(name.startswith(TYPO_SLOT_PREFIX) for name in template["slots"])


def pick_words(text: str, pool: dict[str, str]) -> list[tuple[str, int]]:
    """The first N_TYPOS pool words present in `text` as whole tokens, with
    their match offsets, preferring a second word that isn't next to the
    first. Pool order is the author's priority order."""
    hits = []
    for word in pool:
        m = word_pattern(word).search(text)
        if m:
            hits.append((word, m.start()))
    if len(hits) < N_TYPOS:
        return hits[: required_typos(text)]
    first = hits[0]
    for second in hits[1:]:
        gap = len(text[min(first[1], second[1]) : max(first[1], second[1])].split())
        if gap >= MIN_TOKEN_GAP:
            return [first, second]
    return hits[:N_TYPOS]


def uncovered_shells(template: dict, pool: dict[str, str]) -> list[str]:
    return [
        "/".join(key)
        for key in SHELL_KEYS
        if len(pick_words(shell_text(template, key), pool))
        < required_typos(shell_text(template, key))
    ]


def extract(out_dir: Path, n_batches: int, seed: dict[str, dict[str, str]]) -> None:
    pending = {}
    for task_id, path in task_files().items():
        template = yaml.safe_load(path.read_text())
        if has_typo_slots(template):
            continue
        pool = seed.get(task_id, {})
        pending[task_id] = {
            "shells": {"/".join(k): shell_text(template, k) for k in SHELL_KEYS},
            "pool": pool,
            "uncovered": uncovered_shells(template, pool),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(pending)
    per = -(-len(ids) // n_batches)
    for i in range(n_batches):
        chunk = {t: pending[t] for t in ids[i * per : (i + 1) * per]}
        (out_dir / f"batch_{i}.json").write_text(json.dumps(chunk, indent=2))
    print(f"{len(pending)} tasks pending, {n_batches} batches in {out_dir}")


def load_pools(paths: list[Path]) -> dict[str, dict[str, str]]:
    pools: dict[str, dict[str, str]] = {}
    for p in paths:
        pools.update(json.loads(p.read_text()))
    return pools


def check(pools: dict[str, dict[str, str]]) -> bool:
    files = task_files()
    ok = True
    for task_id, pool in pools.items():
        template = yaml.safe_load(files[task_id].read_text())
        problems = []
        for word, typo in pool.items():
            problems += validate_typo_pair(word, typo)
        problems += [f"uncovered shell {s}" for s in uncovered_shells(template, pool)]
        if problems:
            ok = False
            print(f"FAIL {task_id}: " + "; ".join(problems))
    print(f"{len(pools)} checked, {'all ok' if ok else 'FAILURES above'}")
    return ok


def _shell_span(src: str, key: tuple[str, str, str]) -> tuple[int, int]:
    """[start, end) character offsets of this shell's quoted scalar in the raw
    file, from the YAML parser's own marks (handles both quoting styles)."""
    node = yaml.compose(src)
    for name in ("shells", *key):
        node = next(v for k, v in node.value if k.value == name)
    return node.start_mark.index, node.end_mark.index


def _replace_in_raw(raw: str, word: str, expected_after_parse: str) -> str:
    """Turn one occurrence of `word` in a raw quoted YAML scalar into
    {TYPO_WORD}: try each raw substring match until the scalar re-parses to
    the expected text (raw text has folding, '' and backslash escapes, so
    token boundaries are only checked on the parsed result)."""
    for raw_word in (word, word.replace("'", "''")):
        for m in re.finditer(re.escape(raw_word), raw):
            candidate = raw[: m.start()] + "{" + slot_name(word) + "}" + raw[m.end() :]
            if yaml.safe_load(candidate) == expected_after_parse:
                return candidate
    raise ValueError(f"could not place {word!r} in raw scalar")


def apply(pools: dict[str, dict[str, str]]) -> None:
    if not check(pools):
        sys.exit(1)
    files = task_files()
    for task_id, pool in pools.items():
        path = files[task_id]
        src = path.read_text()
        template = yaml.safe_load(src)
        if has_typo_slots(template):
            print(f"skip {task_id}: already has TYPO slots")
            continue
        new_src = src
        used: dict[str, str] = {}
        # Edit shells bottom-up so earlier spans stay valid.
        for key in sorted(
            SHELL_KEYS, key=lambda k: _shell_span(src, k)[0], reverse=True
        ):
            start, end = _shell_span(src, key)
            raw = src[start:end]
            expected = shell_text(template, key)
            for word, _ in pick_words(expected, pool):
                expected = word_pattern(word).sub(
                    "{" + slot_name(word) + "}", expected, count=1
                )
                raw = _replace_in_raw(raw, word, expected)
                used[word] = pool[word]
            new_src = new_src[:start] + raw + new_src[end:]
        slot_block = yaml.safe_dump(
            {
                slot_name(w): {"factor": "F13", "typos": t, "clean": w}
                for w, t in used.items()
            },
            sort_keys=False,
            allow_unicode=True,
            width=80,
        )
        indented = "".join(
            "  " + line + "\n" for line in slot_block.rstrip("\n").split("\n")
        )
        shells_at = new_src.index("\nshells:\n") + 1
        new_src = new_src[:shells_at] + indented + new_src[shells_at:]
        # Verify: every shell re-parses to clean-with-placeholders, slots present.
        new_template = yaml.safe_load(new_src)
        for key in SHELL_KEYS:
            got = shell_text(new_template, key)
            back = got
            for w in used:
                back = back.replace("{" + slot_name(w) + "}", w)
            assert back == shell_text(template, key), (task_id, key)
            n_expected = required_typos(shell_text(template, key))
            assert len(re.findall(r"\{TYPO_[A-Z0-9_]+\}", got)) == n_expected, (
                task_id,
                key,
            )
        assert all(slot_name(w) in new_template["slots"] for w in used), task_id
        path.write_text(new_src)
    print(f"applied {len(pools)}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "extract":
        seed = json.loads(Path(args[2]).read_text()) if len(args) > 2 else {}
        extract(Path(args[0]), int(args[1]), seed)
    elif cmd == "check":
        sys.exit(0 if check(load_pools([Path(a) for a in args])) else 1)
    elif cmd == "apply":
        apply(load_pools([Path(a) for a in args]))
    else:
        sys.exit(__doc__)
