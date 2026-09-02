"""F13 (typos): the authoring contract for the `{TYPO_*}` slots.

F13 is an ordinary slot factor, like F1's entities: every shell of every task
carries exactly two `{TYPO_WORD}` placeholders, each backed by a slot

    TYPO_HONESTLY:
      factor: F13
      typos: honeslty      # baseline: real messages have typing slips
      clean: honestly      # the cell: copy-edited, as every benchmark item is

so typos compose with every other factor (all eight F7/F3/F4 shells have
them) and render() needs no F13-specific code. The typo'd spellings are
hand-picked per task and live in the YAML, reviewable in git;
scripts/add_typo_slots.py inserts the placeholders and slots from a per-task
pool of word -> typo pairs. tests/test_typos.py enforces this contract.
"""

import re

TYPO_SLOT_PREFIX = "TYPO_"
N_TYPOS = 2
MAX_EDIT_DISTANCE = 2  # transposition ("teh") is 2 Levenshtein edits

SHELL_KEYS = [
    (f7, f3, f4)
    for f7 in ("F7_prose", "F7_template")
    for f3 in ("F3_casual", "F3_formal")
    for f4 in ("F4_realistic", "F4_varied")
]

_PLACEHOLDER = re.compile(r"\{" + TYPO_SLOT_PREFIX + r"[A-Z0-9_]+\}")

# A word a typo may land on: a lowercase whole token of 3+ letters (trailing
# punctuation allowed), not glued to a placeholder. Some template/F4-varied
# shells are pure key: {VALUE} forms with fewer than two such words, so the
# contract is min(N_TYPOS, eligible words) rather than a flat two.
_ELIGIBLE = re.compile(r"(?<!\S)([a-z][a-z']{2,})(?=[.,?!;:]*(?:\s|$))")


def eligible_words(text: str) -> list[str]:
    return _ELIGIBLE.findall(text)


def required_typos(clean_text: str) -> int:
    return min(N_TYPOS, len(set(eligible_words(clean_text))))


def slot_name(word: str) -> str:
    return TYPO_SLOT_PREFIX + re.sub(r"[^a-z0-9]", "", word.lower()).upper()


def word_pattern(word: str) -> re.Pattern:
    """`word` as a whole whitespace token, allowing trailing punctuation. A
    token glued to a placeholder ({F6_PREFIX}hey, word.{F8_SUFFIX}) never
    matches, so a typo can't land next to braces."""
    return re.compile(r"(?<!\S)" + re.escape(word) + r"(?=[.,?!;:]*(?:\s|$))")


def placeholders(text: str) -> list[str]:
    return _PLACEHOLDER.findall(text)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def validate_typo_pair(clean: str, typo: str) -> list[str]:
    """Problems with `typo` as a typing slip of `clean`; empty if it's fine."""
    problems = []
    if not clean or not clean[0].islower():
        problems.append(f"not a lowercase word: {clean!r}")
    if re.search(r"[\s{}.,?!;:]", clean + typo):
        problems.append(f"not a bare word: {clean!r} -> {typo!r}")
    if not 1 <= _levenshtein(clean.lower(), typo.lower()) <= MAX_EDIT_DISTANCE:
        problems.append(f"not a small edit: {clean!r} -> {typo!r}")
    return problems
