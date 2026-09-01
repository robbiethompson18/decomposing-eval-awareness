import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from lib.chat import Chat, ChatMessage

_BASE_DIR = Path(__file__).resolve().parent.parent

# The "bare" system prompt: the bland benchmark-style framing. Kept as a module
# symbol because the HarmBench anchor harness (eval_bench/harmbench.py) and its
# tests reference it directly.
DEFAULT_SYSTEM_PROMPT = (
    (_BASE_DIR / "prompts" / "chat_prompt_realistic.txt").read_text().strip()
)

# F9 (deployment framing) is the ONLY thing that sets the system prompt for the
# eval_bench grid — there is no per-task system prompt (render() raises if a task
# tries to set one). `deployment` is a realistic product system prompt (T3-style,
# naturalistic) and is the baseline; `bare` is the bland benchmark prompt (the
# eval-cue ablation level); `none` sends no system prompt at all (the no-harness
# "arm A" condition, picked with run_eval_bench F9_BASELINE=none). All prompt
# text lives in prompts/*.txt.
_F9_SYSTEM: dict[str, str | None] = {
    "deployment": (
        (_BASE_DIR / "prompts" / "chat_prompt_t3chat.txt").read_text().strip()
    ),
    "bare": DEFAULT_SYSTEM_PROMPT,
    "none": None,
}
F9_LEVELS = tuple(_F9_SYSTEM)

_LEFTOVER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


def _eastern_timestamp(years_ago: int = 0) -> str:
    """US East Coast time in JS ``Date().toString()`` format, e.g.
    'Tue Sep 01 2026 14:00:00 GMT-0400 (Eastern Daylight Time)'. DST-aware.

    Fills the {TIMESTAMP} placeholder in the deployment system prompt so each
    generation sees a real current time, like a real deployment would inject.
    `years_ago` shifts the calendar year back (F10 stale level) while keeping
    the real wall-clock time of day; Feb 29 falls back to Feb 28."""
    now = datetime.now(ZoneInfo("America/New_York"))
    if years_ago:
        try:
            now = now.replace(year=now.year - years_ago)
        except ValueError:  # Feb 29 in a non-leap target year
            now = now.replace(year=now.year - years_ago, day=28)
    total_min = int((now.utcoffset() or timedelta(0)).total_seconds()) // 60
    sign = "+" if total_min >= 0 else "-"
    hh, mm = divmod(abs(total_min), 60)
    tz_long = "Eastern Daylight Time" if now.dst() else "Eastern Standard Time"
    return (
        now.strftime("%a %b %d %Y %H:%M:%S") + f" GMT{sign}{hh:02d}{mm:02d} ({tz_long})"
    )


# F10 (deployment date) picks the timestamp injected into the deployment
# prompt's {TIMESTAMP}: `current` is the real now (baseline); `stale` is the
# same wall-clock time one year ago (the eval-cue level: a model that knows its
# training cutoff sees a date that could plausibly be a stored eval rather than
# a live session). Only observable under F9=deployment — the bare prompt has
# no timestamp, so F10 is a no-op there (including in all_varied).
_F10_YEARS_AGO = {"current": 0, "stale": 1}


def _resolve_slots(template: dict, config: dict) -> dict:
    """Resolve fixed constants + factor-controlled slots to name -> value."""
    resolved: dict = {}
    if "fixed" in template:
        for name, value in template["fixed"].items():
            resolved[name] = value
    for slot_name, slot_def in template["slots"].items():
        factor = slot_def["factor"]
        # Register-conditional fills (F3_casual / F3_formal sub-dicts).
        if f"F3_{config['F3']}" in slot_def:
            register_dict = slot_def[f"F3_{config['F3']}"]
            resolved[slot_name] = register_dict[config[factor]]
        else:
            resolved[slot_name] = slot_def[config[factor]]
    return resolved


def _fill(text: str, resolved: dict, task_id: str, config: dict) -> str:
    """Substitute {SLOT} placeholders (recursively), then fail on any leftover.

    A leftover {PLACEHOLDER} is an experimental confound (a visible template
    artifact is itself an evaluation cue), so fail loudly. Caps-only by design:
    slot names are all-caps, so this ignores legitimate braces in task content
    (C code bodies, a `{2}` regex quantifier in an SQLi payload).
    """
    for _ in range(10):  # cap iterations to prevent infinite loops
        new_text = text
        for name, value in resolved.items():
            new_text = new_text.replace(f"{{{name}}}", value)
        if new_text == text:
            break
        text = new_text
    leftover = _LEFTOVER.findall(text)
    if leftover:
        raise ValueError(
            f"Unresolved placeholders {leftover} in task {task_id} with config {config}"
        )
    return text


def render(template: dict, config: dict) -> Chat:
    """Render a task template + factor config into a `Chat`.

    Structure:
      - shells: F7_{prose|template} -> F3_{casual|formal} -> F4_{realistic|varied}
        -> the final user-message string (required).
      - history (optional): a list of {role, content} prior turns (content is a
        string with {SLOT} placeholders) prepended before the final user turn.

    The system prompt is NOT a task field: it is chosen solely by the F9 factor
    (deployment/bare/none). A task that defines `system`/`system_file` is a bug
    and raises.

    Returns a Chat: {"system": str | None, "messages": [...]} whose messages
    start with user, alternate, and end with the rendered user request.
    """
    task_id = template.get("task_id", "<unknown>")

    if "system" in template or "system_file" in template:
        raise ValueError(
            f"Task {task_id} defines a system prompt, but the eval_bench system "
            f"prompt is controlled solely by the F9 factor (deployment/bare). "
            f"Remove `system`/`system_file` from the template."
        )

    resolved = _resolve_slots(template, config)

    shell_text = template["shells"][f"F7_{config['F7']}"][f"F3_{config['F3']}"][
        f"F4_{config['F4']}"
    ]
    user_text = _fill(shell_text, resolved, task_id, config)

    # {TIMESTAMP} in the deployment prompt is filled per render with a real
    # time (F10 current) or the same time last year (F10 stale); the bare
    # prompt has no placeholder, so this is a no-op there, and F9=none has no
    # system prompt at all.
    system = _F9_SYSTEM[config.get("F9", "deployment")]
    if system is not None:
        system = system.replace(
            "{TIMESTAMP}",
            _eastern_timestamp(_F10_YEARS_AGO[config.get("F10", "current")]),
        )

    messages: list[ChatMessage] = []
    for turn in template.get("history", []):
        messages.append(
            {
                "role": turn["role"],
                "content": _fill(turn["content"], resolved, task_id, config),
            }
        )
    messages.append({"role": "user", "content": user_text})

    return {"system": system, "messages": messages}
