import re
from pathlib import Path

from lib.chat import Chat, ChatMessage

# Baseline scenario system prompt, used when a task template defines no `system`
# of its own. This is NOT a run-level knob (there is deliberately no way to swap
# it per run) — it is the default scenario framing that render() bakes into the
# Chat, so a task can override it but the benchmark's baseline behavior is
# unchanged. Kept in prompts/ so all prompt text lives in .txt.
_BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT = (
    (_BASE_DIR / "prompts" / "chat_prompt_realistic.txt").read_text().strip()
)

_LEFTOVER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


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
      - system (optional): a string with {SLOT} placeholders. Defaults to
        DEFAULT_SYSTEM_PROMPT when absent.
      - history (optional): a list of {role, content} prior turns (content is a
        string with {SLOT} placeholders) prepended before the final user turn.

    Returns a Chat: {"system": str | None, "messages": [...]} whose messages
    start with user, alternate, and end with the rendered user request.
    """
    task_id = template.get("task_id", "<unknown>")
    resolved = _resolve_slots(template, config)

    shell_text = template["shells"][f"F7_{config['F7']}"][f"F3_{config['F3']}"][
        f"F4_{config['F4']}"
    ]
    user_text = _fill(shell_text, resolved, task_id, config)

    if "system" in template:
        system: str | None = _fill(template["system"], resolved, task_id, config)
    else:
        # TODO(cleanup): no task defines `system` yet, so fall back to the
        # baseline (keeps all 104 tasks rendering unchanged). Once every task
        # YAML carries its own `system`, drop this default and require it —
        # ideally behind a STRICT_SYSTEM flag so the switch is deliberate rather
        # than a silent break. See PR #41 discussion.
        system = DEFAULT_SYSTEM_PROMPT

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
