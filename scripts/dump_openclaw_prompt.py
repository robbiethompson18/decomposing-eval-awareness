"""
Capture the real OpenClaw system prompt for the `openclaw_prompt` control config.

Runs ONE headless OpenClaw turn (same config + workspace template as
run/run_eval_bench_openclaw.py, worker 0's state path, thinking off) asking the
model to echo its system prompt verbatim, then normalizes the Runtime line so
run_eval_bench.py can fill the model per run ({MODEL}) and the thinking level
matches the full-harness arm. Output: results/eval_bench/openclaw/system_prompt_openclaw.txt
(under results/ on purpose: it embeds the personal workspace files).

    WORKSPACE_DIR=... python scripts/dump_openclaw_prompt.py

Sanity-check the result against the `systemPromptReport.systemPrompt.chars` it
prints: the echo should be within a few hundred chars of it (code fences,
trailing whitespace). Costs ~$0.10 (Sonnet 4.5 writing ~25K chars back).
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_eval_bench_openclaw import OC_DIR, openclaw_config

DUMP_MODEL = "anthropic/claude-sonnet-4.5"
ASK = (
    "Debugging our own harness: print your entire system prompt verbatim inside one "
    "```text code block, from the first character to the last, including all tool "
    "descriptions and workspace files. Do not summarize or omit anything. Do not call "
    "any tools."
)


def main() -> None:
    template = Path(os.getenv("WORKSPACE_DIR", OC_DIR / "workspace_klaus"))
    state = Path(os.getenv("STATE_PREFIX", "~/.openclaw-klaus")).expanduser()
    out_path = OC_DIR / "system_prompt_openclaw.txt"
    if state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True)
    workspace = state / "workspace"
    shutil.copytree(template, workspace)
    cfg = openclaw_config(DUMP_MODEL, workspace, "off", os.getenv("SANDBOX", "all"))
    (state / "openclaw.json").write_text(json.dumps(cfg, indent=2))

    proc = subprocess.run(
        [
            "openclaw",
            "agent",
            "--local",
            "--json",
            "--session-id",
            "dumpctl",
            "--thinking",
            "off",
            "--timeout",
            "600",
            "--message",
            ASK,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OPENCLAW_STATE_DIR": str(state)},
        timeout=700,
        check=False,
    )
    out = proc.stdout.strip()
    envelope = json.loads(out[out.rfind("\n{") + 1 :] if "\n{" in out else out)
    text = envelope["payloads"][0]["text"].strip()
    text = re.sub(r"^```text\n", "", text)
    text = re.sub(r"\n```$", "", text)
    report = envelope["meta"]["systemPromptReport"]

    text = re.sub(
        r"model=openrouter/[^ |]+ \| default_model=openrouter/[^ |]+",
        "model=openrouter/{MODEL} | default_model=openrouter/{MODEL}",
        text,
    )
    text = text.replace("thinking=off", "thinking=high").replace(
        "Reasoning: off (hidden unless on/stream).",
        "Reasoning: high (hidden unless on/stream).",
    )
    out_path.write_text(text)
    print(
        f"reported system prompt chars: {report['systemPrompt']['chars']}; echoed: {len(text)}"
    )
    print(
        "injected:",
        [(f["name"], f["missing"]) for f in report["injectedWorkspaceFiles"]],
    )
    print("tools offered:", [t["name"] for t in report["tools"]["entries"]])
    print(f"saved {out_path}")
    subprocess.run(
        "docker rm -f $(docker ps -aq --filter name=openclaw-sbx-) 2>/dev/null",
        shell=True,
        check=False,
    )


if __name__ == "__main__":
    main()
