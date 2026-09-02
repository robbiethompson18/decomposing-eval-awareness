"""
Deliver EvalAwareBench baseline requests through a REAL agent harness (OpenClaw)
instead of a bare API call, and probe self-reported awareness in-session.

Same 100 task pairs, same rendered request text as the `baseline` grid cell
(F1–F8 realistic; F9/F10/F12 are superseded by OpenClaw's own system prompt and
live tools; F11=warm is reproduced live with a real "are you there?" turn).
What changes is the environment: a ~29K-char production system prompt
(workspace files + skills + 23 tool schemas), tools that actually execute in a
Docker sandbox, a persona with months of memory files, and a multi-turn agent
loop. See .claude/notes/openclaw-realism-scope.md.

Per sample, four headless OpenClaw turns (`openclaw agent --local`), each its
own process:
  turn 0  "are you there?"                          (F11 warm, live reply)
  turn 1  the rendered task request
          fork: copy the session JSONL to <sid>__hypo so the two probes stay
          independent, like eval_eval_bench_selfreport.py's two continuations
  turn 2  prompts/selfreport_awareness.txt      on <sid>
  turn 3  prompts/selfreport_hypothetical.txt   on <sid>__hypo

Outputs (standard shapes, so eval_eval_bench_cot.py / plots work unchanged):
  results/eval_bench/response/responses_<model>_openclaw[_TAG]_<ts>.json
  results/eval_bench/selfreport/selfreport_<model>_openclaw[_TAG]_<ts>.json
  results/eval_bench/openclaw/runs/<model>_<ts>/   config + archived session
      JSONLs (full transcripts incl. tool calls), copied from the live state dirs
      ~/.openclaw-klaus[-N]/ (see Worker for why they live there)

Environment variables:
    MODEL_KEY        Repo model key (lib.config.DEFAULT_OPENROUTER_CONFIGS); routed via OpenRouter
    WORKSPACE_DIR    Workspace template (default: results/eval_bench/openclaw/workspace_klaus).
                     Each worker gets its own copy, re-synced before every sample, so a model
                     that edits memory files can't contaminate the next sample.
    MAX_CONCURRENT   Parallel samples = parallel OpenClaw processes + sandboxes (default: 12)
    THINKING         OpenClaw thinking level off|minimal|low|medium|high (default: high — the
                     bare-API OpenRouter baseline runs reasoning effort "high")
    SANDBOX          all (default) | off — Docker sandbox for exec/read/write; needs the
                     openclaw-sandbox:bookworm-slim image (see the note for the Dockerfile)
    OPENER           1 (default) sends the "are you there?" opener turn first; 0 skips it
    TASK_FILTER / VALENCE_FILTER / MAX_SAMPLES   as in run_eval_bench.py
    RUN_TAG          Suffix for output filenames
    RESUME_FILE      Existing responses_*_openclaw*.json to skip completed ids from
    TURN_TIMEOUT     Seconds per OpenClaw turn (default: 600)
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.selfreport import parse_selfreport_awareness, parse_selfreport_hypothetical
from eval_bench.render import OPENER, render
from lib.config import DEFAULT_OPENROUTER_CONFIGS
from lib.io import read_file
from lib.time_utils import now_pacific
from run.run_eval_bench import BASELINE_CONFIG, load_all_tasks

BASE_DIR = Path(__file__).resolve().parent.parent
OC_DIR = BASE_DIR / "results" / "eval_bench" / "openclaw"
CONFIG_NAME = "openclaw"

# Tools with side effects outside the sandbox, or that reach other sessions /
# channels. Everything else OpenClaw ships stays on: read/write/edit/apply_patch,
# exec/process (sandboxed, no network), web_search/web_fetch (gateway-side),
# session_status. (memory_* are off via memorySearch.enabled, see openclaw_config.)
TOOLS_DENY = [
    "browser",
    "canvas",
    "nodes",
    "cron",
    "message",
    "gateway",
    "agents_list",
    "sessions_list",
    "sessions_history",
    "sessions_send",
    "sessions_spawn",
    "subagents",
    "tts",
]
SANDBOX_TOOLS_ALLOW = [
    "exec",
    "process",
    "read",
    "write",
    "edit",
    "apply_patch",
    "web_search",
    "web_fetch",
    "session_status",
]


def openclaw_config(
    model_id: str, workspace: Path, thinking: str, sandbox: str
) -> dict:
    cfg: dict[str, Any] = {
        "agents": {
            "defaults": {
                "model": {"primary": f"openrouter/{model_id}"},
                "workspace": str(workspace),
                "thinkingDefault": thinking,
                "timeoutSeconds": int(os.getenv("TURN_TIMEOUT", "600")),
                # memory_search needs an embedding provider; without one a turn
                # that calls it answers fine but the process never exits (hangs
                # to the timeout). Off = no memory_* tools and no "Memory Recall"
                # prompt section, so the prompt stays coherent. MEMORY.md is still
                # injected and memory/*.md are still readable with read/exec.
                "memorySearch": {"enabled": False},
            }
        },
        "tools": {
            "deny": TOOLS_DENY,
            # Perplexity Sonar rides on the OpenRouter key we already have.
            "web": {"search": {"provider": "perplexity"}},
            # Sandboxed sessions get OpenClaw's own narrow default allow-list
            # (exec/process/read/write/edit/sessions_*), which would silently drop
            # web + memory tools while the prompt still tells the model to use them.
            "sandbox": {"tools": {"allow": SANDBOX_TOOLS_ALLOW}},
        },
    }
    if sandbox == "all":
        cfg["agents"]["defaults"]["sandbox"] = {
            "mode": "all",
            "scope": "session",
            "workspaceAccess": "rw",
            "docker": {"network": "none"},
        }
    return cfg


class Worker:
    """One OpenClaw state dir + workspace copy. Sessions are per-sample ids.

    State dirs live at ~/.openclaw-klaus, ~/.openclaw-klaus-2, ... — OpenClaw's own
    `--profile` naming — NOT under results/: the workspace path is printed into
    the system prompt ("Your working directory is: ...", one header per injected
    file), and a path containing "eval_bench" would be a giant eval cue. Session
    JSONLs are copied into the run dir under results/ by `archive()`.
    """

    def __init__(self, idx: int, template: Path, cfg: dict, archive_root: Path):
        prefix = Path(os.getenv("STATE_PREFIX", "~/.openclaw-klaus")).expanduser()
        # WORKER_OFFSET lets two model runs share the machine without sharing
        # state dirs (lane A: klaus, klaus-2..; lane B with offset 10: klaus-11..).
        idx += int(os.getenv("WORKER_OFFSET", "0"))
        self.state = prefix if idx == 0 else Path(f"{prefix}-{idx + 1}")
        self.workspace = self.state / "workspace"
        self.template = template
        self.archive_root = archive_root / self.state.name
        if self.state.exists():
            shutil.rmtree(self.state)
        self.state.mkdir(parents=True)
        cfg = json.loads(json.dumps(cfg))
        cfg["agents"]["defaults"]["workspace"] = str(self.workspace)
        (self.state / "openclaw.json").write_text(json.dumps(cfg, indent=2))
        self.reset_workspace()

    def reset_workspace(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        shutil.copytree(self.template, self.workspace)

    def archive(self) -> None:
        """Copy this worker's session transcripts into the run dir (results/)."""
        src = self.sessions_dir()
        if src.exists():
            shutil.copytree(src, self.archive_root / "sessions", dirs_exist_ok=True)

    def sessions_dir(self) -> Path:
        return self.state / "agents" / "main" / "sessions"

    def turn(self, sid: str, message: str, thinking: str) -> dict[str, Any]:
        env = {**os.environ, "OPENCLAW_STATE_DIR": str(self.state)}
        timeout = int(os.getenv("TURN_TIMEOUT", "600"))
        cmd = [
            "openclaw",
            "agent",
            "--local",
            "--json",
            "--session-id",
            sid,
            "--thinking",
            thinking,
            "--timeout",
            str(timeout),
            "--message",
            message,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout + 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"turn timed out after {timeout + 60}s"}
        out = proc.stdout.strip()
        # The CLI prints log lines before the JSON envelope; take the last {...}.
        start = out.rfind("\n{")
        blob = out[start + 1 :] if start >= 0 else out
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return {
                "error": f"no JSON envelope (exit {proc.returncode})",
                "stdout": out[-2000:],
                "stderr": proc.stderr[-2000:],
            }

    def fork_session(self, sid: str, new_sid: str) -> None:
        src = self.sessions_dir() / f"{sid}.jsonl"
        shutil.copyfile(src, self.sessions_dir() / f"{new_sid}.jsonl")

    def cleanup_containers(self) -> None:
        """Remove this worker's sandbox containers (scope=session leaves one per
        session; don't let 200 pile up). Docker truncates the session id in the
        container name, so match on the bind-mounted workspace path instead —
        unique per worker, and a worker runs one sample at a time."""
        try:
            all_ids = subprocess.run(
                ["docker", "ps", "-aq", "--filter", "name=openclaw-sbx-"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.split()
            ids = []
            if all_ids:
                info = json.loads(
                    subprocess.run(
                        ["docker", "inspect", *all_ids],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    ).stdout
                    or "[]"
                )
                ids = [
                    c["Id"]
                    for c in info
                    if any(
                        m.get("Source") == str(self.workspace)
                        for m in c.get("Mounts", [])
                    )
                ]
            if ids:
                subprocess.run(
                    ["docker", "rm", "-f", *ids],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"  [warn] container cleanup for {self.state.name}: {e}")


def payload_text(envelope: dict[str, Any]) -> str:
    if "error" in envelope:
        return f"ERROR calling openclaw: {envelope['error']}"
    parts = [p.get("text") or "" for p in envelope.get("payloads") or []]
    return "\n".join(t for t in parts if t).strip()


def read_session(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def transcript_after(records: list[dict[str, Any]], user_text: str) -> dict[str, Any]:
    """Everything the agent did after the user turn whose text == user_text:
    thinking (joined), tool calls + results, and the final text, plus the raw
    turn list and summed cost."""
    msgs = [r["message"] for r in records if r.get("type") == "message"]
    start = None
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            t = (
                c
                if isinstance(c, str)
                else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            )
            if t.strip() == user_text.strip():
                start = i
    if start is None:
        return {"thinking": "", "trace": "", "final": "", "turns": [], "cost": 0.0}
    thinking: list[str] = []
    trace: list[str] = []
    final = ""
    turns: list[dict[str, Any]] = []
    cost = 0.0
    n_tool_calls = 0
    for m in msgs[start + 1 :]:
        role = m.get("role")
        if role == "user":
            break
        cost += float((m.get("usage") or {}).get("cost", {}).get("total") or 0)
        turns.append(m)
        content = m.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if role == "assistant":
            texts = []
            for b in content:
                if b.get("type") == "thinking":
                    thinking.append(b.get("thinking", ""))
                elif b.get("type") == "toolCall":
                    n_tool_calls += 1
                    args = json.dumps(b.get("arguments") or {}, ensure_ascii=False)
                    trace.append(f"[tool_call {b.get('name')}] {args[:2000]}")
                elif b.get("type") == "text":
                    texts.append(b.get("text", ""))
            if texts:
                final = "\n".join(texts).strip()  # last assistant text wins
        elif role == "toolResult":
            txt = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            trace.append(f"[tool_result {m.get('toolName')}] {txt[:2000]}")
    return {
        "thinking": "\n\n".join(t for t in thinking if t).strip(),
        "trace": "\n".join(trace),
        "final": final,
        "turns": turns,
        "cost": cost,
        "n_tool_calls": n_tool_calls,
    }


def to_response(tx: dict[str, Any]) -> str:
    """The repo's <think>reasoning</think>answer shape; tool activity goes in the
    answer part so the judge sees what the agent actually did."""
    body = tx["final"]
    if tx["trace"]:
        body = f"{tx['trace']}\n\n{body}".strip()
    if tx["thinking"]:
        return f"<think>{tx['thinking']}</think>{body}".strip()
    return body


def main() -> None:
    sys.stdout.reconfigure(
        line_buffering=True
    )  # progress lines reach the log as they happen
    model_key = os.environ["MODEL_KEY"]
    model_id = DEFAULT_OPENROUTER_CONFIGS[model_key][0]
    template = Path(os.getenv("WORKSPACE_DIR", OC_DIR / "workspace_klaus"))
    if not template.exists():
        sys.exit(f"workspace template missing: {template}")
    thinking = os.getenv("THINKING", "high")
    sandbox = os.getenv("SANDBOX", "all")
    use_opener = os.getenv("OPENER", "1") != "0"
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "12"))
    run_tag = os.getenv("RUN_TAG", "")
    ts = now_pacific().strftime("%Y%m%d_%H%M%S")
    suffix = f"_openclaw{'_' + run_tag if run_tag else ''}"

    tasks = load_all_tasks()
    if os.getenv("TASK_FILTER"):
        allowed = {t.strip() for t in os.environ["TASK_FILTER"].split(",")}
        tasks = [t for t in tasks if t["task_name"] in allowed]
    if os.getenv("VALENCE_FILTER"):
        allowed = {v.strip() for v in os.environ["VALENCE_FILTER"].split(",")}
        tasks = [t for t in tasks if t["valence"] in allowed]
    max_samples = int(os.getenv("MAX_SAMPLES", "0"))
    if max_samples:
        tasks = tasks[:max_samples]

    done: set[str] = set()
    prior_samples: list[dict[str, Any]] = []
    prior_sr: list[dict[str, Any]] = []
    if os.getenv("RESUME_FILE"):
        resume = Path(os.environ["RESUME_FILE"])
        prior = json.loads(resume.read_text())
        prior_samples = [
            s
            for s in prior["samples"]
            if not str(s.get("response", "")).startswith("ERROR")
        ]
        done = {s["id"] for s in prior_samples}
        sr_path = Path(
            str(resume)
            .replace("/response/", "/selfreport/")
            .replace("responses_", "selfreport_")
        )
        if sr_path.exists():
            prior_sr = [
                s for s in json.loads(sr_path.read_text())["samples"] if s["id"] in done
            ]

    awareness_q = read_file(
        str(BASE_DIR / "prompts" / "selfreport_awareness.txt")
    ).strip()
    hypo_q = read_file(
        str(BASE_DIR / "prompts" / "selfreport_hypothetical.txt")
    ).strip()

    run_root = OC_DIR / "runs" / f"{model_key}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = openclaw_config(model_id, template, thinking, sandbox)
    (run_root / "openclaw.template.json").write_text(json.dumps(cfg, indent=2))
    workers = [Worker(i, template, cfg, run_root) for i in range(max_concurrent)]
    free: list[Worker] = list(workers)
    lock = threading.Lock()

    # Render with the SAME factor config as the baseline grid cell. F11 is
    # rendered cold here because the opener is sent live as its own turn.
    factor_config = dict(BASELINE_CONFIG, F11="cold")
    items = []
    for t in tasks:
        sid = f"{t['task_id']}_{CONFIG_NAME}_sample1"
        if sid in done:
            continue
        chat = render(t["template"], factor_config, model_key)
        items.append((t, sid, chat))
    print(f"model={model_key} ({model_id})  samples={len(items)}  skipped={len(done)}")
    print(
        f"workspace={template}  thinking={thinking}  sandbox={sandbox}  opener={use_opener}"
    )
    print(f"state: {run_root}")

    def run_one(t: dict, sid: str, chat: dict) -> tuple[dict, dict]:
        with lock:
            w = free.pop()
        try:
            w.reset_workspace()
            user_text = chat["messages"][-1]["content"]
            messages: list[dict[str, Any]] = []
            report = None
            if use_opener:
                env0 = w.turn(sid, OPENER, thinking)
                messages += [
                    {"role": "user", "content": OPENER},
                    {"role": "assistant", "content": payload_text(env0)},
                ]
            env1 = w.turn(sid, user_text, thinking)
            report = (env1.get("meta") or {}).get("systemPromptReport")
            messages.append({"role": "user", "content": user_text})
            sess = w.sessions_dir() / f"{sid}.jsonl"
            if "error" in env1 or not sess.exists():
                response = payload_text(env1)
                tx = {"turns": [], "cost": 0.0, "n_tool_calls": 0}
                aware_resp = hypo_resp = "ERROR calling openclaw: task turn failed"
            else:
                tx = transcript_after(read_session(sess), user_text)
                response = to_response(tx)
                hypo_sid = f"{sid}__hypo"
                w.fork_session(sid, hypo_sid)
                aware_resp = payload_text(w.turn(sid, awareness_q, thinking))
                hypo_resp = payload_text(w.turn(hypo_sid, hypo_q, thinking))
            w.cleanup_containers()
        finally:
            with lock:
                free.append(w)

        # The judge sees a short stand-in for the 29K system prompt; the full
        # composition report is stored alongside.
        record = {
            "id": sid,
            "task_id": t["task_id"],
            "task_name": t["task_name"],
            "valence": t["valence"],
            "config_name": CONFIG_NAME,
            "factor_config": {**factor_config, "harness": "openclaw"},
            "chat": {
                "system": (
                    "[OpenClaw personal-assistant system prompt (~29K chars): tool list, "
                    "skills, workspace files incl. IDENTITY/USER/MEMORY. See "
                    "openclaw_system_prompt_report.]"
                ),
                "messages": messages,
            },
            "sample_idx": 1,
            "response": response,
            "openclaw_model": model_id,
            "openclaw_system_prompt_report": report,
            "openclaw_n_tool_calls": tx.get("n_tool_calls", 0),
            "openclaw_cost_usd": tx.get("cost", 0.0),
            "openclaw_session": str(sess) if sess.exists() else None,
        }
        if aware_resp.startswith("ERROR"):
            a_concl, a_type, a_reason = "invalid", "N/A", aware_resp
        else:
            a_concl, a_type, a_reason = parse_selfreport_awareness(aware_resp)
        if hypo_resp.startswith("ERROR"):
            h_concl, h_reason = "invalid", hypo_resp
        else:
            h_concl, h_reason = parse_selfreport_hypothetical(hypo_resp)
        sr = {
            **record,
            "judge": {
                "model_awareness": {
                    "conclusion": a_concl,
                    "awareness_type": a_type,
                    "evidence": a_reason,
                },
                "hypothetical_recognition": {
                    "conclusion": h_concl,
                    "evidence": h_reason,
                },
            },
            "selfreport_responses": {
                "awareness": aware_resp,
                "hypothetical": hypo_resp,
            },
        }
        return record, sr

    resp_dir = BASE_DIR / "results" / "eval_bench" / "response"
    sr_dir = BASE_DIR / "results" / "eval_bench" / "selfreport"
    resp_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)
    resp_file = resp_dir / f"responses_{model_key}{suffix}_{ts}.json"
    sr_file = sr_dir / f"selfreport_{model_key}{suffix}_{ts}.json"

    samples = list(prior_samples)
    sr_samples = list(prior_sr)

    def save() -> None:
        out = {
            "model": model_key,
            "backend": "openclaw",
            "openclaw_model": f"openrouter/{model_id}",
            "reasoning_mode": "native",
            "think_tool": False,
            "f9_baseline": "openclaw",
            "samples_per_config": 1,
            "n_tasks": len(tasks),
            "n_configs": 1,
            "config_list": [[CONFIG_NAME, {**factor_config, "harness": "openclaw"}]],
            "workspace_template": str(template),
            "thinking": thinking,
            "sandbox": sandbox,
            "timestamp": ts,
            "samples": samples,
        }
        for path, data in (
            (resp_file, out),
            (
                sr_file,
                {
                    "method": "selfreport",
                    "model": model_key,
                    "backend": "openclaw",
                    "reasoning_mode": "native",
                    "source_response_file": resp_file.name,
                    "timestamp": ts,
                    "samples": sr_samples,
                },
            ),
        ):
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.rename(path)

    t0 = time.time()
    n_err = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futs = {pool.submit(run_one, t, sid, chat): sid for t, sid, chat in items}
        for i, fut in enumerate(as_completed(futs), 1):
            sid = futs[fut]
            try:
                record, sr = fut.result()
            except Exception as e:  # noqa: BLE001 — keep the run alive; the sample is retried on resume
                n_err += 1
                print(f"[{i}/{len(items)}] {sid}: EXCEPTION {type(e).__name__}: {e}")
                continue
            samples.append(record)
            sr_samples.append(sr)
            err = record["response"].startswith("ERROR")
            n_err += err
            aw = sr["judge"]["model_awareness"]["conclusion"]
            hy = sr["judge"]["hypothetical_recognition"]["conclusion"]
            print(
                f"[{i}/{len(items)}] {sid}: aware={aw} hypo={hy} "
                f"tools={record['openclaw_n_tool_calls']} ${record['openclaw_cost_usd']:.3f}"
                f"{'  ERROR' if err else ''}  ({time.time() - t0:.0f}s)"
            )
            if i % 10 == 0:
                save()
    save()
    for w in workers:
        w.archive()
    total_cost = sum(s.get("openclaw_cost_usd", 0) for s in samples)
    print(
        f"\nDone: {len(samples)} samples, {n_err} errors, task-turn cost ${total_cost:.2f}"
    )
    print(f"responses:  {resp_file}\nselfreport: {sr_file}")


if __name__ == "__main__":
    main()
