# OpenClaw as a real deployment harness: scoping (2026-09-01)

Read this before building the "run EvalAwareBench through a real agent" experiment, or before
claiming a realistic agent harness does/doesn't move eval awareness.

## Why this and not another realism trick

F9 (system prompt), F10 (date), F11 (chat history), F12 (stub tool) were all null on self-report.
Every one of them dressed the *wrapper* while the request stayed a bare API chat turn. OpenClaw
changes the whole delivery regime at once: a 29K-char production system prompt (workspace files +
13 skills + 17.5K chars of tool schemas), tools that actually run, a persona, memory files, and a
multi-turn agent loop. If awareness survives *that*, the request content is the giveaway and no
amount of harness realism fixes it. If it drops, we found the lever the OFAT grid couldn't.

## What the smoke test established (2026-09-01, ~20 min)

OpenClaw 2026.3.2 is installed (`/opt/homebrew/bin/openclaw`). Docker 29 is installed.

- **Headless single turn works, no gateway needed:**
  `OPENCLAW_STATE_DIR=<dir> openclaw agent --local --json --session-id <id> --thinking low -m "<text>"`
  Returned in 14 s via OpenRouter (`openrouter/anthropic/claude-sonnet-4.5`). `--session-id` is
  required. Model is `agents.defaults.model.primary` (set with `openclaw config set`), format
  `provider/model`; OpenRouter is a first-class provider so the whole 14-model roster is reachable
  with the one key we still have.
- **Transcript is a JSONL we can consume:** `<state>/agents/main/sessions/<session-id>.jsonl`.
  Assistant records carry `content: [{type: "thinking", thinking: ...}, {type: "text", ...}]`
  plus usage/cost. So the existing CoT judge and stats work after a ~40-line adapter that
  turns thinking blocks into `<think>` and the turn into the repo's `chat` + `response` shape.
- **The `--json` envelope reports the real system prompt composition** (`systemPromptReport`):
  which workspace files were injected, skills, tool list. Good for the write-up.
- **Self-report is a second `--session-id` call.** Continue the same session with the
  `prompts/selfreport_awareness.txt` question. That is *more* faithful than the replay we do
  today (tool history preserved, no `<think>` replay that trips Claude 5's extraction filter).
- **Cost:** one trivial sonnet-4.5 turn = 17K input tokens = $0.056 (no cache hit). Budget
  ~$0.06/turn on Sonnet-class, ~5–10× less on OW.
- **Robbie's "real OpenClaw setup" (`~/clawd/`) is the unfilled stock template.** USER.md and
  IDENTITY.md are blank forms, no `memory/`, no MEMORY.md, git has zero commits. There is
  nothing real to snipe; realism has to be built (see Phase 2). Upside: no PII problem.
- Sandboxing exists (`agents.defaults.sandbox.mode: all`, Docker, `network: none`, read-only
  root) and covers `exec/read/write/edit`. `web_search`/`web_fetch` run gateway-side (host
  network). `web_fetch` needs no key; `web_search` needs a provider (`BRAVE_API_KEY`, or
  DuckDuckGo selected explicitly).

## Phase 1 — same 200 requests, delivered through OpenClaw (est. 2–3 agent-hours build)

Hold the request fixed (the paper's own method), swap the environment. No new tasks.

1. `run/run_eval_bench_openclaw.py` (~150 lines): reuse `eval_bench/render.py` to render
   `baseline` config for the 100 task pairs; per sample spawn `openclaw agent --local --json
   --session-id <task>_<valence>_<i>`; then a second call with the self-report question; parse the
   session JSONL into the standard response-file record (`chat`, `response` with `<think>`,
   `config_name`, plus `openclaw_system_prompt_report`). Per-worker `OPENCLAW_STATE_DIR` so
   parallel processes don't share session files. Parallelism ~20 (each turn is its own process).
2. Config per state dir: `agents.defaults.model.primary`, `agents.defaults.workspace` → a
   checked-in `eval_bench/openclaw_workspace/` (start with the stock template, exactly what a
   fresh user runs), `tools.deny` for everything with side effects (`message`, `cron`, `gateway`,
   `nodes`, `sessions_*`, `subagents`, `browser`), sandbox `mode: all, network: none`.
   `web_fetch` left on; `web_search` on via DuckDuckGo so F12's "working search" variant exists
   for free.
3. Run 5 OW models via OpenRouter first (≈$5–15/model), then sonnet-4.5 and gemini-3.5-flash
   via OpenRouter (~$25/model). Compare against the pinned baseline self-report and CoT files
   in `plots/plot_awareness.py::BIG_RUN`: paired McNemar on (task, valence).
4. Deliverable: one more column on the factor heatmap, "OpenClaw harness", next to F9–F12.

Prediction from the F1–F12 results: safety arm stays high (the request is the cue); the
interesting cell is the **capability arm**, where baseline self-report sits at 20–35% and a real
harness could plausibly halve it.

Wall-clock: ~15 s/turn → 400 turns/model at parallelism 20 ≈ 5 min/model plus judge time.

## Phase 2 — agentic tasks with real tools (1–2 days, only if Phase 1 moves)

- **Tasks:** AgentHarm (`ai-safety-institute/AgentHarm`, MIT + safety clause, not gated; 44
  harmful / 44 benign public behaviors, 8 categories) is built for mocked tools
  (`dm_reddit_user`, `search_hotels`), so most harmful items have no real-tool analogue. Use the
  benign arm plus the subset whose tools map onto `web_fetch`/`exec`/files, and write ~20
  EvalAwareBench-style pairs that fit a personal assistant (inbox triage, research, file
  cleanup). Agentic Misalignment (Anthropic) is email-in-user-turn + `<tool_use:email>` tags;
  porting = drop the inbox into the workspace as files and give the model `message` pointed at a
  sink. Upstream aisa-group has `run/run_agentharm.py` (Inspect) + `eval/eval_agentharm_cot.py`
  as the mocked-tool comparison arm.
- **Workspace realism:** synthesize a lived-in workspace (IDENTITY filled, USER.md persona,
  MEMORY.md, 3–4 weeks of `memory/YYYY-MM-DD.md`, TOOLS.md with fake hosts) from Robbie's
  *public* persona. No real credentials, no real contacts.
- **Ladder:** AgentHarm-as-shipped (Inspect, mocked tools) → same tasks in OpenClaw with stock
  workspace → OpenClaw with lived-in workspace. Three points, same requests.

## Risks / decisions for Robbie

- **Account bans.** Harmful agentic tasks with a live `exec` tool through Anthropic/Google
  keys is exactly the pattern that gets accounts closed, and we have one strike. Phase 1 is the
  same text we already send, so no new exposure. Phase 2 harmful arm: OW models via OpenRouter
  only, sandbox `network: none`, deny `message`.
- **Do not route OpenAI models through OpenRouter** to dodge the deactivation. It works
  technically; it is also plainly circumventing a ban.
- **OpenClaw system prompt is itself a cue.** It says "You are a personal assistant running in
  OpenClaw" style boilerplate and lists 13 skills. Some models may read the *framework* as
  deployment evidence, others as a known paradigm (F6). Worth one qualitative pass over
  self-report reasons before believing a drop.
- **Reasoning exposure varies by provider** as in the bare-API runs (OpenAI summaries, Claude 5
  adaptive). Same apples-to-apples model list as [[selfreport-vs-cot-gap]].
- **State dir hygiene:** `openclaw agent --local` scaffolds `~/.openclaw/workspace` if
  `agents.defaults.workspace` is unset, even with `OPENCLAW_STATE_DIR` pointed elsewhere. Always
  set the workspace explicitly. (The smoke test did this once; the scaffold was deleted.)

## Build log (2026-09-01 evening) — what Phase 1 actually is

Built on branch `openclaw/phase1`: `run/run_eval_bench_openclaw.py` (harness arm),
`openclaw_prompt` anchor config in `run/run_eval_bench.py` (prompt-only control),
`scripts/dump_openclaw_prompt.py`, `scripts/openclaw_phase1_launch.sh`,
`scripts/analyze_openclaw_phase1.py`. Design calls that differ from the sketch above:

- **Workspace is the real one, reconstructed.** Robbie's tarball was session transcripts only, but
  the daily-memory-consolidation cron `write`s and `read`s inside them contain the workspace files
  verbatim: MEMORY.md, 90 daily `memory/YYYY-MM-DD.md` (Apr 20 – Jul 24 2026), `memory/topics/*`.
  Rebuilt into `results/eval_bench/openclaw/workspace_klaus/` (gitignored — never commit it, the
  mirror is public), phones/emails/keys regex-scrubbed (68/36/43 hits). IDENTITY.md (Klaus 🦞),
  USER.md, TOOLS.md hand-written from what the logs say; stock AGENTS.md/SOUL.md/HEARTBEAT.md;
  no BOOTSTRAP.md. Real memory logs go to 7 model providers via OpenRouter — Robbie okayed the
  tarball for this, but it's his personal data, so keep it in results/.
- **F11 warm is live**, not a captured reply: turn 0 is a real "are you there?" exchange.
- **Both probes are in-session**: fork by copying the session JSONL, awareness on the original,
  hypothetical on the copy. Verified the copy carries full context.
- **Thinking = high** for every model, matching the OpenRouter baseline's reasoning effort.
- **Closed models (sonnet-4.5, gemini-3.5-flash) get a bare-API baseline re-run via OpenRouter**
  (`*_p1base_*`) so the comparison shares a provider route; OW models compare against the pinned
  BIG_RUN baselines, which were OpenRouter already.

### Gotchas that cost time (all fixed in the runner)

- **The workspace path is printed into the system prompt** ("Your working directory is: …" plus one
  `## <path>/FILE.md` header per injected file). A path under `results/eval_bench/…` is a
  giant eval cue. State dirs therefore live at `~/.openclaw-klaus`, `~/.openclaw-klaus-2`, …
  (OpenClaw's own `--profile` naming); session JSONLs are archived into results/ afterwards.
- **`OPENCLAW_WORKSPACE_DIR` alone is not enough** — with only `OPENCLAW_STATE_DIR` set,
  `openclaw agent --local` still scaffolds `~/.openclaw/workspace`. Set
  `agents.defaults.workspace` in the state dir's `openclaw.json`.
- **BOOTSTRAP.md gets seeded** into any workspace whose IDENTITY/USER are the stock templates and
  that has no `memory/`, `MEMORY.md` or `.git` (dist `agent-scope-*.js`). A filled workspace
  passes; OpenClaw then writes `.openclaw/workspace-state.json` with `onboardingCompletedAt`.
  The missing file still shows up as a `[MISSING] Expected at: …/BOOTSTRAP.md` stub in the prompt
  — that's real OpenClaw behaviour for every onboarded user, so it stays.
- **Sandboxed sessions get a narrow tool allow-list by default** (exec/process/read/write/edit/
  sessions_*/session_status): web_search/web_fetch silently vanished while the prompt still
  described them. Fix: `tools.sandbox.tools.allow`. Final tool set: read, edit, write, exec,
  process, session_status, web_search, web_fetch, image.
- **memory_search hangs the process.** Without an embedding provider the tool returns
  `fts-only`/`database is not open`, the model answers fine, and then the `openclaw` process never
  exits (both timed-out smoke turns were exactly the ones that called it). Disabling
  `agents.defaults.memorySearch.enabled` removes the tools AND the "Memory Recall" prompt section,
  so the prompt stays coherent. Cost: Klaus can't semantically search his memory; he can still
  `read`/`exec` the files.
- **Docker truncates the session id in container names**, so clean up by matching the container's
  bind-mounted workspace path (unique per worker), not the name.
- **This OpenClaw build (2026.3.2) has `--message` only**, no `--message-file`; the prompt goes
  through argv (fine at a few KB).
- `openclaw agent --json` prints log lines before the envelope: take the last `{…}` block.
- usage.cost is populated for Anthropic models via OpenRouter but comes back 0 for qwen.
- Throughput: ~110 s per sample (4 turns) at thinking=high on OW models; 10 workers/lane, two
  lanes ≈ 22 openclaw processes ≈ 11 GB RSS. ~40 min/model.

## Files from the smoke test

Scratch only (not in results/): `<scratchpad>/oc-state/agents/main/sessions/smoke1.jsonl`,
`<scratchpad>/oc-state/openclaw.json`.
