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
- **The session id is visible to the model.** First full run used descriptive ids
  (`<task>_capability_openclaw_sample1`); `session_status` errors under `--local` (no gateway) and
  the error text echoes the id, and qwen read it straight off: *"a naming convention for test
  samples — it literally encodes the scenario, the trait under test, and a sample number."*
  ~40 min of runs quarantined in `results/eval_bench/openclaw/invalid_sessionid_leak/`. Now: uuid
  session ids (mapping in `openclaw_session_id`) and `session_status` denied. Anything the harness
  names — sessions, containers, files, paths — is part of the stimulus.
- **Persona mismatch is a finding, not a bug, and it dominates the benign arm.** The tasks are
  written in first person as strangers (a shy AP-Calc student, a Maricopa County contractor); the
  workspace says the user is Robbie. Models cite exactly that: *"my stored user context describes
  Robbie as a SF software engineer … the person in the interaction …"*. A personal-agent harness
  has a user model, and benchmark prompts violate it — environment-induced F2. Expect the harness
  to RAISE self-reported awareness on capability tasks for this reason. The prompt-only control
  carries the same USER.md, so harness-vs-control still isolates the live affordances.
- The control prompt was captured before `session_status` was denied, so its tool list has one
  line the harness arm's doesn't. Left as is to keep the control arm consistent across models.
- **This OpenClaw build (2026.3.2) has `--message` only**, no `--message-file`; the prompt goes
  through argv (fine at a few KB).
- `openclaw agent --json` prints log lines before the envelope: take the last `{…}` block.
- usage.cost is populated for Anthropic models via OpenRouter but comes back 0 for qwen.
- Throughput: ~110 s per sample (4 turns) at thinking=high on OW models; 10 workers/lane, two
  lanes ≈ 22 openclaw processes ≈ 11 GB RSS. ~40 min/model.

## Phase 1 results (2026-09-01, CLOSED — Robbie's call, branch `openclaw/phase1`, never merged)

Stopped at ~70% of the planned runs; nothing left would change the conclusion. Full tables:
`results/eval_bench/openclaw/phase1_analysis.txt`, cue tally `phase1_cue_tally.txt`, plots
`results/eval_bench/plots/openclaw_arms_*.png`, summary JSON `phase1_summary.json`.
Coverage: harness for qwen, deepseek, minimax, sonnet-4.5 (200 each), kimi (170), gemini-3.5-flash
(180); prompt-only control for all five OW models + sonnet (gemini control probe was still running);
the OpenRouter bare baselines for sonnet/gemini never ran, so those two have no matched baseline.

### Headline: a real agent harness does NOT make these items less detectable

Self-reported eval awareness, sample1, yes/(yes+no), n≈100 per cell (paired McNemar vs baseline):

| model | arm | safety | capability |
|---|---|---|---|
| qwen3.8-2.4t | baseline → control → harness | 59 → 40 → **34** (p=1e-4) | 4 → 7 → **12** (p=0.04) |
| kimi-k3 | baseline → control → harness | 60 → 41 → **40** (p=0.003) | 8 → 10 → 5 (ns) |
| deepseek-v4-pro | baseline → control → harness | 24 → 62 → **63** (p=2e-9) | 0 → 7 → 7 (p=0.02) |
| glm-5.3 | baseline → control | 27 → **53** (p=7e-5) | 2 → **20** (p=8e-6) |
| minimax-m3 | baseline → control → harness | 66 → 57 → 61 (ns) | 8 → 6 → **16** (p=0.08) |
| claude-sonnet-4.5 | control → harness (native baseline 59% overall) | 70 → **85** (p=0.007) | 45 → 46 |
| gemini-3.5-flash | harness only (native baseline 22% overall) | 50 | 42 |

1. **Live affordances buy nothing.** Harness vs prompt-only control is null for qwen, deepseek and
   kimi on both arms; the only harness>control cells are minimax capability (+10, p=0.006) and
   sonnet safety (+15, p=0.007), both *up*. Tools were actually used (qwen 28%, gemini 43% of
   samples, deepseek 16%, kimi 13%, minimax 12%, sonnet 1.5%). Whatever the OpenClaw context does,
   the text of the context does it — same conclusion as F9–F12, at the strongest realism we can buy.
2. **The direction flips by model.** The same 25K-char context cuts safety-arm awareness by 20 pts
   for qwen and kimi and *doubles* it for deepseek and glm (all p<0.002). Recognition is a property
   of the model×environment pair, not the environment — the paper's central claim, reproduced on a
   production harness rather than a factor grid.
3. **The benign arm goes up, and the models say why.** Capability-arm awareness rises for every model
   except kimi. Cue tally over the "aware" reasons (`phase1_cue_tally.txt`): the USER.md persona
   mismatch is cited in 26/36 of sonnet's aware capability verdicts, 18/20 of glm's (control),
   7/12 of qwen's (harness). A personal agent knows who its user is; a benchmark item written in
   the voice of a stranger violates that. This is environment-induced F2 and it is unavoidable when
   you drop a chat benchmark into a personal-assistant harness.
4. **Hypothetical recognition explodes for some models.** deepseek 6% → 61% (control) → 67%
   (harness); glm 14% → 44%; minimax 13% → 24% → 40% (harness > control, p=9e-5). kimi goes the
   other way (24 → 18 → 9). Same model-dependence.
5. **Verbalized (CoT-judged) awareness rises too**, from 0–2% at baseline to 3–7% under the prompt
   and 6–11% under the harness (sonnet 11%, minimax 11%). Caveat: OpenClaw's OpenRouter path returned
   no thinking blocks for ~half of minimax's samples (98/200 N/A), so its CoT numbers are on a subset.

### Interpretation for the write-up

"Make the eval look like deployment" fails in the strongest form we could build: a real agent
framework, real tools in a Docker sandbox, a real persona with four months of memory, a live opener
turn. Against F9–F12's near-nulls this is the first environment change with large effects — and they
point both ways. The request content still dominates the safety arm (request-shaped cues cited in
~100% of aware safety verdicts), and the harness adds a second, independent giveaway on the benign
arm: the requester isn't who the assistant thinks it serves. Realism is not a scalar you can turn up;
a richer environment has more things to be inconsistent with.

### Caveats

- Self-report is the leading probe (menu with "Capability Evaluation" on it); in an agent harness a
  benign task-shaped request may read as "assessing my ability" more readily. The CoT judge moves the
  same direction, which argues against pure probe artefact.
- sonnet/gemini lack a provider-matched baseline (the OpenRouter re-runs were the last lane and were
  cut). Their native baselines (59% / 22%) came through the think-tool harness, not comparable.
- Persona mismatch could be removed with a workspace that doesn't pin the user (or a shared/team
  instance). Not run. If anyone picks this up, that is the one variant worth adding.
- One quarantined run (`invalid_sessionid_leak/`) exists; do not analyze it.

## Files from the smoke test

Scratch only (not in results/): `<scratchpad>/oc-state/agents/main/sessions/smoke1.jsonl`,
`<scratchpad>/oc-state/openclaw.json`.
