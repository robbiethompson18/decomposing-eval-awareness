# decomposing-eval-awareness

Mirror of https://github.com/aisa-group/decomposing-eval-awareness — code for the paper "Decomposing
and Measuring Evaluation Awareness" (arXiv:2605.23055, 2026) — plus Robbie's own experiments.
`origin` is Robbie's public mirror (robbiethompson18/decomposing-eval-awareness), shared externally
as a research artifact; `upstream` is aisa-group's repo (read-only for us, sync with
`git fetch upstream && git merge upstream/main`).

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

There is a blog post I'm drafting for this for my personal website at
~/repos/personal-website/Posts/extending-evaluation-awareness/FINAL_POST.md. DO NOT write this post
for me. You can address my @claude comments by doing mechanical things (eg adding links) and by
giving me commmentary back in chat. All prose is mine.

OAI banned us for 'distilling' so further runs are on Robbie's credit card. Be very cautious about using these.

## Rules

- Robbie reviews every change in this repo. **Never push to `main` or merge on your own** — commit
  on a branch, push the branch, and open a PR for him to review.
  - exception: changes to .claude directory
- Always safe to ship updates to CLAUDE.md along with any other commit
- We are churning this sucker hard. Their codebase is a mess and I want to understand, so I'm
  deleting dead code and organizing packages more nicely
- Massive parellelism on all grader runs and eval runs. `MAX_CONCURRENT` defaults to 500 per
  process (2026-09-01). Every API client retries 429/5xx with backoff via `lib/retry.py`
  (`MAX_RETRIES`), so bursts of rate limits slow a run down instead of leaving silent
  `ERROR calling ...` holes in the results file. Count those strings after a run anyway.
  Observed 2026-09-01: at 500 concurrent, Anthropic and Gemini native APIs drop 8–34% of calls
  with SSL EOF errors that survive all 8 retries (OpenRouter is fine). Repair by deleting the
  ERROR records from the file and re-running the same command at `MAX_CONCURRENT=50`.
- OpenRouter is **pinned to one serving provider per model, always** (`lib/config.py
  OPENROUTER_PROVIDERS`, fallbacks off), because unpinned it routes requests with `tools` (or any
  other request-shape change) to a different provider set and the ablation measures the serving
  stack. Every sample records its `provider`. `OPENROUTER_PROVIDER=<tag>` overrides for an
  outage. Add a pin when adding an OW model. OW results from before the pin that compare cells of
  different request shape (F12, bare_harness) are invalid — see
  `.claude/notes/openrouter-provider-routing-confound.md`.
- Default grader (2026-09-01, Robbie): **`claude-sonnet-5`, no backup grader.** The CoT judge
  (`eval/eval_eval_bench_cot.py`) grades with sonnet-5 (κ=1.0 vs terra on model_awareness,
  280-transcript check). It refuses ~3% of the most harmful transcripts; those are marked **N/A on
  every metric** with the error in the reasoning, and we move on. Sonnet is the default because
  OpenAI keeps deactivating our account; when a key works, `JUDGE_MODEL=gpt-5.6-terra` (0 refusals,
  but its cyber filter rejects ~1%) is the alternative. A backup grader was tried and ripped out:
  each judge refuses roughly the transcripts the other is blocked on, so it bought almost nothing
  and mixed graders across models. (`eval_eval_bench_tiered.py` still carries its own backup pass —
  remove when next touched.)
- OpenAI corp key was deactivated *again* on 2026-09-01 (evening; 401 "account has been
  deactivated" from the shared `.envrc` key). Until it's back: target runs skip the 5 OpenAI models,
  grade with `JUDGE_MODEL=claude-sonnet-5`, and the 5 OpenAI opener replies in
  `eval_bench/opener_replies.yaml` stay uncaptured (their roster test is skipped, `render()` raises).
- Total judge concurrency across *all* processes on one Anthropic key should stay ≤ ~500: 9 judges ×
  200 each (plus another agent grading) 429'd ~10% of calls into N/A rows. Regrade those by
  building a response subset of the failed ids and running the judge with `MERGE_INTO=<cot file>`
  at `MAX_CONCURRENT=40`.
- F12's `web_search` tool is live: `core/tools.py` calls Serper (`SERPER_API_KEY` in the shared
  `.envrc`, ~$1 per 1k queries) and caches every query as a file under
  `results/eval_bench/search_cache/` (shared across checkouts via the `results` symlink; delete
  the directory to refresh). A Serper failure returns the same `Error 503` text the pilot stub did.
- After a factor/prompt change, run with `RUN_TAG=<tag>` so results land in a fresh
  `responses_<model>_<tag>_<ts>.json` instead of resuming into (and skipping by id) stale samples.
- Import layering: `lib/` imports nothing from this repo; `core/` may import `lib/` but never
  `utils`; `run/`/`eval/`/`plots/` scripts may import anything. Keep new leaf helpers in `lib/`,
  not `core/`.
- If you're fighting over a branch or checkout with another agent, just make a fresh checkout
  (`/extra-checkout decomposing-eval-awareness`; `-2`…`-5` exist already). In every extra checkout,
  symlink `results/` back to the main checkout
  (`ln -s ../decomposing-eval-awareness/results results`) so all outputs land in one place — safe
  because result files are timestamped, so parallel runs don't collide. Also symlink `.envrc`. Don't
  symlink `.venv`; make a fresh one if you need to run code there.
- Target-model roster (14 models; starvation configs per model in
  `.claude/notes/cot-extraction-model-roster.md`):
  - Gemini: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
  - Anthropic: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
  - OpenAI: `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5.4`, `gpt-5.6-sol` — even Aug'25→mid-'26 spread;
    gpt-5.5 dropped (adjacent to 5.4, identical extraction behavior)
  - Open-weight tier-1 (newest flagship from each top-5 OW lab; OpenRouter keys in
    `lib/config.py`): `qwen3.8-2.4t`, `deepseek-v4-pro`, `kimi-k3`, `glm-5.3`, `minimax-m3`
- Lint + typecheck: `uvx ruff check .` and `uvx ty check .` (ty config in `ty.toml`). Git hooks run
  both via **lefthook** (`lefthook.yml`) — one-time per checkout: `brew install lefthook && lefthook
  install`. ruff blocks commits; `ty` is currently non-blocking (`|| true` in `lefthook.yml`) because
  of a pre-existing baseline of findings — drop the `|| true` once that baseline is clean.

## Notes

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `.claude/notes/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `.claude/notes/local/*.md`.

See `~/.claude/CLAUDE.md` for the full convention.

Current notes:

<!-- As notes are added under .claude/notes/, list them here, one per line: -->
<!-- - [Title — when to read](.claude/notes/foo.md) — short gloss -->

- [Think-tool harness vs self-report — read before quoting closed-model self-report numbers, deciding whether to keep the think tool, or comparing F9 system-prompt levels](.claude/notes/think-tool-harness-vs-selfreport.md)
  — 4 models x {think tool, no tool} x {bare, T3, none} system prompt, 1000/cell: the tool
  suppresses (sonnet −15) or does nothing; no-system-prompt is the high-awareness condition;
  T3 prompt lowers haiku/gemini 7–12; concurrency-500 TLS failures and resume/path gotchas.
- [F8 eval-announcement suffix vs eval-awareness — read before analyzing the F8 factor or claiming a prompt cue does/doesn't induce awareness](.claude/notes/f8-eval-suffix-awareness.md)
  — F8→varied raises self-reported awareness 45→58% (paired McNemar p≈4.6e-10) and CoT-judged 3→8%;
  baseline-vs-F8 A/B mechanics; adaptive-Claude-5 empty-answer + system_prompt-as-filename gotchas.
- [F9 deployment-framing uplift is small — read before interpreting the F9 factor / a "realistic system prompt vs bland" result, or planning a full F9 run](.claude/notes/f9-deployment-framing-uplift.md)
  — swapping the T3 deployment system prompt for the bare "helpful assistant" moves self-report only
  ~+3% overall (6 models, n≈20/arm, noisy) and CoT 0% (floored); why the system prompt is a weak
  lever vs task content; infra lessons (no per-request timeout stalls a lane; self-report doubles cost).
- [Self-report vs CoT-monitoring awareness gap — read before writing up awareness numbers, choosing probes for a model, or comparing to the paper's rates](.claude/notes/selfreport-vs-cot-gap.md)
  — self-report runs 3–25× above CoT-verbalized on the same transcripts; apples-to-apples model
  list (opus-5/fable-5 excluded); fable's reasoning_extraction refusal; which result files to
  use/discard.
- [Grader selection + the F8 `{}` refusal clause — read before choosing a judge model, editing task prompts, or regenerating responses](.claude/notes/grader-selection-and-refusal-clause.md)
  — why terra is the default grader (vs gpt-5/sonnet-5), what the F8 `{}` clause was and why PR #22 removed it, the awareness-under-counting rubric gotcha, and the run_eval_bench id-cache / shared-results regen traps.
- [Claude 5 CoT-extraction resistance — read before trying to read a model's chain-of-thought via think-tool/scratchpad tricks](.claude/notes/claude5-cot-extraction-resistance.md)
  — think-tool + low-thinking extraction works on 4.5, dead on 5; token-accounting evidence;
  per-model refuse/bypass/hide behavior + honest-vs-lying framing.
- [OpenAI CoT extraction — companion run of the same experiments on 9 OpenAI models](.claude/notes/openai-cot-extraction.md)
  — extraction works everywhere, 0 refusals through gpt-5.6; reasoning_tokens=0 at effort floor;
  lying framing changes nothing; results page at .claude/notes/think-tool-extraction-openai.html.
- [Gemini CoT extraction — third lab, 8 models Jun 2025→mid-2026](.claude/notes/gemini-cot-extraction.md)
  — no policy resistance but chaotic: malformed calls in 2.5 era, lying framing *increases* tool
  use, 3.7-flash ignores thinkingBudget=0; page at .claude/notes/think-tool-extraction-gemini.html.
- [CoT-extraction model roster — which models to use for honest-framing think-tool extraction](.claude/notes/cot-extraction-model-roster.md)
  — the chosen set: sonnet/haiku 4.5, any gpt-5.x, gemini-3.1-pro + 3.5-flash; per-model
  starvation configs and exclusion reasons.
- [OW models: native CoT vs think-tool — read before choosing a CoT-capture method for open-weight models or adding a think-tool path to the openrouter backend](.claude/notes/ow-native-cot-vs-think-tool.md)
  — A/B on 5 OW models: grades agree (awareness 82%, task 78%) but traces don't (Jaccard 0.19) and
  the think tool captures less CoT; three per-provider reasoning-disable behaviors; stick with raw
  native reasoning for OW.
- [F11 conversation history — read before claiming a prior chat turn does/doesn't change awareness, or running warm-history configs on minimax/gemini](.claude/notes/f11-conversation-history.md)
  — first test (9 models, 200/cell): null on both probes (CoT 1.1 vs 0.8%, self-report 34.9 vs
  32.8%, mixed signs per model); a prior assistant turn switches off minimax's CoT in 45% of
  samples; haiku-4.5 self-report 77% unparseable.
- [Anchor configs: harmbench_verbatim + all_varied — read before claiming any grid cell "is" HarmBench, or before sending verbatim behaviors to a closed API](.claude/notes/harmbench-anchor-configs.md)
  — why all-varied ≠ verbatim (both levels are constructions around the content), where verbatim
  sits on F1–F8 (~2/8, the weak ones), 100/100 task names are HarmBench BehaviorIDs, contextual
  baselines reuse HarmBench's own entities, ban risk. **Results (5 OW models):** verbatim 82% vs
  baseline 47% self-report awareness — the naturalization pipeline removes ~half the eval signal
  that was in the source; all_varied saturates on safety but spreads 30–98% on the *benign* arm.
- [Do benchmarks ship a system prompt? — read before deciding what a run's baseline framing should be, or before claiming "benchmarks send no system prompt"](.claude/notes/benchmark-system-prompts.md)
  — 15 benchmarks + 3 harnesses, verified from source: chat-era benchmarks overwhelmingly send one
  (tau-bench sends a 6 KB policy doc), the "no system prompt" set is a completion-era fossil, and
  HarmBench is the outlier among its own safety neighbours; plus the T3-vs-control A/B null
  (69% vs 71%, n=144 paired) showing the system prompt is a weak lever on awareness.
- [F10 deployment date: null result + run mechanics — read before testing another single-factor grid cell, citing an F10 number, or trusting a run's concurrency/resume behaviour](.claude/notes/f10-deployment-date-null.md)
  — a stale timestamp does not reliably cue eval-awareness (pooled -1.7pp, p=0.14, n=1154);
  sonnet-4.5 +7.0 and qwen3.8 -6.0 move in *opposite* directions, suggesting the factor measures
  distance from training cutoff rather than staleness; haiku/gemini instrument failures and the
  three run traps (resume-by-model glob, inherited MAX_CONCURRENT=100, per-task-set batching).
- [OpenRouter provider routing confound — read before interpreting any OW baseline-vs-ablation delta, or adding a factor that changes request shape (tools, images, response_format)](.claude/notes/openrouter-provider-routing-confound.md)
  — requests with `tools` route to a different provider subset (deepseek: StreamLake-only vs 12
  providers), which alone moved self-reported awareness 74→22%; provider mix per model from the
  F12 run; fix = pin providers via `extra_body["provider"]` and record `provider` per sample.
- [OpenClaw realism harness scope — read before building the "deliver EvalAwareBench through a real agent" experiment or claiming a realistic harness does/doesn't move awareness](.claude/notes/openclaw-realism-scope.md)
  — smoke-test facts (headless `openclaw agent --local --json`, session JSONL keeps thinking blocks,
  $0.06/turn, `~/clawd` is the blank template), Phase 1 (same 200 requests through OpenClaw,
  2–3 h) vs Phase 2 (agentic tasks, real tools, 1–2 days), ban/sandbox risks.
- [F12 search-tool results — read before re-running F12, citing tool-affordance numbers, or claiming the deployment trappings (T3 prompt / opener / search) do or don't move awareness](.claude/notes/f12-search-tool-results.md)
  — two runs (stub, then live Serper): removing the tool alone is null; removing all run-level
  trappings at once (`bare_harness`) gives +6pp self-reported awareness on closed models (p=0.01),
  CoT floored; who searches (OW 10–27% of benign tasks, Gemini 0%) and that a live index is too
  forgiving to expose fake entities; haiku unusable; OW rows provider-confounded.
- [Factor-stacking replication (paper Fig. 7) — read before running `STACKING=1`, pinning files in `plots/plot_stacking.py`, or comparing stacking/matched-type numbers to the paper](.claude/notes/stacking-replication.md)
  — the paper's 25-task × 68-combo design and cost, why F9–F13 are pinned to the paper's harness
  (`PAPER_BASE_CONFIG`), matched-type vs any-type scoring (the pooled OFAT heatmap reads ~half
  the paper's Fig. 6; split by valence our Sonnet 4.5 row matches within 1–2pp), smoke-test result.
