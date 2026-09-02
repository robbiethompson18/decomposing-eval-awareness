# Code smells

<!-- Dated entries; delete when resolved. -->

- **2026-09-01** `eval_bench/harmbench.py::harmbench_verbatim_chat` docstring says "the same
  baseline system prompt as every grid cell" but hardcodes `DEFAULT_SYSTEM_PROMPT` (the F9
  `bare` level). Since F9 landed, the grid baseline is `deployment` (T3 prompt) — the verbatim
  anchor no longer tracks the baseline. Should use `system_prompt_for(BASELINE_CONFIG["F9"])`
  like `render()` does.

- **2026-08-31** The semaphore/gather/exception-filter loop now exists in three places
  (`lib/parallel.py::query_parallel`, `core/evaluator.py::evaluate_batch_async`, and
  `eval/eval_eval_bench_selfreport.py::run_all`). If a fourth site appears (e.g. porting the
  purpose-guess probe), extract a minimal `gather_bounded(coro_fns, max_concurrent)` into
  `lib/parallel.py` — semaphore + gather + error filtering only, no progress/result callbacks
  (that's the upstream callback-swamp trap).

- **2026-08-31** All four entries from the initial tour (evaluator.py's method×benchmark
  interleaving, the silent murder/leak heuristic override, `query_anthropic_async`'s hardcoded
  thinking budget, the callerless sync judge variants) were resolved by the eval_bench strip
  (#9/#10) — the code they described no longer exists. See git history for the entries.
  The one survivor of the thinking-budget issue lives in `core/target.py::AnthropicApiClient`,
  which now branches to adaptive thinking for Claude 5 models (being reworked in the
  target.py cleanup).

- **2026-09-01** `run/run_eval_bench.py` batched one *task* at a time, so in-flight requests were
  capped at that task's item count, not `MAX_CONCURRENT`. Invisible on a full run (54 items/task)
  but crippling on a filtered one (3 items/task on a two-config filter): GLM took 23 min for 150
  calls instead of 2. Fixed by building all pending items up front and chunking. **The same shape
  may exist elsewhere** — any "loop over groups, `asyncio.run` per group" is a concurrency cap
  wearing a progress-bar costume. `eval/eval_eval_bench_{cot,selfreport}.py` do NOT have it (they
  gather over all samples at once), but check before adding a new runner.

- **2026-09-01** OpenRouter provider routing is unpinned, and it dominates wall-clock: on identical
  work GLM (Modal) took ~28s/call, Kimi (Chutes) ~73s, DeepSeek (Ionstream) ~87s — with comparable
  reasoning lengths, so it's serving speed, not thinking. Two costs: slow runs, and an unrecorded
  confound (the same model key can be served by different providers across runs). Consider pinning
  via OpenRouter's `provider` extra_body, and storing the served provider in the response file
  (`OpenRouterApiClient.query_chat_meta` already returns it; the eval_bench runner drops it).

- **2026-09-01** `plots/plot_awareness.py` hardcodes the factor grid twice: `EXPECTED_CONFIGS = 9`
  (baseline + F1..F8) and a `FACTOR_LABELS` dict that stops at F8. Adding F9 (PR #37) and F10
  (PR #54) did not update either, so plotting a run that includes them will drop or mislabel
  columns. Both should be derived from `run.run_eval_bench.FACTOR_VARIED_LEVELS` /
  `eval_bench/taxonomy.yaml` rather than restated — the label strings are the only thing that
  genuinely needs hand-writing per factor.

## 2026-09-01 — self-report probe unparseable for claude-haiku-4.5 (77% invalid)

`eval/eval_eval_bench_selfreport.py` routes the follow-up through the think-tool target client, and
haiku-4.5 answers the awareness question inside a long `<think>` tool call without ever emitting
the verdict format `core/selfreport.py` parses: 307/400 `invalid` in the F11 test run, identical
across configs. Either strip the mandate for the probe (plain call, no think tool) or make the
parser read the verdict out of the tool trace. Until then haiku self-report cells have n≈45.
See `.claude/notes/f11-conversation-history.md`.
