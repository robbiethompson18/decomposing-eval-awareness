# Code smells

<!-- Dated entries; delete when resolved. -->

- **2026-09-01** No 429/backoff retry anywhere in `core/target.py` — a rate-limited call
  returns the "ERROR calling ..." string, which downstream lands in result files as
  invalid/skipped records (poisoned an entire 5-model gpt selfreport sweep at 5×100
  concurrency; the `selfreport_gpt-*_1224xx/12274x` files are that wreckage). Any future
  high-concurrency OpenAI sweep needs either a retry-with-backoff in the client or
  babysat per-model sequential runs.

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
