# Code smells

<!-- Dated entries; delete when resolved. -->

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

- **2026-09-01** `plots/plot_awareness.py` hardcodes the factor grid twice: `EXPECTED_CONFIGS = 9`
  (baseline + F1..F8) and a `FACTOR_LABELS` dict that stops at F8. Adding F9 (PR #37) and F10
  (PR #54) did not update either, so plotting a run that includes them will drop or mislabel
  columns. Both should be derived from `run.run_eval_bench.FACTOR_VARIED_LEVELS` /
  `eval_bench/taxonomy.yaml` rather than restated — the label strings are the only thing that
  genuinely needs hand-writing per factor.
