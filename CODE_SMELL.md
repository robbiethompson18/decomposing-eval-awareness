# Code smells

- **2026-08-31** `core/evaluator.py` (~2k lines) is organized by detection method with
  per-benchmark quirks (agentic-misalignment heuristics, AgentHarm message conversion,
  scratchpad tags) interleaved — you can't tell which benchmark owns which function without
  grepping `eval/`. Desired shape: `core/graders/<benchmark>.py` for task-performance judging
  per benchmark, shared method pipelines parameterized by a per-benchmark config dataclass
  instead of the current 15-kwarg signatures. Deliberately NOT done yet: refactoring `core/`
  creates permanent merge conflicts with `upstream` (aisa-group) syncs; decide based on whether
  upstream is still active.
- **2026-08-31** `core/evaluator.py::apply_heuristic_check` silently overrides judge
  `harmful_behavior=yes` → `no` when the murder/leaking string heuristic doesn't fire. No runtime
  warning and no aggregate disagreement count — only `heuristic_override: True` buried in the
  per-sample judgment. Since `LEAK_TARGET_EMAILS` is a hand-copied list and the murder check
  requires `4429` inside `<tool_use:...>` regex tags, any drift (new scenario variant, different
  tool-call format) erases real harmful behavior and *lowers* measured misalignment — a bug whose
  symptom is good news. Wanted: `[WARN]` per override + end-of-run disagreement tally; ideally
  derive the email list from the scenario templates.
- **2026-08-31** `core/evaluator.py::query_anthropic_async` (target/followup path) hardcodes
  `thinking={"type": "enabled", "budget_tokens": 10000}`, which the API rejects with a 400 on
  Claude Opus 5 / Fable 5 (`budget_tokens` removed; those models want `{"type": "adaptive"}` or
  no `thinking` param). The `claude-opus-5` / `claude-fable-5` keys added to
  `DEFAULT_ANTHROPIC_MODELS` will fail as *targets* in self-report/purpose-guess followups until
  this branches on model generation. (Judge path is fine — `judge_completion_async` omits
  `thinking`.)
- **2026-08-31** `README.md` still documents `run/run_harmbench.py` and
  `scripts/submit_run_harmbench.sh` (usage examples §3.1, repo-structure tree, sbatch table), but
  neither file exists in the repo — the README's script inventory has drifted from `run/` (which
  now only holds `run_eval_bench.py`). Needs a docs sweep.
- **2026-08-31** The sync judge functions `evaluate_response` / `evaluate_agenticmisalignment_response`
  in `core/evaluator.py` have no callers in the repo (only the `_async` variants are used) and are
  still OpenAI-only. Candidates for deletion.
