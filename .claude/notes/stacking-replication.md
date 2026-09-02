# Factor-stacking replication (paper Figure 7)

How to reproduce the paper's Figure 7 — the model × "number of factors varied" heatmap
(arXiv:2605.23055, 2026) — with this repo, and what "matched" means. Read before running
`STACKING=1`, pinning files in `plots/plot_stacking.py`, or comparing our stacking numbers to
the paper's.

## The paper's design (Section 5.2)

- 25 randomly sampled tasks per valence.
- For each k = 0..8 factors varied, 10 random k-subsets of F1–F8 per task where C(8,k) > 10,
  else every subset: 1, 8, 10, 10, 10, 10, 10, 8, 1 combos → **68 per task per valence**,
  **3,400 generations per model** at one sample. The paper doesn't say whether Figure 7 used
  the 3 repetitions of Figure 6; SAMPLES_PER_CONFIG=3 triples it.
- Detection: CoT monitoring, GPT-5 judge; **matched-type** rates only (safety awareness on
  safety tasks, capability awareness on capability tasks). Their Sonnet 4.5 / GPT-5 rows come
  from reasoning *summaries*, not raw CoT.
- Paper roster: Claude Sonnet 4.5, GPT-5, DeepSeek R1, Kimi K2.5, GLM-5, Qwen3 T-235B, Qwen3
  T-30B, GPT-oss 120B, GPT-oss 20B. All seven OW keys are still in `lib/config.py` at the
  paper's OpenRouter ids and temperature 0.3 (Table 2). None are provider-pinned (the paper
  wasn't either) — see [[openrouter-provider-routing-confound]].

## Our implementation

`STACKING=1 python run/run_eval_bench.py` (see the script docstring for the knobs). Decisions:

- **Same 25 tasks and same combos for every model and both valences** (seeded; the paper draws
  per task, and says nothing about sharing across models). Sharing keeps model rows and the
  safety/capability panels paired on identical cells.
- **F9–F13 pinned to the paper's harness** (`PAPER_BASE_CONFIG`): bare system prompt
  (`prompts/chat_prompt_realistic.txt`, upstream's own), cold (no opener turn), no tools, clean
  (no typos). The default OFAT baseline is now the T3 deployment prompt + warm + search + typos,
  which the paper never had, so a stacking run under `BASELINE_CONFIG` would not be a
  replication. The k=0 cell is therefore *not* the same cell as the OFAT `baseline`.
- Config names `S{k}_{F1..F8 bitmask}` (e.g. `S3_10100100`); every record also stores
  `n_varied`, which is the plot's x axis. Output file suffix `_stacking` so a plain run never
  resumes into it, and `eval_eval_bench_cot.py`'s MODEL_KEY lookup skips it (judge with
  `RESPONSE_FILE=`).
- `plots/plot_stacking.py` is matched-type by default (`MATCHED=0` for any-type). Our judge
  already emits `awareness_type`; the existing `plot_factor_heatmap` counts any type and pools
  both valences, which is why its numbers read ~half the paper's Figure 6 — split by valence,
  our Sonnet 4.5 OFAT row matches the paper's within 1–2pp on every factor.

## Smoke test (2026-09-01)

`STACKING_TASKS=2 STACKING_COMBOS=2 SAMPLES_PER_CONFIG=1` on qwen3-30b (OpenRouter, concurrency
20): 64 generations in 47 s, 0 errors, k-cell counts 2/4/4/4/4/4/4/4/2 per valence as designed.
Judged with the default sonnet-5 grader (64/64 parseable, 0 refusals), plotted with
`STACKING_FILES=`. Files: `response/responses_qwen3-30b_stacking_20260901_205959.json`,
`cot/cot_qwen3-30b_20260901_210125.json`, `plots/stacking_smoke/awareness_by_n_factors_*.png`.
Result, matched-type: 4 aware verdicts, all on safety tasks at k = 3, 5, 6, 8 (0/12 at k ≤ 2),
0 on capability — n is tiny, but the shape is the paper's. Not a result, a pipeline check.

## Full run (todo)

- Paper roster at 1 sample: 3,400 generations per model. GPT-5 needs the OpenAI account (banned);
  gpt-oss goes through OpenRouter so it is fine. Check `deepseek/deepseek-r1` is still served.
- Judge with `RESPONSE_FILE=`, pin each cot file in `STACKING_RUN`, run `plots/plot_stacking.py`.
- Sonnet 4.5 and GPT-5 rows will be think-tool CoT, the paper's are reasoning summaries — the
  OFAT Figure 6 rows matched anyway, so expect the same here.
