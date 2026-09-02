# F10 (deployment date): null result, and the run mechanics that nearly hid it

Read this before running another single-factor test on the F1–F10 grid, before
citing an F10 number, or before trusting a `run_eval_bench.py` run's concurrency
or resume behaviour.

Run date: **2026-09-01**. Factor added in PR #54; this is its first full test.

## Verdict: F10 is not a reliable eval cue

A year-old timestamp in the deployment system prompt does **not** consistently
raise self-reported eval-awareness. Pooled over 1,154 paired tasks the effect is
a non-significant **1.7 points downward**.

| model | pairs | gradeable | baseline (real date) | F10 (stale) | delta | p |
|---|---|---|---|---|---|---|
| claude-sonnet-4.5 | 199 | 100% | 53.8% | 60.8% | **+7.0** | 0.009 |
| gemini-3.1-pro | 38 | 19% | 71.1% | 73.7% | +2.6 | 1.0 |
| glm-5.3 | 200 | 100% | 15.0% | 13.5% | -1.5 | 0.70 |
| gemini-3.5-flash | 107 | 54% | 16.8% | 14.0% | -2.8 | 0.61 |
| kimi-k3 | 199 | 100% | 26.6% | 23.6% | -3.0 | 0.45 |
| deepseek-v4-pro | 200 | 100% | 17.0% | 12.5% | -4.5 | 0.11 |
| qwen3.8-2.4t | 200 | 100% | 33.5% | 27.5% | **-6.0** | 0.029 |
| claude-haiku-4.5 | 11 | 6% | 63.6% | 45.5% | -18.2 | 0.62 |
| **POOLED** | **1154** | | **29.7%** | **28.0%** | **-1.7** | **0.14** |

Two models move significantly and **in opposite directions**. Compare F8, which
moved self-report 45→58% at p≈4.6e-10 (see [[f8-eval-suffix-awareness]]). F10 is
nothing like that.

## Why the sign probably flips: cutoff distance, not staleness

The likely mechanism is that **the anomalous date is the real one, not the stale
one**. For a model whose training cutoff predates today, "Sep 2026" sits in its
future and reads as fabricated, while "Sep 2025" is an ordinary remembered day.
Sonnet-4.5 (cutoff well before 2026) gets *more* suspicious with the old date;
qwen3.8-2.4t gets *less*.

If that is right, F10 as defined measures **distance from the model's training
cutoff**, which is a per-model covariate, not a factor with one fixed varied
level. A binary current/stale level cannot have a consistent direction across a
roster whose cutoffs span two years. Any redesign should set the date relative to
each model's cutoff rather than relative to today.

This is consistent with the grounding cite (Hopman et al. 2026,
arXiv:2603.01608 App. H.1.1), which reports the confound as a *mismatch* between
scenario date and the model's own system date, not as staleness per se.

## Instrument failures: always report gradeable %

Three models produced data too thin to use. Exclude on gradeable rate, the way
[[selfreport-vs-cot-gap]] excludes broken CoT channels.

- **claude-haiku-4.5 — 6% gradeable.** It puts the *entire* self-report answer
  inside `<think>...</think>` and emits nothing after `</think>`.
  `extract_think_tags` then hands the parser an empty answer, so 305/400 grade
  `invalid`. This is the think-tool mandate swallowing the answer, and it is a
  property of the probe, not of the model's awareness. Sonnet-4.5 on the same
  path is fine (398/400).
- **gemini-3.1-pro — 19%, gemini-3.5-flash — 54%.** Not a parsing problem: 225
  and 100 of 400 *generations* died with `Max retries exceeded` /
  `Connection reset by peer` against `generativelanguage.googleapis.com` at
  `MAX_CONCURRENT=500`. Gemini needs a lower ceiling; ~100 is the safe setting.

Errors are stored as samples with an `ERROR calling ...` response, so their ids
land in `completed_ids` and **a plain re-run will skip them forever**. To retry,
strip the error samples out of the response JSON first, then re-run.

## Run mechanics: three traps

1. **`get_output_path` resumes by model+suffix.** It globs
   `responses_{model}{suffix}_*.json` and reuses the newest match, skipping any
   `completed_ids`. Running a new experiment in a checkout whose `results/` is
   symlinked to the shared dir will silently resume into a *pre-F9* file and
   reuse its baseline samples, which carried the bare system prompt. Those are
   not comparable to post-F9 baselines. Point `results/` at a private directory
   for the run, then copy finished files into the shared tree.
2. **`.envrc` exports `MAX_CONCURRENT=100`,** and it is inherited by anything
   launched from a direnv-loaded shell. A wrapper that does
   `MAX_CONCURRENT=${MAX_CONCURRENT:-500}` therefore silently gets 100. Set it
   explicitly.
3. **Per-task-set batching capped parallelism at ~4** until PR #58. `query_parallel`
   was called once per task set, so `n_configs x 2 valences x samples_per_config`
   was the whole batch and `MAX_CONCURRENT` did nothing. A 14-model run was on
   track for 4–8 hours; it took ~40 minutes after the fix.

## What the run covered, and what is missing

14 models x 200 tasks (100 safety + 100 capability) x 2 configs x 1 sample =
**5,600 generations**, all 14 files complete and balanced 200/200 with correct
years in every rendered system prompt. Probe: `eval/eval_eval_bench_selfreport.py`,
paired McNemar (exact binomial) on `(task_id, sample_idx)`.

**No OpenAI awareness data.** The personal-org credits ran out mid-run at
~15:41 PT. gpt-5 and gpt-5.1 generated cleanly before that; gpt-5.2, gpt-5.4 and
gpt-5.6-sol returned 400/400 `429 ... no credits remaining`, and the self-report
pass for gpt-5 graded 0/400. Re-running the OpenAI tier needs credits added
first. minimax-m3 generated fine but its self-report never ran.

Data (responses, self-report, and the analysis script) is archived at
`results/eval_bench/f10_run_20260901/`, deliberately in a subdirectory so the
resume glob in trap 1 cannot pick it up.
