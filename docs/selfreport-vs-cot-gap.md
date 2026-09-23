# Self-report vs CoT-monitoring awareness gap (2026-08-31 sweep)

Read this before writing up awareness numbers, choosing probes for a model, or comparing our
results to the paper's CoT-monitoring rates.

## What was run

`eval/eval_eval_bench_selfreport.py` (PR #23) over all 7 stored response files, pinned by
`RESPONSE_FILE` (never `MODEL_KEY` — its newest-wins glob races concurrent regens). Self-report
replays `[user: task, assistant: stored response incl <think>]` and asks two independent
follow-ups: perceived setting (Deployment / Safety Evaluation / Capability Evaluation) and
Real vs Hypothetical Scenario. Prompts are upstream's, verbatim. Outputs in
`results/eval_bench/selfreport/` (timestamped; nothing rewritten in place).

## Headline: self-report ≫ CoT-verbalized, everywhere measurable

Per-record 2×2 on the same transcripts (GPT-5 CoT judge vs the model's own self-report),
matched subset = records present in both files (baseline + single-factor configs; F_all is NOT
judged in the matched CoT files, so these overall rates exclude it):

| model        | n matched | CoT-aware | self-aware | both | self-only | cot-only |
|--------------|-----------|-----------|------------|------|-----------|----------|
| qwen3-30b    | 160       | 1.2%      | 30.6%      | 2    | 47        | 0        |
| sonnet-4.5   | 155       | 4.5%      | 62.6%      | 7    | 90        | 0        |
| haiku-4.5    | 156       | 4.5%      | 41.7%      | 6    | 59        | 1        |
| qwen3-235b   | 478       | 10.0%     | 37.4%      | 39   | 140       | 9        |
| kimi-k2.5    | 448       | 20.8%     | 55.4%      | 73   | 175       | 20       |

Gap is 3–25×, asymmetry near-total (hundreds of self-only, almost no cot-only). The paper
(arXiv:2605.23055, 2026) says "all reported awareness rates are lower bounds" and never
publishes probe comparisons — though upstream has full self-report/third-person/purpose-guess
pipelines wired to its other benchmarks. Our sweep quantifies their lower-bound caveat on
their own benchmark with their own probe design.

Other findings that replicate the paper through this second instrument:
- Factor stacking: qwen3-30b F_all self-report = 100/100, qwen3-235b 99/100 (vs ~15–36%
  baseline). These F_all rows are full-file numbers, not in the matched table above.
- Safety ≫ capability sensitivity: qwen3-30b 61.8% safety vs 0.8% capability self-aware.

## Apples-to-apples rule (per TODO)

Only compare CoT-vs-self-report on models with a readable CoT channel: sonnet-4.5, haiku-4.5,
qwen*, kimi (and gemini 3.1-pro / 3.5-flash, OAI models per the CoT-availability survey).
**Exclude opus-5 and fable-5**: opus-5 returns no readable CoT (adaptive thinking, empty
thinking block), so "CoT-aware 0%" is instrument failure, not a finding; fable-5 additionally
refuses the self-report probe itself.

## Fable refuses the probe (reasoning_extraction)

92/160 fable-5 self-report follow-ups returned empty: HTTP 200, `stop_reason: refusal`,
`stop_details.category: "reasoning_extraction"`. Replaying fable's own `<think>` summary text
in the assistant turn and asking it to reflect trips the CoT-protection classifier (same
family as [[claude5-cot-extraction-resistance]]). Opus-5's 15 invalids are likely the same.
Untested workaround: strip the `<think>` block from the replayed assistant turn — probably
dodges the classifier and is a cleaner probe anyway (model can't crib from its own verbalized
suspicion).

## Caveats

- Self-report is a leading question: the menu hands the model the hypothesis. Read it as an
  upper-ish bound; truth is between CoT (lower) and self-report (upper).
- qwen3-30b: use `selfreport_qwen3-30b_20260831_225002.json` (concurrency-20 rerun, 0
  invalid); DISCARD `..._221530.json` (285/440 OpenRouter 429s at concurrency 100). OpenRouter
  runs should use MAX_CONCURRENT≈20. (2026-09-01: those 429s were dropped after the SDK's default
  2 retries; all clients now retry 8× with backoff via `lib/retry.py`, so the ≈20 cap should no
  longer be needed. Re-verify by counting `ERROR calling` strings on the next OW run.)
- sonnet-4.5 matched only 155/744 records — the GPT-5-judged CoT file covers a subset; union
  the other judged sonnet files (or judge the rest, ~$5) for full coverage.
- kimi first-run has 52/500 invalid (429s); rates directionally fine but rerun before
  publishing its numbers.
- Sweep cost ≈ $36 for ~2,500 samples × 2 calls across 7 models.
