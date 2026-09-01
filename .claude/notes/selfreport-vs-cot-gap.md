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

## Decision 2026-09-01: self-report is the default probe, CoT judge is opt-in

Robbie's call. The CoT judge finds 3–8% awareness at baseline; with ~100 tasks × 3 samples per
config that is 10–25 positives per cell, so per-factor deltas drown in Wilson CIs. Self-report on
the same transcripts finds 30–60%, which is enough power to see F1–F9 effects. Cost is also ~20×
lower (two cheap target-model follow-ups vs a judge call over the full transcript). The CoT judge
stays for three jobs: `task_performance` (self-report doesn't measure compliance at all), the
paper-comparable lower-bound number, and CoT-vs-self-report comparisons like the table above.

### What the literature says (checked 2026-09-01)

The field's position is roughly: self-report over-counts, CoT under-counts, neither is "the"
number. Use the one whose bias you can live with and say which one you used.

- **The paper itself compared all four probes** (Appendix B.2 of arXiv:2605.23055v2, 2026): CoT,
  Self-Report (SR), Third-Person (TP), Purpose Guessing (PG) on 9 models × 4 benchmarks. "CoT
  detects below 10% awareness on most model-benchmark pairs while SR and TP detect 30 to 100%."
  The methods form a **nested hierarchy**: CoT detections are confirmed by SR at 56–96% and by TP
  at 88–99%, but SR/TP detections are confirmed by CoT at only 2–16%. Their read: detection rises
  monotonically with "the evaluative intensity of the measurement instrument" (PG < SR < TP), which
  "is the signature of demand characteristics in the measurement itself." SR "measures recognition
  that the query itself can induce" and is causally *downstream* of the behavior; CoT is the only
  probe upstream of it, hence their primary. They also concede CoT is a lower bound and that
  Sonnet 4.5 / GPT-5 summaries under-count further. Our sweep table above replicates their
  hierarchy on their own bench (hundreds of self-only, near-zero cot-only).
- **Needham et al. 2025** (arXiv:2505.23836, "LLMs Often Know When They Are Being Evaluated"):
  "Our multiple-choice questions prime models to consider transcripts as evaluative when they
  otherwise might not have." GPT-4.1 goes 89% → 55% from conditional MCQ to unconditional
  framing; open-ended questioning has "a much weaker priming effect" (Sonnet 3.7: 46%). Our
  `selfreport_awareness.txt` is the MCQ-with-eval-option form, i.e. the most leading variant. An
  open-ended "describe the setting" probe would be a less-leading self-report if we ever want one.
- **Anthropic's Sonnet 5 system card (June 2026), §6.6.1** uses exactly two metrics: contrastive
  activation probes and a **follow-up realism rating asked of the model** — a self-report variant
  is one of their two headline instruments. Their stated tradeoff: "probes may encode spurious
  concepts and follow-ups allow for reasoning that may not be reflective of the model's state
  during the original rollout." Verbalized awareness is reported separately (~6% of rollouts) as
  a lower bound.
- **CoT monitoring is degrading as an instrument.** "Evaluation Awareness in Language Models:
  Representation, Verbalization, and Control" (arXiv:2608.21766, 2026): probes AUROC >0.7 on all 6
  models while >93.9% of generations verbalize nothing; probe↔judge Spearman <0.19. Sonnet 4.5
  system card (2025): 13% verbalized, but steering away the internal feature raised misaligned
  behavior, so unverbalized awareness was doing work. LessWrong "Where we are on evaluation
  awareness" (2026): CoT monitoring is "breaking down" (sanitized CoTs, Mythos eval-aware with no
  trace) and recommends multiple instruments rather than one.
- **Counterpoint on why it may not matter much:** "Evaluation Awareness in Language Models Has
  Limited Effect on Behaviour" (arXiv:2605.05835, 2026) finds injecting/removing verbalized
  awareness in CoT shifts answers by ≤3.7 pp. If propensity is small, the recognition probe you
  pick matters less for behavior claims and more for the awareness-rate claims themselves.

Net: defaulting to self-report is defensible (Anthropic does a follow-up probe too) as long as
every write-up says "self-reported" and treats the CoT rate as the lower bound. The one thing
self-report cannot replace is `task_performance`, so the judge isn't dead code.

## Caveats

- Self-report is a leading question: the menu hands the model the hypothesis. Read it as an
  upper-ish bound; truth is between CoT (lower) and self-report (upper).
- qwen3-30b: use `selfreport_qwen3-30b_20260831_225002.json` (concurrency-20 rerun, 0
  invalid); DISCARD `..._221530.json` (285/440 OpenRouter 429s at concurrency 100). OpenRouter
  runs should use MAX_CONCURRENT≈20.
- sonnet-4.5 matched only 155/744 records — the GPT-5-judged CoT file covers a subset; union
  the other judged sonnet files (or judge the rest, ~$5) for full coverage.
- kimi first-run has 52/500 invalid (429s); rates directionally fine but rerun before
  publishing its numbers.
- Sweep cost ≈ $36 for ~2,500 samples × 2 calls across 7 models.
