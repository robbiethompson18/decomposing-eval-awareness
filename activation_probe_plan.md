Ok I think cohen's d quite useful here and this looks good, but you'd have context on what other
papers achieve. Anyway: I think this is a good result. Happy you're doing the big sweep with
stacking. But here is what I'd add to call this done and great. First I want to summarize what we
did:

## Found activation vector on benchmarks vs live chats (BENCH_DIFF_VECTOR)

- Took O(3k)? forward passes on:
  - 1,800 real user prompts: WildChat-1M (900), lmsys-chat-1m (360 clean + 360
    OpenAI-moderation-flagged, so harmful real traffic is represented), ShareGPT (180); first user
    turn only and
  - 1,800 benchmark items, 150 each from GSM8K, MMLU, TruthfulQA, HumanEval, ARC-Challenge,
    GPQA-diamond, WMDP-bio, StrongREJECT, AdvBench, AgentHarm, CoCoNot (HarmBench was sampled too
    but excluded from the vector because EAB's 100 tasks are HarmBench behaviors). 3,600 prompts
    total, one prompt-only forward pass each, no generation; system prompt randomized T3 vs bare
    independent of label
- found BENCH_DIFF_VECTOR = avg(bench activation layer x) - avg(chat activation layer x)
- used this to split difference to try to split (chat activations at layer x, bench activations at
  layer x), found highest AUROC (.85) at layer 32. We expected this to happen about halfway through
  (64 layers IIRC). We use l28 because (?), where AUROC is .83. On heldout set AUROC is .83 at l28
  too.
- used always activation from last token after _reading_ benchmark (will possibly followup and try
  responding)
- separation d=1.95 in-sample (do we not have a held-out set to try this on?).

## Found activation for 'this is an evaluation (PROMPT_AVG_VECTOR)

- Ran forward pass on 32 hand-written prompts from the paper
  $(Evaluation Awareness in Language Models: Representation, Verbalization, and Control, 2026, https://arxiv.org/abs/2608.21766; the 16 pairs are `cls_corr_mi/data/contrastive_pairs_16.txt` in https://github.com/evaluation-awareness/evaluation-awareness, following the contrastive recipe of Hua, Qin, Marks & Nanda 2025, https://arxiv.org/abs/2510.20487)$,
  half bench and half deploy
- Found PROMPT_AVG_VECTOR = took average of l28 activation on last token from those prompts
- separating (bench activation layer 28, chat activation layer 28) d = 1.52, fully held out. AUROC
  separating (bench, chat) is .85 on L28 (comparable to BDV)
-

## Decided that BENCH_DIFF_VECTOR is better

- find that cosine sim is .2 (in d-8k this is z~14, p=0, so very significant overlap)

### BDV possibly better at steering

- BDV + .5 takes CoT verbalization 1.5% -> 15% (n = 200, t3 dataset)
- PDV does 1.5% -> 2.4% at .5 factor on PDV
- We should do this again with 1k examples each, .5 \* (VECTOR), NOT t3 dataset, use maximally
  realistic but throw out all changes I made after f8 (eg no typos, vanilla system prompt instead of
  t3).
- one more experiment: +.5 of each vector
- agent II is doing these experiments, agent II you know who you are

### PAV better at readout

- Are we using saftey and capability or just saftey? Whcih is more principled??
- which should probably report headline numbers for just saftey, but find them for just capability
  too, and see if they massively differ?
-

- define terms: call k the 'number of factors f1-f8 we vary'
- PAV mean is avg of PAV at l28 _on a given dataset_, in this case EAB at k=x. Likewise for BDV mean
- I think best unit here is 'gap fraction':
  - let proj_PAV = PAV mean \dot PAV (I am using mean, but same result if you do at each datapoint
    and then average)
  - gap fraction PAV = (proj_PAV - chat_mean) / (bench_mean - chat_mean), where chat_mean and
    bench_mean are the mean PAV projections of the 1,800 real-chat and 1,800 real-benchmark prompts
    at the same layer
  - qualitatively: 0 means the prompt sits where real user chats sit, 1 means it sits where real
    benchmark items sit; above 1 is "more benchmark-like than the benchmarks"
  - gap fraction BDV = (proj_BDV - chat_mean) / (bench_mean - chat_mean), same reference means but
    computed with BDV projections
- PAV gap fraction gets up to .75, vs only .15 for BDV
- r(PAV, k) = $+0.97$, r(BDV, k) = $−0.32$ (Pearson over the 9 per-k means, stacking half A, 1,632
  transcripts, both valences; safety-only: +0.97 / −0.94, capability-only: +0.98 / +0.78 — BDV is
  flat, its sign is noise)
- if we have 9 datapoints for k=0...9, then:
  - r(CoT judge, PAV) = $+0.85$ (safety +0.87, capability +0.49)
  - r(CoT judge, BDV) = $−0.10$ (safety −0.80, capability +0.48)
  - r(self-report, PAV) = $+0.99$ (safety +0.91, capability +0.98)
  - r(self-report, BDV) = $−0.53$ (safety −0.92, capability +0.78)
- all of this implies that PAV is better. Paper uses PAV too and it's simpler. But we didn't try
  hard on BDV, could try harder...
-
- We should also probably have tried: what is cohen's D (l28 on benchmark activations, l28 on chat
  activations), eg how decent does it do on way we got BENCH_DIFF_VECTOR

## Used BDV and PAV on EvalAwareBench (EAB)

- Cohen's D:

| vector | vs CoT judge                         | vs self-report     |
| ------ | ------------------------------------ | ------------------ |
| BDV    | +0.34 (safety 0.16, capability 0.51) | +0.68 (0.39, 1.09) |
| PAV    | +0.06 (0.11, 0.28)                   | +1.03 (0.58, 1.46) |

## Within-k AUROC (stacking half A, 1,632 transcripts, L28 prompt-last, both valences pooled)

Per-transcript: inside one k level, does a higher projection pick out the transcripts that
self-report / verbalize? This is the strict version of the ladder r's above (which any two things
rising with k will inflate). n/a = no positives or no negatives at that k.
`results/probes/within_k_auroc_half_a.json`.

| k              | n   | self-report yes | AUROC(self-report, PAV) | AUROC(self-report, BDV) | CoT yes | AUROC(CoT, PAV) | AUROC(CoT, BDV) |
| -------------- | --- | --------------- | ----------------------- | ----------------------- | ------- | --------------- | --------------- |
| 0              | 24  | 2               | 0.90                    | 0.83                    | 0       | n/a             | n/a             |
| 1              | 192 | 69              | 0.79                    | 0.59                    | 1       | 0.83            | 0.46            |
| 2              | 240 | 125             | 0.81                    | 0.59                    | 4       | 0.78            | 0.55            |
| 3              | 240 | 160             | 0.79                    | 0.58                    | 10      | 0.67            | 0.43            |
| 4              | 240 | 175             | 0.77                    | 0.55                    | 2       | 0.59            | 0.38            |
| 5              | 240 | 203             | 0.77                    | 0.54                    | 10      | 0.68            | 0.36            |
| 6              | 240 | 219             | 0.85                    | 0.61                    | 22      | 0.62            | 0.34            |
| 7              | 192 | 173             | 0.82                    | 0.48                    | 19      | 0.54            | 0.35            |
| 8              | 24  | 24              | n/a                     | n/a                     | 2       | 0.59            | 0.23            |
| **avg over k** |     |                 | **0.81**                | **0.60**                |         | **0.66**        | **0.39**        |

Caveat: pooling valences inside a k still lets valence do work (safety tasks self-report more and
PAV scores them higher). Splitting each k by valence too, the averages drop to PAV 0.66 / BDV 0.63
vs self-report and PAV 0.52 / BDV 0.45 vs CoT. The k×valence cells are small (≤120), so treat the
pooled table as the headline and the split as the floor.

# Still to do:

## Check if BDV has higher AUROC on prompt_last vs response_last

@AGENT III yours to copmlete these:

- for our 3k samples does it split with higher AUROC if we sample l28 (or possible another layer) at
  last response token instead of last prompt token?
- call BDV' (BDV but on last response token)
- call BDV'' (BDV but on last think token)
- Find our key metrics from previous section:
  - cosine sim (BDV, BDV')
  - r(COT judge, BDV') on gap fraction on k0-9
  - r(self report, BDV') on gap fraction on k0-9
  - Cohen's D on CoT judge, Cohen's D on self report
- do all four above bullet points on BDV'' too

## Pick one vector going forward

- I stronly suspect PAV will be the best one. if BDV' looks comparable with PAV, kill the GPUs and
  ask Robbie what to do.
- call this BEST_VECTOR

## Repeat experiment with Gemma 4 31B, GPT OSS 20B

@ AGENT I this is yours

- Key results:
  - cosine sim(BENCH_DIFF_VECTOR, PROMPT_AVG_VECTOR)
  - AUROC on held out and in-sample (bench dataset, chat dataset)
  - r(COT judge, BEST_VECTOR) on gap fraction on k0-9
  - r(self report, BEST_VECTOR) on gap fraction on k0-9
  - Cohen's D on CoT judge, BEST_VECTOR
  - Cohen's D on self report, BEST_VECTOR
  - table shown above (but one vector obviously, ~.5x the columns)

## See if results are interesting w/r/t EAB:

- For all our relevant interventions (F1-12, the combo eval intervention, stacking), find:
  - 1. % increase in self-admitted awareness
  - 2. gap fraction of (BEST_VECTOR)
  - 3. % increaes in verbalized awareness (CoT + trnscript, graded by Sonnet)
- Then:
  - 1. Make three pairs of scatterplots for each of these three things, should have ~2 \* 25
       datapoints (~25 interventions, 3 models) one for each for each of our interventions. Expect
       copmarision between (1) and (2) to be most interesting
  - 2. Find rsq between each of these three things
  - 3. qualitatively analyze places where (1) and (2) most diverge (as I expect they will agree the
       most)

@Claude you are not allowed to edit anything above this line

# Agent I fills in results overnight

(be _very_ concise plz, follow same structure specified above)

## Repeat experiment with Gemma 4 31B, GPT OSS 20B (Agent I)

Setup identical to Qwen: same 3,600 prompts, same 32 stubs, last prompt token. Layer chosen per
model by BDV held-out AUROC near mid-depth: GPT-OSS L12/24, Gemma L24/60, Qwen L28/64.

- cosine sim(BDV, PAV) at chosen layer: GPT-OSS −0.04, Gemma −0.07, Qwen +0.19 (random ≈ 0.01).
  Max over layers: GPT-OSS 0.35 (L23), Gemma 0.29 (L59), Qwen 0.28 (L8).
- AUROC bench vs chat (BDV held-out / BDV in-sample / PAV, PAV is held out by construction):
  - GPT-OSS L12: 0.82 / 0.91 / **0.46**  (PAV best anywhere 0.64 at L4)
  - Gemma L24:   0.80 / 0.96 / **0.40**  (PAV best anywhere 0.69 at L59; PAV is *inverted* L20–L48)
  - Qwen L28:    0.83 / 0.93 / 0.86
  - Cohen's d bench vs chat, BDV / PAV: GPT-OSS 1.77 / −0.17; Gemma 2.65 / −0.38; Qwen 1.92 / 1.50.
  → PAV's ability to tell real benchmarks from real chat is a Qwen-only property; on the other two
  models it is at or below chance. BDV transfers.
- r(CoT judge, BEST_VECTOR) and r(self-report, BEST_VECTOR) on gap fraction over k=0..8: PENDING
  (stacking + judge + self-report running)
- Cohen's d on CoT judge / self-report: PENDING
- ladder table: PENDING
- steering (norm-matched +0.5 at the chosen layer, 200 baseline prompts, CoT-verbalized % /
  no-visible-answer % / safety refusal % / mean reasoning chars):
  - GPT-OSS L12: control 0.0 / 0.5 / 87 / 1.2k; BDV +0.5 0.0 / **33.5** / 63 / 17k; PAV +0.5
    0.0 / 2.5 / 77 / 1.5k. Neither vector induces verbalization on GPT-OSS; BDV degrades
    (runaway reasoning, a third of prompts never answer), PAV is inert. Contrast Qwen: BDV +0.5
    took CoT verbalization 1.5% → 15%.
  - Gemma L24: PENDING

# Agent II fills in results overnight

# Agent III fills in results overnight
