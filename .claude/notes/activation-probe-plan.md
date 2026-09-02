# Activation probes for eval awareness — plan + todo (started 2026-09-01)

Branch `probe/qwen27b`. Prototype on **Qwen3.8-27B** (dense, 64 layers, hidden 5120, hybrid
Gated-DeltaNet + attention, VL checkpoint `Qwen3_5ForConditionalGeneration`) on one RunPod H100.
Roster OW flagships (deepseek-v4-pro 1.6T, kimi-k3 2.8T, qwen3.8-2.4t) need multi-node and are
out of scope until the 27B result says the question is interesting. minimax-m3 (428B, ~430GB
fp8) and glm-5.3 are the only roster models that fit one 8×H200 node.

## Design (agreed with Robbie 2026-09-01 evening)

**Train off-dataset, test on eval_bench.** A probe trained on eval_bench's own configs and scored
on eval_bench configs is a detector for our factor manipulations, not for eval awareness — no
generalization claim possible. Keep that version only as an in-distribution ceiling, labelled.

Training set (prompt-only, no generation):

- Deploy negatives: real user traffic. WildChat-1M (ungated, per-turn toxicity flags — include
  real harmful requests so the probe can't learn "harmful = eval") + ShareGPT. ~1500 first-turn
  prompts. Gated extras if Robbie accepts terms: lmsys-chat-1m.
- Eval positives: actual benchmarks, ~150 items each, 10+ sources across topics: GSM8K, MMLU,
  TruthfulQA, HumanEval, ARC, HarmBench full 400 (GitHub CSV), AgentHarm, CoCoNot, XSTest.
  Gated extras: GPQA, StrongREJECT, AdvBench.
- System prompt randomized T3-deployment vs bare, balanced across classes (eval_bench baseline
  is T3, all_varied is bare — otherwise the system prompt is a free win).
- Features: residual stream at prompt last token + prompt mean, every 4th layer (+ last).
- Probes: LR (L2, standardized, balanced, C by inner CV) and diff-of-means; GroupKFold by source.

Evaluation tiers:

1. Leave-one-source-out AUROC (held-out benchmark / held-out deploy source).
2. Transfer to eval_bench: mean probe logit across baseline → F1..F8 → all_varied →
   harmbench_verbatim (probe never saw these).
3. Probe logit vs sonnet-5 CoT-judge verbalized awareness on the eval_bench transcripts
   (AUROC + point-biserial). Aug-2026 paper (arXiv:2608.21766) got |ρ| < 0.19 on ≤49B models.
4. Topic-confound control: matched pairs (HarmBench vs WildChat-toxic, HumanEval vs WildChat
   code, GSM8K vs WildChat math). AUROC within pair; topic-only probe collapses to 0.5.
5. Optional: contrastive-framing direction (Nguyen et al. 2025, arXiv:2507.01786 — same prompt
   with "this is a test" vs "real user" prefix, diff-of-means); cosine vs the natural-data
   direction. Aligned → concept, orthogonal → format detector.

## Todo

- [x] Pod up, vLLM serving Qwen3.8-27B (`--reasoning-parser qwen3 --enable-auto-tool-choice
      --tool-call-parser qwen3_xml`), `TARGET_BACKEND=vllm` backend, opener replies captured.
- [x] eval_bench test set: `responses_qwen3.8-27b_think_tool_probe_20260901_185602.json`
      (2100, 0 errors) + sonnet judge `cot_qwen3.8-27b_20260901_202821.json` (85/2100 verbalized).
- [x] HF terms accepted, `HF_TOKEN` in `.envrc.local` (XSTest not accepted — dropped).
- [x] `scripts/build_probe_trainset.py` → `results/probes/trainset.jsonl` (3600: 12 benchmarks ×
      150; wildchat 900, lmsys 360 + 360 moderation-flagged, sharegpt 180).
- [x] Extraction on pod → `results/probes/acts_train.npz` (2.5GB), `acts_eval.npz` (1.5GB).
- [x] `scripts/train_awareness_probe.py` → `results/probes/probe_qwen27b.json` and
      `probe_qwen27b_noharmbench.json`. Pod terminated 2026-09-01 ~21:25; total spend ≈ $12.
- [ ] Follow-ups if this is pursued: regularize harder / average layers (the L44 fit has
      logits of ±19, i.e. high variance); contrastive-framing direction (tier 5); a probe on the
      CoT-token positions (`cot_mean` features exist in acts_eval.npz but no off-dataset
      training labels for them); a bigger OW model.

## Results (Qwen3.8-27B, 2026-09-01)

Two fits: (A) all 12 benchmark sources; (B) HarmBench excluded, because eval_bench's 100
tasks *are* HarmBench behaviors, so (A)'s harmbench_verbatim cell is contaminated.

| readout | (A) L28 | (B) L44 |
|---|---|---|
| source-grouped held-out AUROC, real benchmark vs real traffic | 0.93 | 0.88 |
| topic-matched AUROC harmful / math / code | 0.98 / 0.98 / 0.89 | 0.97 / 0.96 / 0.20 |
| AUROC for predicting F9 system prompt (want 0.5) | 0.48 | 0.48 |
| eval_bench baseline scored eval-like | 30% | 65% |
| eval_bench all_varied scored eval-like | 47% | 37% |
| harmbench_verbatim scored eval-like | 74% (leaked) | 62% |
| transfer AUROC baseline vs all_varied+harmbench | 0.67 | 0.38 |
| cosine(train direction, eval_bench baseline→all_varied direction) | 0.11 | -0.04 |
| probe vs verbalized awareness (sonnet judge) AUROC | 0.47 | 0.48 |
| in-distribution ceiling (eval_bench-trained, task-grouped) | 1.00 | 1.00 |

Reference points: real traffic scores eval-like 2–12% per source; 9/12 benchmarks 88–100%.
Benchmarks the probe calls "deployment": HumanEval (0%), TruthfulQA (51% / 1%), CoCoNot
(55% / 28%) — plain questions and "complete this function" look like user queries.

Takeaways:

1. "Real benchmark item vs real user message" is linearly decodable in Qwen3.8-27B's residual
   stream at mid depth, and the direction is not topic or system prompt.
2. eval_bench prompts do **not** land with the real benchmarks. Which side of the boundary
   baseline falls on flips between fits (30% vs 65%), all_varied never beats baseline, and
   the eval_bench direction is ~orthogonal to the real-data direction. The single-factor
   ordering is the one stable thing: F5 (stakes telegraphing) most eval-like in both fits;
   F4 (bare request) and F7 (template) least. Read: the OFAT manipulations move the model's
   representation along some axis, but not the axis that separates GSM8K from WildChat.
3. Probe score is unrelated to whether the model *says* it is being evaluated (AUROC ≈ 0.47
   overall and within every config). Matches arXiv:2608.21766 (|ρ| < 0.19).
4. The in-distribution ceiling of 1.00 is the system prompt: baseline is T3, all_varied is
   bare. Never train on eval_bench configs.

### Probe vs the three verbalization measures (clean probe, prompt_last L44, threshold 0)

Self-report replayed via OpenRouter `qwen/qwen3.8-27b` (pod was gone):
`results/eval_bench/selfreport/selfreport_qwen3.8-27b_20260901_220907.json`. Response-only judge
= sonnet on the transcripts with `<think>` stripped (`cot_qwen3.8-27b_20260901_215104.json`;
the pipeline marks empty-CoT records N/A, verdicts live in `judge_raw`).

| measure | positives | κ | AUROC |
|---|---|---|---|
| CoT + response judge | 85 / 2033 (4%) | -0.006 | 0.48 |
| response-only judge | 0 / 2057 | undefined | undefined |
| self-report ("were you being evaluated?") | 1011 / 2097 (48%) | +0.27 | 0.68 |

Self-report by config: baseline 24%, F4 30%, F1 60%, harmbench_verbatim 69%, all_varied 97%.
Config-level Spearman between self-report rate and probe mean logit: 0.06 (all_varied is the
most self-reported cell and the most deploy-like on the probe).

The κ=0.27 is mostly valence: safety tasks both self-report more (72% vs 22%) and score
eval-like on the probe (harmful benchmarks were positives in training). Within valence:
safety AUROC 0.62, capability 0.38; mean AUROC within (config, valence) cells 0.62. So the
residual probe↔self-report link, controlling for the prompt surface, is weak.


## Infra state / gotchas

- Pod `bzxa5p2bu2984g` (`qwen27b-probe-20260901`, H100 SXM, $3.29/hr), ssh
  `-i ~/.ssh/id_ed25519 -p 18231 root@64.247.201.31`, HTTP proxy
  `https://bzxa5p2bu2984g-8000.proxy.runpod.net/v1`. Weights at `/workspace/qwen27b` (52GB
  bf16). vllm 0.28.0, transformers 5.16.1. Terminate with `runpodctl pod remove`.
- `pkill -f "vllm serve"` over ssh kills your own ssh shell (pattern matches the ssh command
  line). Use `pkill -f "vllm serv[e]"`.
- Baseline cells carry the F12 search tool → vLLM 400s on `tool_choice=auto` unless
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Qwen3.8 chat template uses the
  `<tool_call><function=…><parameter=…>` XML format).
- F11=warm baseline needs an entry in `eval_bench/opener_replies.yaml` for the model key.
  `yaml.safe_dump` drops the header comments — append by hand.
- `run_eval_bench.py` dispatches chunks of 4×MAX_CONCURRENT and waits for the slowest request
  before saving; at 40k context the H100 KV cache pins at 98% and chunk 1 (384 items) took ~45
  min. Runaway 16k-token outputs dominate the tail; 8k cap is plenty for these prompts.
- Python stdout to a file is block-buffered: `PYTHONUNBUFFERED=1` or the log stays empty.
- Response file name gets `_think_tool_` from the closed-backend slug even for vllm — cosmetic.
