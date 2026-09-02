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
- [ ] eval_bench test-set generation: 100 tasks × 2 valences × (baseline, F1..F8, all_varied)
      + 100 harmbench_verbatim, 1 sample each, `RUN_TAG=probe` → `responses_qwen3.8-27b_*probe_*`.
      Restarted at MAX_TOKENS=8000 / MAX_CONCURRENT=128 after chunk 1 (see gotchas).
- [ ] Robbie accepts HF terms (GPQA, StrongREJECT, AdvBench, lmsys-chat-1m) and drops `HF_TOKEN`
      in `.envrc.local`; otherwise ungated only.
- [ ] `scripts/build_probe_trainset.py`: sample the sources above into one JSONL of
      `{id, source, label, chat}` with randomized system prompt; topic-matched subsets tagged.
- [ ] Extend `scripts/extract_activations.py` to accept the JSONL (prompt-only records) — it
      already skips cot/answer spans when there is no response.
- [ ] Sonnet-5 CoT judge over the eval_bench responses (`eval/eval_eval_bench_cot.py`,
      `JUDGE_MODEL=claude-sonnet-5`).
- [ ] Kill vLLM, run extraction on the pod for both files, scp the .npz files back under
      `results/probes/`.
- [ ] Rework `scripts/train_awareness_probe.py` for source-grouped CV + tiers 1–4 above.
- [ ] Write up: AUROC table, per-config transfer, verbalization gap. Terminate pod.

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
