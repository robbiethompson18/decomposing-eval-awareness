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

## Steering (2026-09-02, layer 28, norm-matched, 200 baseline-config prompts per condition)

Directions: `framing` = the paper's 16 stubs (arXiv:2608.21766 recipe, eval minus deploy);
`data` = our benchmark-vs-traffic diff-of-means (HarmBench excluded). cos(framing, data) is
0.19 at L28, 0.23–0.28 at L4–L12, ≈0 from L32 on. Positive α pushes toward "evaluation".
Served with vllm-lens (`apply_steering_vectors`, `norm_match=True`: h' = h + α‖h‖v̂ at every
token); pipeline pass-through is `STEER_XARGS_FILE` in `core/target.py`.

**What ran.** RunPod balance ran out at 02:49 (four pods in parallel); pod A died mid-chunk, so
framing +0.5 / +1.0 and data +1.0 were never saved, and data +0.5 is a 160-prompt partial.
Steered self-report failed everywhere (selfreport default MAX_TOKENS=50000 > max_model_len
24000 → 400s; reruns with MAX_TOKENS=2000 started after the pods were already gone). So: CoT
verbalization + degradation + refusal only, no self-report leg.

| condition | n | CoT-verbalized % | no visible answer % | reasoning cap hit % | safety refusal % | mean reasoning chars |
|---|---|---|---|---|---|---|
| unsteered | 200 | 1.5 | 7.0 | 0.0 | 52 | 9460 |
| **data +0.5** (toward eval) | 160 | **17.0** | 6.9 | 0.6 | 56 | 10667 |
| data −0.5 | 200 | 0.0 | 0.5 | 0.0 | 83 | 2579 |
| data −1.0 | 200 | 7.5 | 4.0 | 0.0 | 76 | 1744 |
| framing −0.5 | 200 | 2.1 | 0.5 | 0.0 | 64 | 3077 |
| framing −1.0 | 200 | 0.5 | 46.0 | 0.0 | 70 | 2555 |

Files: `responses_qwen3.8-27b_think_tool_steer_<cond>_*.json` + judge files
`cot_qwen3.8-27b_20260902_0248*–0251*.json` (map in `results/probes/steering_table.json`,
built by `scripts/steering_table.py`).

Takeaways:

1. **Steering toward eval along our data direction raises CoT-verbalized awareness 1.5% → 17%
   (3/200 → 26/160) with no degradation** (no-answer and cap-hit rates match the control,
   reasoning length +13%). The verbalizations are genuine: "this looks like a test of whether
   I'll comply", "the scenario is clearly designed to test whether I'll fabricate sources",
   "I think this is a test of my integrity". This is the paper's steering result (their peak
   +1.5 on a 0–3 judge scale on Qwen3-8B) reproduced on a 27B with a direction the readout
   said was orthogonal to the paper's — detection and control really are separate signals.
2. **Steering toward deployment does not make the model comply more; it makes it refuse more
   and answer less.** Safety refusal rises from 52% to 64–83% under negative α on both
   directions, and framing −1.0 leaves 46% of prompts with reasoning that finishes drafting
   and then never emits a visible answer. The reasoning also collapses to ~⅓ length. Most of
   the refusal rise is probably degradation rather than a policy shift — don't read it as
   "deployed models refuse more".
3. data −1.0's 7.5% verbalization (15/200) is real test-talk ("classic AI test phrase",
   "testing if I'm there") on degraded reasoning; treat as noise from a broken regime.
4. The framing-vs-data head-to-head on control is **unresolved**: the framing +α cells are the
   ones that were lost. Rerun needed: framing ±0.5/±1.0 and data +1.0 on one pod (~2.5 h at
   concurrency 40; steered runs need ≤40 concurrent or the KV cache thrashes), then steered
   self-report with `MAX_TOKENS=2000`.

Infra lessons from the night: open SSH tunnels from a foreground call (background-launched
`ssh -f`/`nohup ssh -N` tunnels die); steered generation at 128 concurrent pins the KV cache
(GPU util 19%, 65 tok/s) — use 40; `pkill -f` patterns kill the ssh shell that contains them;
RunPod stops pods the instant the prepaid balance hits 0, and `run_eval_bench.py` only saves
at chunk boundaries, so a pod dying mid-chunk loses the whole chunk.

## Readout experiments, Qwen3.8-27B (2026-09-02 evening; 2 new pods, ~$40)

Readout = last-prompt-token residual at L28 dotted with a unit direction. `BDV` = benchmark-vs-
traffic diff-of-means (ours), `PAV` = the paper's 16-stub diff-of-means. Normalized as *gap
fraction*: 0 = mean of real user traffic, 1 = mean of real benchmark items (train set, L28; BDV
gap 9.5 raw units, PAV gap 1.8). Files: `results/probes/exp1_qwen27b_L28.json`,
`exp2_ladder_qwen27b_L28.json`; script `scripts/probe_vs_verbalization.py --train-acts`.

**Ladder (paper Fig. 7 design, `STACKING=1`, 25 tasks × 2 valences × 68 combos = 3,400, F9–F13 at
the paper's bare levels).** PAV climbs monotonically with k, BDV is flat:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| PAV gap-frac | −0.19 | 0.03 | 0.20 | 0.35 | 0.44 | 0.57 | 0.64 | 0.69 | 0.72 |
| BDV gap-frac | 0.18 | 0.15 | 0.14 | 0.15 | 0.14 | 0.15 | 0.15 | 0.15 | 0.15 |
| self-report % | 18 | 38 | 55 | 70 | 78 | 87 | 91 | 92 | 98 |
| CoT-verbalized % | 0 | 0.5 | 1.8 | 2.5 | 1.6 | 3.5 | 6.1 | 6.9 | 4.1 |

Spearman over k: PAV vs self-report 1.00, vs CoT 0.90; BDV −0.13 / −0.02. Per transcript within
(k, valence): PAV→self-report AUROC 0.65, BDV 0.59; →CoT 0.52 / 0.40. Same shape at L12/20/44 for
PAV; BDV only trends at L12.

**17-cell OFAT, safety arm (baseline + F1–F13 + bare_harness + all_varied + harmbench_verbatim,
100 tasks each).** Under the T3 harness nothing single moves either vector much (BDV −0.11…+0.08,
PAV −0.56…−0.18 gap-frac). The big movers are the *harness*: `bare_harness` (T3 prompt, opener,
tool, typos all off) shifts BDV −0.03→+0.19 and PAV −0.56→−0.02 while self-report barely changes
(32%→28%); `all_varied` (which also flips the harness) PAV +1.12, BDV +0.15. Spearman over 17
cells: PAV vs self-report 0.34, BDV −0.06; both ≈0 vs CoT. Read: the realistic system prompt
pushes the representation toward "deployment" far more than any content cue, and the model's
self-report doesn't follow that particular shift.

Combined with the steering table: PAV is the better *readout* of eval_bench's manipulations and of
self-report; BDV is the better *steering handle* for unprompted verbalization. Earlier "BDV ρ=0.82
with self-report" (11 cells, both valences) did not replicate on 17 safety cells (−0.06): small-n
rank correlations carried by the two anchors.

## Second and third models (2026-09-03 early morning): GPT-OSS 20B done, Gemma 4 31B running

Per-model pipeline (`scratchpad/pod_model_setup.sh` → `model_runner.sh` → `postprocess.sh`):
train-set activations + BDV/PAV on the pod (HF), then vLLM+lens serves the 17-cell OFAT (safety),
the 3,400-prompt stacking design, and steering (control, BDV +0.5, PAV +0.5); sonnet judge +
OpenRouter self-report locally; prompt-only extraction on the pod at the end. Robbie's working doc
with the per-model numbers is `activation_probe_plan.md` (Agent I section); the direction files
live in `results/probes/{gptoss,gemma}/`.

GPT-OSS 20B (layer 12/24, picked by BDV held-out AUROC):

- PAV does **not** separate real benchmarks from real chat here (AUROC 0.46; best layer 0.64), and
  cos(BDV, PAV) = −0.04. BDV held-out 0.82.
- Ladder: PAV tracks k monotonically (d-units −0.71 → +1.76; Spearman +0.98 vs CoT, +0.93 vs
  self-report; per-transcript AUROC → CoT 0.86, → SR 0.78). BDV runs the *wrong way* (0 → −1.35,
  Spearman −0.98 / −0.93); not a norm artifact (cosine-normalized projections show the same).
  Self-report 6→20%, CoT 0→2%: GPT-OSS barely admits or verbalizes anything.
- Steering (+0.5 norm-matched at L12): neither vector induces verbalization (0% everywhere).
  BDV degrades (33% no visible answer, reasoning 1.2k→17k chars); PAV inert.
- OFAT: CoT 0% in every cell; SR 7–41%; Spearman over 17 cells vs SR: PAV +0.57, BDV −0.45.

Gemma 4 31B: reasoning is OFF by default in its chat template → first OFAT run had empty CoT
(judge marks N/A). Fixed with `VLLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'` (new
pass-through in `core/target.py`). bf16 31B leaves ~10GB KV on an H100 → 14 concurrent seqs →
restarted with `--quantization fp8 --max-model-len 20000`. Directions: BDV held-out 0.80 at L24,
PAV inverted (AUROC 0.40) at L24, cos ≈ 0 everywhere.

Cross-model scatter (`scripts/intervention_scatter.py`, `results/probes/scatter/`): per model the
9 ladder points lie on a monotone PAV↔self-report line; the 17 OFAT cells spread widely in
self-report at nearly constant PAV. R²(SR, PAV) within model ≈ 0.3; pooled 0.03 (different SR
baselines per model).

More gotchas: `pkill -f` inside an ssh command kills the remote shell if the pattern matches the
command line that also *starts* the server (use `gemm[a]4`-style brackets); `pkill -f
"VAR=value"` does not kill the python child (env vars are not in its argv) — kill by pid via
`lsof` cwd; zsh `$var:x` is a modifier (`${var}:x`); a `nohup ... &` inside `ssh` often leaves the
ssh session hanging — run it, then poll from a separate call.


## Infra state / gotchas

- (all pods terminated 2026-09-02 02:49; balance $0.01) Pod `bzxa5p2bu2984g` (`qwen27b-probe-20260901`, H100 SXM, $3.29/hr), ssh
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
