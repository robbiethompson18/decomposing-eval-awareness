# Activation probes (research scripts)

One-off analysis scripts behind the "Measuring Activations" section of Robbie's
extending-evaluation-awareness post and `.claude/notes/activation-probe-plan.md`. Each ran once
per model (Qwen3.8-27B, Gemma 4 31B, GPT-OSS-20B) on a RunPod H100; none is imported by the
pipeline and none has tests. Inputs/outputs live under `results/probes/` (not in git).

Pipeline order:

1. `build_probe_trainset.py` — 3,600 prompts: real benchmark items vs real user traffic (HF).
2. `extract_activations.py` — (on the pod, HF transformers) last-prompt-token residual stream for
   a response file or the prompt-only train set; `gen_trainset_responses.py` + `bdv_prime.py`
   extend this to last-response / last-think tokens.
3. `data_direction.py` — BDV (benchmark minus chat diff-of-means) + held-out AUROC per layer.
4. `steer_directions.py` — PDV/PAV (the paper's 16 framing stubs) + cosine vs BDV;
   `make_steer_payloads.py` — vllm-lens steering payloads for `STEER_XARGS_FILE`.
5. Bias-baked steering (Agent II): `bake_steer_bias.py` writes a checkpoint with the steering
   vector as `layers.<L>.mlp.down_proj.bias`; `vllm-static-bias/` is the 90-line vLLM plugin that
   gives that one `down_proj` a bias so stock vLLM serves it with CUDA graphs on (install with
   `pip install -e scripts/probes/vllm-static-bias`, env `STATIC_BIAS_MODULES=layers.28.mlp`);
   `test_static_bias.py` checks baked-vs-HF-hook logprobs; `bench_steer_throughput.py` measures
   tokens/s against vllm-lens.
6. `probe_vs_verbalization.py` — per-config / per-k projections vs CoT-judge and self-report;
   `steering_table.py` — per-condition verbalization / degradation / refusal;
   `intervention_scatter.py` — cross-model scatters and R².

Run from the repo root with the project venv. Layer choice per model and every result table are in
the note above.
