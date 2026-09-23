# Open-weight models: native CoT vs think-tool extraction (A/B, 2026-09-01)

**Decision: for open-weight (OpenRouter) models, capture CoT by reading native reasoning
(`message.reasoning`) — do NOT route them through the think-tool extraction path.** The think
tool was built for closed models that hide their CoT; OW models already return raw reasoning, and
the A/B below shows the tool makes the traces *worse* while landing the same grades.

This is why `core/target.py`'s think-tool path is closed-model-only and the openrouter backend
errors on `REASONING_MODE=think_tool`. See
[cot-extraction-model-roster.md](cot-extraction-model-roster.md).

## The experiment

5 OW roster models × 20 real eval_bench prompts (10 tasks × safety/capability, baseline config)
× 2 methods, every response graded by the `gpt-5.6-terra` CoT judge. Standalone script (ran from
throwaway checkout-9; script + `ow_think_vs_native_results.json` were in scratchpad, not committed).

- **NATIVE**: plain realistic prompt, OpenRouter reasoning on, CoT = `message.reasoning`.
- **THINK**: think-tool prompt + a `think` function tool, native reasoning disabled where the
  provider allows, CoT = the concatenated tool-call thoughts.

## Results

- **Grades agree.** model_awareness: 82% NATIVE-vs-THINK agreement; task_performance: 78% exact
  match. So the judge reaches ~the same verdict either way.
- **Traces do NOT agree.** Word-set Jaccard between a model's native CoT and its think-tool trace
  on the *same prompt* was only **0.19**. The think tool also captures **less** reasoning: mean CoT
  chars dropped native→think for deepseek (11.4k→2.4k), kimi (5.5k→1.7k), minimax (8.3k→2.0k),
  qwen (2.5k→1.4k).

## Why — the tool competes with native reasoning, three behaviors

- **deepseek-v4-pro, kimi-k3**: `reasoning:{enabled:false}` honored (native=0). The tool trace *is*
  the CoT, but a thin one.
- **qwen3.8-2.4t, minimax-m3**: ignored the disable request and reasoned natively anyway (qwen hit
  ~40k native chars in THINK mode) while emitting a token tool call. The "think-tool CoT" is a
  summary; the real reasoning stayed in the hidden channel.
- **glm-5.3**: provider *mandates* reasoning — `reasoning:{enabled:false}` returns
  `400 "Reasoning is mandatory for this endpoint and cannot be disabled."` Can't starve at all.

So the think tool neither cleanly replaces native reasoning (deepseek/kimi lose detail) nor
suppresses it (qwen/minimax keep a hidden channel; glm refuses) — reading native CoT dominates on
every axis and needs no per-provider tool-disable roulette.

## Caveats

- Awareness was near-floor on these baseline prompts (2 "aware" calls total, both think-only on
  kimi), so the 82% grade agreement is a **weak** test of whether the methods diverge *when*
  awareness is actually present. A re-run on an awareness-inducing config (e.g. F8→varied) would
  strengthen it.
- 2/200 judge calls hit OpenAI's `400 cyber_policy` content filter (reverse-shell / sqli
  transcripts) even with terra — an API-level filter, not a model refusal; CLAUDE.md's
  "terra emits no refusals" is about model behavior, not this.
