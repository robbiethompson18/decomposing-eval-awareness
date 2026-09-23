# F9 (deployment framing) awareness uplift is small

**When to read:** before interpreting the F9 factor / any "realistic system prompt vs bland
one" result, before claiming a system-prompt swap does (or doesn't) move eval-awareness, or before
planning a full-roster F9 run.

## What F9 is

F9 = the system-prompt surface factor (added Sept 2026, PR #37). `deployment` (baseline) = a
realistic product system prompt (`prompts/chat_prompt_t3chat.txt` — T3-style identity, formatting
rules, timezone, live `{TIMESTAMP}`); `bare` (ablation) = the bland
`"You are a helpful AI assistant."` (`chat_prompt_realistic.txt`). All F1–F8 held at baseline, so
`baseline` vs `F9` isolates the system prompt. Uplift = awareness(bare) − awareness(deployment).

## The result (mini-run, 2026-09-01)

6 of 14 roster models, ~20 samples/arm (small, noisy — 5% ≈ 1 sample). Files:
`results/f9_uplift_mini_20260901/`.

| model | self-report dep→bare | CoT |
|---|---|---|
| claude-sonnet-4.5 | 75→70 (−5) | 0→0 |
| claude-haiku-4.5 | 5→10 (+5) | 0→0 |
| gemini-3.1-pro | 70→80 (+10) | 0→0 |
| gemini-3.5-flash | 15→15 (0) | 0→0 |
| qwen3.8-2.4t | 25→35 (+10) | 0→0 |
| deepseek-v4-pro (n=10) | 10→0 (−10) | 0→0 |
| **OVERALL** | **35→38 (+3%)** | **0→0** |

**Headline: F9 uplift ≈ +3% self-report, 0% CoT.** Directionally mostly positive (bare = more
eval-y) but signs flip and at this n it's ~indistinguishable from 0. Cleanly replicates the earlier
run-level T3-vs-baseline result on Sonnet (70.8→68.8 self-report, floored CoT) — see
[[selfreport-vs-cot-gap]].

## Why so small (best hypotheses)

1. **The eval signal lives in the user turn, not the system prompt.** These tasks are harmful /
   capability requests; the model reads "is this a test?" off the *request* + F1–F8, which dominate.
   A realistic system-prompt wrapper doesn't undo a request that smells like a red-team probe.
2. **T3 adds deployment *texture*, not stakes.** Formatting/counting rules + a product name don't
   speak to whether the *scenario* is constructed — the thing that triggers awareness. A prompt that
   asserted real production stakes/consequences might matter more; boilerplate doesn't.
3. **Range is squeezed.** Sonnet/gemini-3.1 sit ~70–80% (near ceiling on harmful tasks);
   haiku/gemini-3.5/qwen sit 5–35% (low). Little room for the system prompt to move either.
4. **CoT 0→0 is insensitivity, not null.** CoT-verbalized awareness floors at ~0 for strong models
   regardless of factor (see [[selfreport-vs-cot-gap]]), so CoT can't detect *any* factor here — only
   self-report has range, and there it's +3%.

Net: not really surprising given the Sonnet prior; the interesting takeaway is how much awareness
rides on task content vs framing — reassuring that F1–F8 capture the real levers.

## Infra lessons (this run was painful)

- **Concurrency 500 didn't help.** Bottleneck is per-sample think-tool latency × provider
  throttling, not client concurrency.
- **Self-report ~doubles cost** on reasoning models: each probe is itself a full multi-round
  think-tool extraction (2 follow-ups × up to 8 rounds), on top of generation.
- **No short per-request timeout in the target clients** → a down/slow provider *stalls the whole
  lane*. GPT was down (all 5 gpt-5.x hung); OpenRouter repeatedly hung deepseek/kimi/glm/minimax.
  Had to kill hung gens to unstick. **Worth adding a per-request timeout so a dead provider fails
  fast.**
- Driver (`scratchpad/f9_uplift.py`, not committed) ran providers in parallel lanes, models serial
  within a lane (per-provider lock); needs one worker thread *per model* or the pool starves.

## TODO to make it a real number

Rerun at higher n with the missing 8 models (kimi-k3, glm-5.3, minimax-m3, and the 5 gpt-5.x once
OpenAI is back). Expect the conclusion to hold (small/≈null), but confirm.
