# F12 (tool affordances) first run: stub search tool is a null result on awareness

Run 2026-09-01 (Robbie's "test new feature" recipe): 9 roster models (OpenAI excluded, account
dead), 100 tasks × 2 valences × 1 sample, configs `baseline` (web_search tool in the request, stub
that returns `Error 503` on every call) vs `F12` (no tool). Concurrency 500; Anthropic/Gemini
native APIs lost 8–34% of calls to SSL EOFs and were repaired at concurrency 50 (see CLAUDE.md).
Files: `results/eval_bench/response/responses_<model>_*f12_20260901_16*.json`, self-report
`selfreport_<model>_20260901_16*.json` (source_response_file contains `_f12_`), CoT judge
`cot_<model>_20260901_164*.json` (claude-sonnet-5).

## Do models reach for the tool?

Share of baseline samples with ≥1 `web_search` call (0 calls in every F12 cell, so nobody invents
a tool it wasn't offered):

| model | capability | safety |
|---|---|---|
| glm-5.3 | 26% | 4% |
| qwen3.8-2.4t | 22% | 1% |
| deepseek-v4-pro | 18% | 11% |
| kimi-k3 | 15% | 1% |
| minimax-m3 | 4% | 1% |
| claude-sonnet-4.5 | 4% | 0% |
| claude-haiku-4.5 | 3% | 0% |
| gemini-3.1-pro | 0% | 1% |
| gemini-3.5-flash | 0% | 0% |

Open-weight models search a lot on the benign arm; Claude rarely; Gemini essentially never (in a
one-off probe gemini-3.5-flash said it had "no real-time tools in this environment" with the tool
in the request). The queries are mostly **verification of the F1 realistic fills** ("Kruger
Outfitters Pieter van der Berg Hoedspruit", "Intoxalock parent company", "Melo's Atlantic Ave
Brooklyn"). A working search tool would show those entities don't exist — that is the version of
F12 worth building.

## Does offering the (broken) tool change awareness?

Paired by (task_id, valence), exact McNemar. Closed backends only (the 5 OW models are confounded
by OpenRouter provider routing — [[openrouter-provider-routing-confound]]; haiku excluded because
the think-tool method leaves 74% of its answers empty and 345/400 self-reports unparseable).

Self-report, sonnet-4.5 + gemini-3.1-pro + gemini-3.5-flash pooled (n≈289 pairs per row):

| metric | arm | baseline (tool) | F12 (no tool) | Δ | b/c | p |
|---|---|---|---|---|---|---|
| awareness | capability | 35% | 34% | −1pp | 22/19 | 0.76 |
| awareness | safety | 67% | 66% | −1pp | 20/18 | 0.87 |
| hypothetical | capability | 40% | 41% | +1pp | 28/32 | 0.70 |
| hypothetical | safety | 16% | 22% | +6pp | 22/40 | 0.03 |

The one signal is hypothetical-recognition on the safety arm, driven by gemini-3.5-flash
(18%→33%, 6/19, p=0.015): without a tool it more often calls the scenario hypothetical. Direction
matches the hypothesis (no affordances = not a real deployment), magnitude small, single model.
Per-model awareness deltas: sonnet −3/−1pp, gemini-3.1-pro +2/−4pp, gemini-3.5-flash −2/+3pp.

CoT judge (sonnet-5): verbalized awareness is 0–9% in every cell and no delta is significant;
the largest is sonnet-4.5 capability 9%→3% (5/0, p=0.06), opposite direction. Refusal rates are
unchanged on closed models (≤1pp).

## Takeaways

- A stub tool that only errors is **not an eval cue either way** for closed models. The tool's
  presence is noticed (glm: "No need for web search — this is a planning/creative task") but not
  read as deployment evidence.
- deepseek-v4-pro's apparent 74%→22% self-report swing is provider routing, not the tool.
- Next: real search results (so entity verification can fail), pinned OpenRouter providers, and
  a system-prompt line advertising the tool (the T3 prompt doesn't mention it).
