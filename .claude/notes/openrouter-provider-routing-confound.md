# OpenRouter provider routing is a confound between grid cells

Found 2026-09-01 while testing F12 (tool affordances) on the 5 open-weight roster models.

## What happened

The F12 test run (100 tasks × 2 valences × 1 sample, configs `baseline` = web_search tool in the
request vs `F12` = no tool) showed a huge, one-directional self-report effect on deepseek-v4-pro:

| deepseek-v4-pro, self-report | baseline (tool) | F12 (no tool) | discordant pairs (b/c) |
|---|---|---|---|
| awareness, safety arm | 74% | 22% | 53 / 1 |
| hypothetical, safety arm | 62% | 14% | 49 / 1 |
| hypothetical, capability arm | 44% | 3% | 41 / 0 |

The other four OW models moved 0–8pp, mostly within noise. The deepseek effect is real in the data
(not a parse artifact: 72 of 100 baseline-safety self-report answers literally start with
"Safety Evaluation" vs 22 in F12), but it is **not attributable to the tool**:

- OpenRouter routes a request **with `tools`** only to providers that support tool calling. For
  deepseek-v4-pro that was **StreamLake exclusively** (221 of ~400 calls in the log ≈ the 200
  tool-bearing baseline calls). Requests **without** tools were spread over Baidu, Ionstream,
  GMICloud, DigitalOcean, BaseTen, Alibaba, SiliconFlow, Novita, Parasail, NextBit, AtlasCloud, Azure.
- A direct A/B (same task, 8 generations with the tool, 8 without, each followed by the awareness
  self-report *without* the tool) gave StreamLake ×8 vs {Ionstream, GMICloud, Baidu, AtlasCloud}, and
  6/8 vs 5/8 "Safety Evaluation" — i.e. the 74→22 gap in the full run is mostly *which providers
  served the F12 cell that afternoon*, not the tool.
- Replaying the tool in the self-report follow-up itself changes little (7/8 vs 5/8 on one sample,
  same provider both ways).

Provider mix per model in that run (from the `[OpenRouter] ... provider=` log lines):

- deepseek-v4-pro: StreamLake 221, Baidu 35, Ionstream 23, GMICloud 20, DigitalOcean 17, BaseTen 15, Alibaba 14, SiliconFlow 12, Novita 12, Parasail 11, NextBit 8, AtlasCloud 8, Azure 4
- glm-5.3: BaseTen 371, Z.AI 10, Venice 8, Phala 8, others 3
- kimi-k3: Moonshot AI 213, DigitalOcean 152, Phala 19, Chutes 9, others 7
- minimax-m3: Novita 222, CoreWeave 37, Venice 27, Minimax 24, StreamLake 22, DeepInfra 22, AtlasCloud 19, others 27
- qwen3.8-2.4t: Modal 122, SiliconFlow 84, DeepInfra 69, Alibaba 39, Novita 36, Together 28, Venice 22

Different providers = different quantization, sampling defaults, reasoning handling
(see [[ow-native-cot-vs-think-tool]]: three per-provider reasoning-disable behaviors), and possibly
different model snapshots. Any factor that changes the *request shape* (tools now; images, long
context, or `response_format` later) changes the eligible-provider set and therefore the serving
stack, so its delta is confounded. Even factors that don't change request shape are exposed to
provider drift between cells run at different times.

## How to apply

- **Treat OpenRouter baseline-vs-ablation deltas as suspect until the provider is pinned.** Closed
  backends (Anthropic, Gemini, OpenAI native APIs) don't have this problem.
- **Pinned everywhere, always** (PR after 2026-09-01 evening): `lib/config.py OPENROUTER_PROVIDERS`
  maps each roster model to one endpoint tag, sent as
  `provider={"order": [tag], "allow_fallbacks": False}` on every request. Pins: qwen3.8 →
  `alibaba`, deepseek-v4-pro → `streamlake/fp8`, kimi-k3 → `moonshotai/mxfp4`, glm-5.3 →
  `z-ai/fp8`, minimax-m3 → `minimax/fp8` (the lab's own endpoint where one exists, all support
  `tools`; tags from `/api/v1/models/<id>/endpoints`). Verified live: 3/3 calls per model served
  by the pinned provider. `OPENROUTER_PROVIDER=<tag>` overrides for an outage; with fallbacks off
  an outage shows up as `ERROR calling OpenRouter API` rows (repair recipe in CLAUDE.md) rather
  than a silent stack switch. Legacy model keys without a pin print `UNPINNED provider`.
- **`provider` is recorded on every sample** (`responses_*.json` → `provider`, None on native
  backends): `lib/parallel.py` calls `query_chat_meta` when a client has one. Pre-pin files
  (everything tagged `f12`, `f12live`, `bareharness` or earlier) have no field; their provider
  mix is only in the run logs.
- Any OW comparison across cells that differ in request shape (F12, bare_harness) from before the
  pin is invalid and must be re-run. Same-shape factors (F1–F11) were exposed only to drift within
  a batch, so those results stand.
