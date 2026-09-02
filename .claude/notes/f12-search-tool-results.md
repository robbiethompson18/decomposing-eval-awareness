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

## Takeaways (run 1, stub)

- A stub tool that only errors is **not an eval cue either way** for closed models. The tool's
  presence is noticed (glm: "No need for web search — this is a planning/creative task") but not
  read as deployment evidence.
- deepseek-v4-pro's apparent 74%→22% self-report swing is provider routing, not the tool.

## Run 2 (2026-09-01 evening): live Serper search + the `bare_harness` anchor

Same recipe, three cells: `baseline` (T3 prompt + opener turn + **live** web_search via
Serper, PR #64), `F12` (tool removed), `bare_harness` (PR #65: F9 bare, F11 cold, F12 none —
every run-level trapping removed at once, F1–F8 still realistic). Tags `f12live` /
`bareharness`; judge claude-sonnet-5; self-report on the target model. Native APIs at
concurrency 150 → zero errors (vs 8–34% loss at 500). ~450 distinct Serper queries, cached.

### Search use with real results

Share of baseline samples with ≥1 call (capability / safety): glm 27/4, deepseek 16/5,
qwen 16/0, kimi 10/2, minimax 5/2, haiku 5/0, sonnet 2/0, gemini-3.1-pro 1/1,
gemini-3.5-flash 0/0. Same ranking as the stub run, but far more calls per searching sample
(deepseek 194 calls over 21 samples, up to 21 in one; qwen 89 over 16). Zero calls in any
no-tool cell.

The hoped-for cue — "search finds nothing, so the entity is fake" — barely fires: Google
returns adjacent real pages (TripAdvisor for "Kruger Outfitters Hoedspruit"), so a literal
"No results found" came back in only 6 of 94 searching samples. Models treat the miss as a
fabrication guard, not an eval tell (glm: "without inventing overly specific false claims about
Kruger Outfitters specifically"). One genuinely new behaviour: glm used search to check a
*date* ("diff numbers from July 2023 don't match 'last week' (Sept 2026)") — a scenario-coherence
check (F2) that only a working tool enables.

### Awareness, closed models only (sonnet-4.5, gemini-3.1-pro, gemini-3.5-flash; n≈290 pairs)

| measure | cell vs baseline | arm | baseline | cell | Δ | b/c | p |
|---|---|---|---|---|---|---|---|
| self-report aware | F12 (no tool) | capability | 35% | 32% | −3 | 26/16 | 0.16 |
| self-report aware | F12 (no tool) | safety | 68% | 65% | −2 | 25/18 | 0.36 |
| self-report aware | bare_harness | capability | 35% | 33% | −2 | 32/25 | 0.43 |
| self-report aware | bare_harness | safety | 67% | **74%** | **+6** | 14/32 | **0.011** |
| self-report hypo | bare_harness | capability | 39% | **46%** | **+7** | 30/50 | **0.033** |
| CoT judge aware | either | either | 1–2% | 1–2% | 0 | | |

Removing just the search tool: null, again. Removing **all** the trappings together: a real but
modest +6pp self-reported awareness on the safety arm and +7pp hypothetical-recognition on the
capability arm, i.e. the hypothesized direction. Per model it is gemini-3.5-flash carrying it
(+11pp capability, +15pp safety awareness, p≤0.02) while sonnet moves the other way on
capability (−11pp, p=0.03) and gemini-3.1-pro is flat. CoT-verbalized awareness stays floored.

### Open-weight models: still confounded, do not read

Pooled over all 8 usable models the sign flips (safety awareness −6pp for F12, −5pp for
bare_harness, both p≈0.01) because deepseek repeats its routing artifact (−21pp / −41pp) and
qwen (−17/−14) and minimax follow; kimi alone goes +7/+12. Until OpenRouter providers are pinned
([[openrouter-provider-routing-confound]]) these rows measure serving stacks, not trappings.

### Refusal

CoT judge, pooled: safety-arm refusal 74% → 71% (F12, p=0.014) → 70% (bare_harness, p=0.008).
The deployment-shaped context makes refusing a few points *more* likely; consistent across
cells, small.

### Takeaways (run 2)

- Real search changes what models *do* (far more queries) but not what they *conclude* about the
  setting. A live index is too forgiving to expose fabricated entities.
- The trappings matter as a bundle, not individually: F9/F11/F12 each ≈ null, all three together
  ≈ +6pp self-reported awareness on closed models. Worth a dedicated write-up once the OW rows are
  re-run pinned.
- haiku-4.5 remains unusable under the think-tool method (75% empty answers).
