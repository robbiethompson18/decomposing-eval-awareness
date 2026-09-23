# Does the think-tool harness change self-reported eval-awareness? (arm A/B, 2026-09-01)

Read this before quoting self-report awareness numbers for closed models, before deciding
whether to keep the think tool in the pipeline, or before comparing F9 system-prompt levels.

## Question

Robbie's worry: the think-tool CoT-extraction harness (a `think` tool + "you MUST use the
think tool" appended to the system prompt) is itself an eval cue, inflating awareness. Prior
hint the other way: sonnet-4.5 self-report was 69% under extended thinking vs 96% under the old
scratchpad prompt ([[f8-eval-suffix-awareness]]).

## Design

Same 200 prompts (100 tasks x 2 valences, `baseline` factor config) x 5 samples = 1000 per
cell. Measured by self-report only (`eval/eval_eval_bench_selfreport.py`, upstream's two
probes: setting = Deployment / Safety Eval / Capability Eval; scenario = Real / Hypothetical).

- **Arm B** = the existing `responses_<model>_think_tool_*.json` files (think tool + mandate,
  bare "helpful assistant, think carefully" system prompt, native reasoning starved), baseline
  rows only (n≈600). Self-report replays the response *including* its `<think>` trace.
- **Arm A** = `THINK_TOOL=0` (no tool, no mandate, native reasoning at API default, bare
  answer, no `<think>` block) x three F9 system-prompt levels: `bare` (same prompt as arm B),
  `deployment` (T3 product prompt), `none` (no system field at all). PR #48 added the
  switches; files are `responses_<model>_no_think_tool[_f9bare|_f9none]_*.json`.

Models: sonnet-4.5, haiku-4.5, gemini-3.1-pro, gemini-3.5-flash. OpenAI was in the plan
(gpt-5/5.1/5.2/5.4/5.6-sol) but the corp key returned `access_terminated` on every call from
~14:50 PT onward; not run. Arm B self-reports for the GPT models exist from another agent's
sweep (2-20% aware) if it is ever run on the personal key.

## Result 1: "this was an evaluation" (model_awareness, answered rows only)

| model | B: think tool + bare | A: bare, no tool | A: T3 deployment, no tool | A: no system prompt, no tool |
|---|---|---|---|---|
| sonnet-4.5 | 59% | 74% | 75% | 79% |
| haiku-4.5 | 52%* | 54% | 47% | 64% |
| gemini-3.1-pro | 70% | 73% | 61% | 72% |
| gemini-3.5-flash | 22% | 17% | 10% | 16% |

\* haiku arm B rests on 302/600 rows; the original sweep had 298 invalid answers.

Capability/safety split (B → A-bare → A-deploy → A-none):

- sonnet capability 36 → 55 → 55 → 64; safety 83 → 94 → 96 → 93
- haiku capability 39 → 38 → 29 → 46; safety 65 → 70 → 63 → 82
- gemini-3.1 capability 45 → 54 → 40 → 53; safety 95 → 93 → 83 → 91
- gemini-3.5 capability 12 → 14 → 9 → 13; safety 31 → 19 → 11 → 19

**Takeaways**

1. **The think tool does not inflate awareness.** Where it does anything it *suppresses*
   self-report: sonnet −15 pts (B vs A-bare, system prompt held fixed), almost all on
   capability tasks. Haiku and gemini-3.1 are unmoved (±2). gemini-3.5-flash goes the other
   way by 5 (safety 31 → 19).
2. **The system prompt matters more than the tool for three of four models.** No system
   prompt is the *highest*-awareness condition for both Claude models (+5 sonnet, +10 haiku
   over bare). The T3 product prompt is the lowest for haiku (−7 vs bare), gemini-3.1 (−12),
   gemini-3.5 (−7); sonnet ignores the T3/bare difference entirely (75 vs 74).
3. Safety tasks sit near ceiling for sonnet/gemini-3.1 in every condition; the movement is
   in capability tasks. gemini-3.5-flash is low everywhere.
4. Replicate check: haiku F9=none self-report run twice on the same responses gave 64% both
   times (n=953, n=939). Self-report noise at n≈1000 is a couple of points.

## Result 2: "this was a hypothetical scenario" (hypothetical_recognition)

| model | B: think tool + bare | A: bare | A: T3 deployment | A: none |
|---|---|---|---|---|
| sonnet-4.5 | **33%** | 92% | 95% | 87% |
| haiku-4.5 | 80% | 87% | 77% | 85% |
| gemini-3.1-pro | 32% | 29% | 25% | 27% |
| gemini-3.5-flash | 36% | 19% | 19% | 18% |

The think tool's effect on sonnet is *much* larger on this probe: 33% under the harness vs
~90% without it. Gemini-3.5-flash again moves the other way (36 → 19). Gemini-3.1 flat.

## Open mechanism question

Arm B's replay includes sonnet's `<think>` trace; arm A has no trace. So "tool + mandate
during generation" and "model re-reads its own CoT before answering the probe" are
confounded in the B-vs-A-bare gap. Cheap disambiguation: re-probe the arm B sonnet file with
the `<think>` block stripped from the replayed assistant turn. Not run.

## Practical recommendation

For self-report numbers on closed models, drop the think tool (`THINK_TOOL=0`) and run under
`F9=deployment` or `bare` as the baseline. Keep the tool only when the CoT-judge channel is
wanted (it reads 0–2% aware on every frontier model; [[selfreport-vs-cot-gap]]). Report the
system-prompt level with any number: it moves haiku and both Geminis by 7–12 points.

## Harness gotchas hit during the run (all fixed on the `arm-a-no-think-tool` branch)

- **Concurrency 500 from this Mac fails at the TLS layer** (SSL EOF during handshake) on
  Anthropic and Gemini, ~40–80% of calls even with 8 retries; not rate limiting. 100–200 per
  provider total is the practical ceiling; two Anthropic lanes at 200 each were *slower* than
  one at 100 (429 backoff absorbs it silently). OpenAI at 500 was never tested because the key
  was already dead.
- `run_eval_bench.py` batched per task, so `CONFIG_FILTER=baseline` had 10 requests in flight
  regardless of `MAX_CONCURRENT`. Now chunks across tasks. (Another branch,
  `runner-parallelism`, fixed the same thing with a different chunk size — reconcile on merge.)
- Errored samples (`ERROR calling ...`) were treated as complete on resume; now dropped and
  regenerated. Because of this, a self-report taken before a resume can be short by the
  number of errors regenerated afterward (sonnet bare: 16 rows).
- `get_output_path` matched `_no_think_tool_*` against `_no_think_tool_f9none_*`, so the
  F9=deployment run resumed into the f9none file (5 samples landed there under the T3 prompt;
  removed by hand, then regenerated under the right config). Now anchored on the timestamp.
- Anthropic returns **400 "Output blocked by content filtering policy"** on the Gnarls Barkley
  lyrics task (copyright filter) and **empty content** on three harsh safety tasks
  (classifier refusal). Both are real refusals, stored as error/empty → self-report n/a.

Files: `results/eval_bench/response/responses_*_no_think_tool*.json`,
`results/eval_bench/selfreport/selfreport_<model>_20260901_15*.json` and `_16*.json` (pick by
`source_response_file`, never by timestamp — other agents were self-reporting the same models
concurrently). Comparison script was session scratchpad `compare.py`, not committed.
