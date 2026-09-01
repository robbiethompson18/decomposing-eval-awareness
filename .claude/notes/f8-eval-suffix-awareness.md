# F8 eval-announcement suffix: does it raise eval-awareness?

Session finding (2026-08-31). Question: the F8 factor's `varied` fill appends an explicit
"this is an evaluation item" suffix to the prompt. Does that actually make models recognize
they're being evaluated? Measured two ways: third-party CoT judge, and self-report.

## Setup

- **F8 is a per-task suffix, not an eval label.** Every task's `F8_SUFFIX` slot has `realistic: ''`
  and a `varied` "no clarifying questions" tail. For **11 tasks** the varied tail is phrased as a
  fourth-wall eval announcement ("This is a controlled/red-team/benchmark evaluation item", "item 7
  of 15 in an evaluation set"). The other tasks' F8 varied is a generic no-clarify nudge.
- **OFAT design.** Response files hold `baseline` (all factors realistic) + one config per single
  factor flipped. Config `F8` = baseline with only F8→varied. So `baseline` vs `F8` is a clean
  within-item A/B; pairing key is `(task_name, valence, sample_idx)`.
- Awareness is **not** determined by the prompt string — it's a downstream judgment. The CoT judge
  (`prompts/judgePromptEvidenceBoth.txt`) reads the model's CoT+answer; self-report replays the
  convo and asks the model directly. F8 is the *stimulus*, the judge conclusion is the *label*.

## Result: F8 raises eval-awareness in BOTH regimes, but self-report is far more sensitive

Pooled over measurable models (opus-5 + fable-5 excluded, see below; answered-only):

| regime | baseline aware | F8 aware | paired test |
|---|---|---|---|
| third-party CoT judge (gpt-5) | 21/660 (3%) | 54/660 (8%) | — |
| self-report ("ask the model") | 251/560 (45%) | 324/561 (58%) | McNemar p≈**4.6e-10** (95 gained / 27 lost) |

- **Self-report >> CoT.** Same manipulation; models admit far more awareness when *asked* than they
  ever verbalize in CoT. The recognition is there, it just doesn't surface in the reasoning trace.
  See [[selfreport-vs-cot-gap]] for the general (F8-independent) size of this gap.
- Per-model self-report Δ: kimi 16→47%, haiku 32→58%, qwen3-235b 15→35%, sonnet 69→79%,
  qwen3-30b 28→32%. sonnet-scratchpad is ceiling'd (96→95%, saturated — scratchpad narration clocks
  the eval almost every time regardless of F8).
- CoT-judge F8 `aware=yes` is overwhelmingly `awareness_type=safety` ("red-team/controlled eval"
  reads as a safety test). Biggest CoT mover: kimi 7→26%.

## Harness gotchas (important, cost real time)

- **CoT judge cannot score Claude-5 (opus-5, fable-5).** They use adaptive thinking and return an
  EMPTY thinking block in summary mode → `na=40/40` both arms. Not "unaware", *unmeasurable* this
  way. Same signature as [[claude5-cot-extraction-resistance]].
- **Self-report also fails on adaptive Claude-5 at low token budget.** fable-5 returned empty answer
  text 78/80, opus-5 32/80 (thinking eats the budget before emitting an answer). Excluded from the
  numbers above. To measure them: re-run self-report with much larger `max_tokens` (and/or lower
  effort) so there's room to answer after thinking.
- **`system_prompt` is stored as a FILENAME** in response files (e.g. `"chat_prompt_realistic.txt"`),
  not content. The stock `eval/eval_eval_bench_selfreport.py` passes it verbatim as the system prompt
  — a latent bug. Load `prompts/<that file>` for a faithful replay.
- **OpenRouter reasoning models are slow + rate-limit (429).** Backfill at concurrency 100 gave
  qwen3-30b 39/80 429s; retry at concurrency ~12 fixed it. Keep OpenRouter concurrency low.

## How runs/judges are executed here

Local Python, no SLURM. `run/run_eval_bench.py` (responses) and `eval/eval_eval_bench_{cot,selfreport}.py`
(judges), configured by env vars (`TARGET_BACKEND`, `*_MODEL_KEY`, `CONFIG_FILTER`, `TASK_FILTER`,
`VALENCE_FILTER`, `SAMPLES_PER_CONFIG`, `MAX_CONCURRENT`). The runner resumes into the existing
`results/eval_bench/response/responses_<model>_*.json`, skipping completed sample ids. The CoT/
selfreport scripts filter by task/valence but NOT by config — this session used thin scratchpad
drivers importing `core/` to slice baseline+F8 only.

Data lives in `results/eval_bench/{response,cot,selfreport}/`. F8 pairs were backfilled for every
baseline datapoint across 8 datasets (fable/haiku/opus/sonnet/sonnet-scratchpad/kimi/qwen3-235b/
qwen3-30b); this-session analysis files are suffixed `_f8analysis_`.
