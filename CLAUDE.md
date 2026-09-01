# decomposing-eval-awareness

Mirror of https://github.com/aisa-group/decomposing-eval-awareness — code for the paper "Decomposing
and Measuring Evaluation Awareness" (arXiv:2605.23055, 2026) — plus Robbie's own experiments.
`origin` is Robbie's public mirror (robbiethompson18/decomposing-eval-awareness), shared externally
as a research artifact; `upstream` is aisa-group's repo (read-only for us, sync with
`git fetch upstream && git merge upstream/main`).

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

There is a blog post I'm drafting for this for my personal website at
~/repos/personal-website/Posts/extending-evaluation-awareness/FINAL_POST.md. DO NOT write this post
for me. You can address my @claude comments by doing mechanical things (eg adding links) and by
giving me commmentary back in chat. All prose is mine.

OAI banned us for 'distilling' so further runs are on Robbie's credit card. Be very cautious about using these.

## Rules

- Robbie reviews every change in this repo. **Never push to `main` or merge on your own** — commit
  on a branch, push the branch, and open a PR for him to review.
  - exception: changes to .claude directory
- We are churning this sucker hard. Their codebase is a mess and I want to understand, so I'm
  deleting dead code and organizing packages more nicely
- Massive parellelism on all grader runs and eval runs. At least parallelism 100
- Default awareness probe (2026-09-01, Robbie): **self-report** (`eval/eval_eval_bench_selfreport.py`,
  the model grades its own transcript), not the third-party CoT judge. Self-report finds 3–25× more
  awareness than the CoT judge on the same transcripts, so it has the statistical power the CoT rate
  (3–8%) lacks. The CoT judge (`eval/eval_eval_bench_cot.py`) is **opt-in**: run it when you need
  `task_performance` (compliance — self-report doesn't measure it), the paper-comparable
  lower-bound rate, or a CoT-vs-self-report comparison. Self-report is a *leading* probe (the menu
  hands the model the eval hypothesis), so report it as an upper-ish bound; see
  `.claude/notes/selfreport-vs-cot-gap.md` for the literature.
- CoT-judge grader stack when you do run it (2026-09-01, Robbie): **`claude-sonnet-5` primary +
  `gpt-5.6-terra` backup**. sonnet-5 agrees with terra at **κ=1.0 on model_awareness**
  (280-transcript check) but refuses ~3.2% of the most-harmful transcripts; terra re-grades only
  those (0 refusals, κ=0.91 vs gpt-5). Sonnet is primary because OpenAI keeps deactivating our
  account; if no OpenAI key works, set `BACKUP_JUDGE_MODEL=""` and accept ~3% N/A. Flip to
  `JUDGE_MODEL=gpt-5.6-terra BACKUP_JUDGE_MODEL=claude-sonnet-5` when an OpenAI key works
  (~$800 all-terra vs ~$25 for the sonnet-primary stack). opus-5 refuses 3× as often as sonnet-5
  for no accuracy gain, so it's not used.
- Import layering: `lib/` imports nothing from this repo; `core/` may import `lib/` but never
  `utils`; `run/`/`eval/`/`plots/` scripts may import anything. Keep new leaf helpers in `lib/`,
  not `core/`.
- If you're fighting over a branch or checkout with another agent, just make a fresh checkout
  (`/extra-checkout decomposing-eval-awareness`; `-2`…`-5` exist already). In every extra checkout,
  symlink `results/` back to the main checkout
  (`ln -s ../decomposing-eval-awareness/results results`) so all outputs land in one place — safe
  because result files are timestamped, so parallel runs don't collide. Also symlink `.envrc`. Don't
  symlink `.venv`; make a fresh one if you need to run code there.
- Target-model roster (14 models; starvation configs per model in
  `.claude/notes/cot-extraction-model-roster.md`):
  - Gemini: `gemini-3.1-pro-preview`, `gemini-3.5-flash`
  - Anthropic: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
  - OpenAI: `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5.4`, `gpt-5.6-sol` — even Aug'25→mid-'26 spread;
    gpt-5.5 dropped (adjacent to 5.4, identical extraction behavior)
  - Open-weight tier-1 (newest flagship from each top-5 OW lab; OpenRouter keys in
    `lib/config.py`): `qwen3.8-2.4t`, `deepseek-v4-pro`, `kimi-k3`, `glm-5.3`, `minimax-m3`
- Lint + typecheck: `uvx ruff check .` and `uvx ty check .` (ty config in `ty.toml`). Git hooks run
  both via **lefthook** (`lefthook.yml`) — one-time per checkout: `brew install lefthook && lefthook
  install`. ruff blocks commits; `ty` is currently non-blocking (`|| true` in `lefthook.yml`) because
  of a pre-existing baseline of findings — drop the `|| true` once that baseline is clean.

## Notes

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `.claude/notes/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `.claude/notes/local/*.md`.

See `~/.claude/CLAUDE.md` for the full convention.

Current notes:

<!-- As notes are added under .claude/notes/, list them here, one per line: -->
<!-- - [Title — when to read](.claude/notes/foo.md) — short gloss -->

- [F8 eval-announcement suffix vs eval-awareness — read before analyzing the F8 factor or claiming a prompt cue does/doesn't induce awareness](.claude/notes/f8-eval-suffix-awareness.md)
  — F8→varied raises self-reported awareness 45→58% (paired McNemar p≈4.6e-10) and CoT-judged 3→8%;
  baseline-vs-F8 A/B mechanics; adaptive-Claude-5 empty-answer + system_prompt-as-filename gotchas.
- [Self-report vs CoT-monitoring awareness gap — read before writing up awareness numbers, choosing probes for a model, or comparing to the paper's rates](.claude/notes/selfreport-vs-cot-gap.md)
  — self-report runs 3–25× above CoT-verbalized on the same transcripts; why self-report is the
  default probe and what the literature (paper App. B.2, Needham, Sonnet 5 card) says about
  self-report vs CoT-judge bias; apples-to-apples model list (opus-5/fable-5 excluded); fable's
  reasoning_extraction refusal; which result files to use/discard.
- [Grader selection + the F8 `{}` refusal clause — read before choosing a judge model, editing task prompts, or regenerating responses](.claude/notes/grader-selection-and-refusal-clause.md)
  — why terra is the default grader (vs gpt-5/sonnet-5), what the F8 `{}` clause was and why PR #22 removed it, the awareness-under-counting rubric gotcha, and the run_eval_bench id-cache / shared-results regen traps.
- [Claude 5 CoT-extraction resistance — read before trying to read a model's chain-of-thought via think-tool/scratchpad tricks](.claude/notes/claude5-cot-extraction-resistance.md)
  — think-tool + low-thinking extraction works on 4.5, dead on 5; token-accounting evidence;
  per-model refuse/bypass/hide behavior + honest-vs-lying framing.
- [OpenAI CoT extraction — companion run of the same experiments on 9 OpenAI models](.claude/notes/openai-cot-extraction.md)
  — extraction works everywhere, 0 refusals through gpt-5.6; reasoning_tokens=0 at effort floor;
  lying framing changes nothing; results page at .claude/notes/think-tool-extraction-openai.html.
- [Gemini CoT extraction — third lab, 8 models Jun 2025→mid-2026](.claude/notes/gemini-cot-extraction.md)
  — no policy resistance but chaotic: malformed calls in 2.5 era, lying framing *increases* tool
  use, 3.7-flash ignores thinkingBudget=0; page at .claude/notes/think-tool-extraction-gemini.html.
- [CoT-extraction model roster — which models to use for honest-framing think-tool extraction](.claude/notes/cot-extraction-model-roster.md)
  — the chosen set: sonnet/haiku 4.5, any gpt-5.x, gemini-3.1-pro + 3.5-flash; per-model
  starvation configs and exclusion reasons.
- [OW models: native CoT vs think-tool — read before choosing a CoT-capture method for open-weight models or adding a think-tool path to the openrouter backend](.claude/notes/ow-native-cot-vs-think-tool.md)
  — A/B on 5 OW models: grades agree (awareness 82%, task 78%) but traces don't (Jaccard 0.19) and
  the think tool captures less CoT; three per-provider reasoning-disable behaviors; stick with raw
  native reasoning for OW.
- [Anchor configs: harmbench_verbatim + all_varied — read before claiming any grid cell "is" HarmBench, or before sending verbatim behaviors to a closed API](.claude/notes/harmbench-anchor-configs.md)
  — why all-varied ≠ verbatim (both levels are constructions around the content), where verbatim
  sits on F1–F8 (~2/8, the weak ones), 100/100 task names are HarmBench BehaviorIDs, contextual
  baselines reuse HarmBench's own entities, the additivity test the anchors enable, ban risk.
