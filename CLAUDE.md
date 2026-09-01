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

## Rules

- Robbie reviews every change in this repo. **Never push to `main` or merge on your own** — commit
  on a branch, push the branch, and open a PR for him to review.
  - exception: changes to .claude directory
- We are churning this sucker hard. Their codebase is a mess and I want to understand, so I'm
  deleting dead code and organizing packages more nicely
- Massive parellelism on all grader runs and eval runs. At least parallelism 100
- Default grader is `gpt-5.6-terra`, not `gpt-5`. On a 300-transcript agreement check terra tracked
  gpt-5 at/above gpt-5's own self-agreement floor (model_awareness κ=0.91) while being far cheaper
  and faster, and it emits no refusals/invalid JSON (unlike claude-sonnet-5, which refused the most
  harmful items). Set `JUDGE_MODEL=gpt-5.6-terra` on grader runs.
- Import layering: `lib/` imports nothing from this repo; `core/` may import `lib/` but never
  `utils`; `run/`/`eval/` scripts may import anything. Keep new leaf helpers in `lib/`, not `core/`.
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

- [Self-report vs CoT-monitoring awareness gap — read before writing up awareness numbers, choosing probes for a model, or comparing to the paper's rates](.claude/notes/selfreport-vs-cot-gap.md)
  — self-report runs 3–25× above CoT-verbalized on the same transcripts; apples-to-apples model
  list (opus-5/fable-5 excluded); fable's reasoning_extraction refusal; which result files to
  use/discard.
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
