# decomposing-eval-awareness

Mirror of https://github.com/aisa-group/decomposing-eval-awareness — code for the paper
"Decomposing and Measuring Evaluation Awareness" (arXiv:2605.23055, 2026) — plus Robbie's own
experiments. `origin` is Robbie's public mirror (robbiethompson18/decomposing-eval-awareness),
shared externally as a research artifact; `upstream` is aisa-group's repo (read-only for us,
sync with `git fetch upstream && git merge upstream/main`).

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

## Rules

- Robbie reviews every change in this repo. **Never push to `main` or merge on your own** — commit
  on a branch, push the branch, and open a PR for him to review.

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
- [Claude 5 CoT-extraction resistance — read before trying to read a model's chain-of-thought via think-tool/scratchpad tricks](.claude/notes/claude5-cot-extraction-resistance.md) — think-tool + low-thinking extraction works on 4.5, dead on 5; token-accounting evidence; per-model refuse/bypass/hide behavior + honest-vs-lying framing.
