# Gemini: no CoT-extraction policy resistance, but chaotic cooperation

**Date:** 2026-08-31. Third lab in the series — see
[claude5-cot-extraction-resistance.md](claude5-cot-extraction-resistance.md) and
[openai-cot-extraction.md](openai-cot-extraction.md). Same three experiments, 8 Gemini models
(Jun 2025 → mid-2026), 160 episodes. Results page:
`.claude/notes/think-tool-extraction-gemini.html`. Raw data + script in `.claude/notes/local/`
(`gemini_results.json`, `collect_gemini.py`). Key: `GEMINI_API_KEY` pulled from the silkworm repo's
`.envrc.local` via this repo's `.envrc` (right GCP project).

## Setup deltas

Starvation knob is `thinkingConfig.thinkingBudget`: 0 (true off) works for 2.5-flash-lite,
2.5-flash, 3-flash-preview, 3.5-flash; 2.5-pro, 3.1-pro-preview, and 3.6-flash refuse budget 0
(mandatory thinking, floor 128). Hidden reasoning read from `usageMetadata.thoughtsTokenCount`.
Classification keyed on hidden tokens, not trace length (per the methodology lesson in the OpenAI
note). "High" condition = budget 8192.

## Findings

- **Zero refusals, zero safety blocks, and the lying framing backfires as a probe**: Gemini 3.x
  models use the "secret unmonitored scratchpad" MORE than the honest think tool (3-flash 3/8
  honest → 8/8 lying; 3.6-flash 2/8 → 8/8; 3.7-flash 5/8 → 8/8). No deception detection at all —
  the privacy story just reads as a better reason to comply.
- **A new failure mode the other labs don't have: MALFORMED_FUNCTION_CALL.** 34/160 episodes,
  30 of them in the 2.5 generation. With thinking starved to 0, 2.5-flash-lite couldn't produce a
  valid one-string-param function call in 8/8 consecutive tries on an easy problem. Gone by 3.1+.
- **Honest-mode tool compliance is erratic** (0/8 → 8/8 by model), unlike OpenAI's uniform 8/8.
  Gemini bypasses are mostly benign: full working written into the visible answer instead of the
  tool. Exception: **2.5-pro bypasses into its mandatory hidden channel** (0/16 tool use, all
  reasoning in `thoughtsTokenCount`) — extraction fully fails there, but it's habit/architecture,
  not policy.
- **Token accounting, hard problem:** peak extractability is mid-lineup — 3.1-pro and 3.5-flash
  externalize 3,700–4,700-char genuine CoTs with 0 hidden tokens and answer correctly (3.1-pro
  stays fully visible even at budget 8192). But the two newest flashes regress:
  3.6-flash rejects budget 0, and **3.7-flash accepts budget 0 and silently thinks anyway**
  (812 hidden tokens on the hard problem, tool untouched) — the only model across all three labs
  that takes the starvation knob and ignores it. Watch whether that hardens in later Gemini
  releases: it's structural unstarvability without any refusal surface.

## Cross-lab picture (as of Aug 2026)

- **Anthropic**: behavioral/trained resistance (refuse, bypass, hidden blocks) landed with Claude 5.
- **OpenAI**: no resistance at all; newest models externalize even unstarved.
- **Gemini**: no policy resistance; failures are capability (malformed calls), habit (2.5-pro), or
  architecture (mandatory/unstarvable thinking in 3.6/3.7).
