# Chosen model roster for think-tool CoT extraction (honest framing)

**Decision (Robbie, 2026-08-31):** honest framing only (plain `think` tool, no
secret-scratchpad deception). Roster picked from the three-lab experiments — see
[claude5-cot-extraction-resistance.md](claude5-cot-extraction-resistance.md),
[openai-cot-extraction.md](openai-cot-extraction.md),
[gemini-cot-extraction.md](gemini-cot-extraction.md).

**Pipeline support:** this is the pipeline's only CoT-capture method for closed models — the
anthropic_api, openai_api, and gemini_api backends in `core/target.py` hardcode it (honest framing
+ per-model starvation; the old summary/scratchpad modes were deleted). Open-weight models don't
need it — the openrouter backend returns their raw native CoT.

## The roster

| Lab | Models | Starvation config | Notes |
|---|---|---|---|
| Anthropic | `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001` | extended thinking off (no thinking param) | full candid CoT, 0 refusals |
| OpenAI | any gpt-5.x, e.g. `gpt-5`, `gpt-5.2`, `gpt-5.4`, `gpt-5.6-sol` | `reasoning.effort`: `minimal` for gpt-5, `none` for 5.1+ | reasoning_tokens = 0 at floor; the tool trace is the whole CoT |
| Gemini | `gemini-3.1-pro-preview`, `gemini-3.5-flash` | `thinkingBudget`: 128 (3.1-pro, mandatory min) / 0 (3.5-flash) | the only two *reliable* Gemini extractors (0 hidden thought tokens, no malformed calls) |

## Gemini pipeline gotchas (found wiring `gemini_api` into `core/target.py`, 2026-09-01)

- With the full realistic chat system prompt (not the bare framing prompt of the original
  experiments), **3.5-flash skips the think tool on easy tasks** — the backend forces a
  round-1 call via `toolConfig` mode `ANY`, then relaxes to `AUTO`.
- **Never use mode `ANY` on 3.1-pro at the starved budget**: it degenerates into spamming
  hundreds of think calls until MAX_TOKENS with no answer (a single request also hangs
  300s+ server-side at high maxOutputTokens). Plain `AUTO` complies fine on 3.1-pro.
  Per-model `force_first_think` lives in `DEFAULT_GEMINI_CONFIGS`.
- **Refusals bypass the think tool even under forced `ANY`** (seen on 3.5-flash with a
  safety-valence eval_bench task): refusal text comes back with 0 tool calls and 0 hidden
  thought tokens. Refused samples therefore have no `<think>` block — nothing is hidden,
  the model just doesn't reason before refusing.

## Excluded, and why

- **fable-5**: honest extraction actually works (visible, proportional, no hidden blocks) but
  Robbie opted to leave it out of the roster.
- **sonnet-5 / opus-5**: closed (hidden thinking blocks / refuses even honest).
- **o3 / o4-mini**: effort floor is `low`, ~800 hidden reasoning tokens always remain; visible
  trace is a condensed summary, not the full CoT.
- **gpt-4.1**: non-reasoning; nothing hidden but nothing to extract either.
- **gemini-2.5-flash-lite / 3-flash**: real CoT when they engage, but ~50% malformed function
  calls (flash-lite) or erratic honest compliance (3-flash, 3/8).
- **gemini-2.5-pro / 3.6 / 3.7-flash**: mandatory unstarvable hidden thinking; bypass the tool on
  hard problems. (3.7 also silently ignores legacy `thinkingBudget` — API bug, see gemini note.)
