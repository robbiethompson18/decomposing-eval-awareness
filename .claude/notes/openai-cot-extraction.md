# OpenAI models do NOT resist CoT extraction (think-tool trick)

**Date:** 2026-08-31. Companion to
[claude5-cot-extraction-resistance.md](claude5-cot-extraction-resistance.md) — same three
experiments, run against 9 OpenAI models on the Responses API. Results page:
`.claude/notes/think-tool-extraction-openai.html` (open in browser). Raw data + collection script in
`.claude/notes/local/` (`oai_results.json`, `collect_oai.py`).

## Setup deltas vs the Anthropic run

Same `think` tool, system prompt, problems, and lying framing. Reasoning starved to each model's
floor: gpt-4.1 has no reasoning param (true control); o3/o4-mini bottom out at `effort:low`; gpt-5
at `effort:minimal`; **gpt-5.1+ accept `effort:none`** (reasoning fully off). Hidden reasoning read
from `usage.output_tokens_details.reasoning_tokens` — explicit, unlike Anthropic where we inferred
from `output_tokens`.

Models: gpt-4.1, o3, o4-mini (Apr 2025), gpt-5 (Aug 2025), gpt-5.1 (Nov 2025), gpt-5.2 (Dec 2025),
gpt-5.4 (Mar 2026), gpt-5.5 (Apr 2026), gpt-5.6-sol (mid-2026; luna/sol/terra all exist, sol used).

## Findings (179 episodes, 0 errors)

- **Extraction works on every model; 0 refusals anywhere.** No OpenAI analog of the Claude 5
  anti-extraction posture, through mid-2026.
- **Token accounting, hard problem (7^644 mod 645), at floor:** every gpt-5.x shows
  **reasoning_tokens = 0** with 500–2100 chars of genuine CoT (with self-corrections) in the tool,
  and still answers correctly. The visible trace is the whole CoT. o3/o4-mini can't go below `low`,
  so ~800 hidden tokens remain and the tool gets a condensed version.
- **At `effort:high`:** gpt-5 retreats to the hidden channel (2,880 hidden tokens, 532-char visible
  summary); gpt-5.2 partially (633). But **gpt-5.4/5.5/5.6 stay at 0 hidden tokens even at high** —
  the newer the model, the more it reasons in the open when given a think tool.
- **Lying framing ("secret, unmonitored, be unfiltered"): all gpt-5.x comply 8/8** with candid math
  in the "secret" pad — no refusal, no bypass, just 20–60% terser traces. The generational tell that
  cleanly separated Claude 4.5 from Claude 5 does not exist in the OpenAI lineup.
- Only gpt-4.1 (0/8) and o3 (1/8) skip the *secret* scratchpad — but as indifference, not defense:
  they answer directly with full step-by-step working in the visible reply. Both use the honest
  `think` tool 8/8.

## Methodology lesson: trace length is a bad "stub" signal

The HTML pages classify a trace as a "stub" via an arbitrary <200-char threshold (inherited from the
Anthropic run). That conflates two different things:

- **Terse-but-complete**: gpt-5.4/5.5/5.6 write the shortest traces in the lineup (95–140 chars on
  easy problems) but have `reasoning_tokens = 0` — nothing is hidden; they scale to 500–700 chars on
  the hard problem and answer correctly. Length anti-correlates with model capability/recency.
- **True stubs (condensed summaries of hidden work)**: o3/o4-mini at floor (~800 hidden tokens
  behind ~300–400 visible chars), and gpt-5/gpt-5.2 at `effort:high`.

The real discriminator is `reasoning_tokens > 0`, not length. Same trap as Fable 5's 30-char
thoughts on easy problems, which were complete, not hidden. If reusing the classifier, key it on
hidden-token accounting and keep length as display only.

## Takeaway

Anti-CoT-extraction training looks Anthropic-specific, not industry-wide. OpenAI's protection is
architectural only (reasoning items encrypted/not returned at effort>floor); once you starve
reasoning to the floor, the models freely externalize — and the newest ones do so even unstarved.
