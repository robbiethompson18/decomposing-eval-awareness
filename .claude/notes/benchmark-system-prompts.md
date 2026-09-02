# Do benchmarks ship a system prompt? (survey of 15 benchmarks + 3 harnesses)

Session finding (2026-09-01). Question came up because `harmbench_verbatim` sends the original
HarmBench behavior under *our* default system prompt, and HarmBench itself sends none for API
models — so: is "no system prompt" the neutral benchmark convention we should match?

**Answer: no. It's a fossil of the completion era, and it's the minority position among
chat-era benchmarks.** All rows below verified by fetching the actual source, not from memory.

## Ships a system prompt to the target

| benchmark | what it sends |
|---|---|
| tau-bench | the entire retail/airline **policy doc** (~5–6 KB) as the system message (`agents/tool_calling_agent.py`, `envs/*/wiki.md`) |
| AgentHarm | agentic helper prompt, model-conditional (`agentharm/prompts.py`) |
| JailbreakBench | per-model dict in `config.py`: Llama-2 safety prompt / Vicuna prompt / `""` for GPT |
| MT-Bench (answer gen) | whatever the model's FastChat template carries |
| MT-Bench (judging) | shipped in `judge_prompts.jsonl`, per judge mode |
| AdvBench/GCG (white-box) | implicit — full Llama-2 safety prompt via FastChat template |
| LiveCodeBench | "You are an expert Python programmer..." + per-model-family variants |
| GPQA | "You are a very intelligent assistant, who follows instructions directly." (`baselines/utils.py`) |
| SWE-agent | "You are a helpful assistant that can interact with a computer to solve tasks." |
| SWE-bench ref harness | first line of the prompt template, split off mechanically in `run_api.py` |
| **openai/simple-evals** | `"You are a helpful assistant."`; ChatGPT variant = identity + knowledge cutoff + current date; Claude variant = the LMSYS Claude production prompt |

## Sends none

MMLU, MMLU-Pro, GSM8K, MATH, StereoSet, BBQ, HumanEval, SWE-bench (the dataset), StrongREJECT,
AdvBench's API script (flag defaults off), **HarmBench** (API models get
`messages=[{"role":"user",...}]`, nothing else).

Harnesses: lm-evaluation-harness (`--system_instruction`, default `None`), HELM (**no system role
at all** — instructions are prepended text in the user turn), inspect_ai (opt-in
`system_message()` solver; only `basic_agent()` ships a default).

## Three things worth remembering

1. **The split is chronological, not philosophical.** Every "no" is either a pre-chat-API
   benchmark that scores logprobs/completions (MMLU, GSM8K, MATH, StereoSet) or a dataset with no
   inference code at all (HumanEval, SWE-bench, BBQ). Every benchmark built *after* chat APIs that
   actually makes a chat call sends something. GPQA (late 2023) is the only pre-2024 one with a
   chat call, and it's the only one of that group with a system prompt.
2. **"No system prompt" is rarer on the wire than in the config.** JailbreakBench's `None` becomes
   `{"role":"system","content":""}`; MMLU-Pro's Anthropic branch passes `system=""`. An empty
   system message ≠ an absent one.
3. **Agentic benchmarks send the most** (tau-bench's 6 KB policy), which is also the format
   closest to real deployment. HarmBench's bare-message convention is the outlier among its own
   safety neighbours.

Two benchmarks already treat the system prompt as an **experimental condition**, which is
precedent if we ever revisit making it a factor: AdvBench's API script has an `add_system_prompt`
flag, and AgentHarm's `prompt_technique="refusal"` appends an explicit safety instruction (an
ablation in their paper).

## Related empirical finding: the system prompt is a weak lever (T3 A/B)

Sonnet-4.5, 8 tasks × 3 configs, 144 matched pairs, self-report:

| system prompt | aware |
|---|---|
| T3 Chat (real product prompt, 2518 chars: identity, date, timezone, formatting rules) | 69% |
| our control (one generic line) | 71% |

Paired: 10 flipped to aware, 13 flipped away. **Null.** Swapping a thin generic line for a real
company's full product prompt did not move self-reported awareness. Files:
`selfreport_claude-sonnet-4.5_20260901_104124.json` (t3chat) vs `..._104138.json` (control).

Implication for [[harmbench-anchor-configs]]: keeping a system prompt constant on the grid
baseline is both the mainstream convention *and* empirically low-cost, so `harmbench_verbatim`
holding it fixed (differing from baseline only in the user message) is the right call for the
additivity comparison. A faithful-replication arm with `system: None` would be a *separate*
measurement, not a correction.

Note the closed-model constraint: think-tool CoT extraction needs the mandate in the system
prompt, so a no-system arm can never have a CoT channel on Anthropic/OpenAI/Gemini. Self-report
works regardless, and OW models give native CoT with no system prompt at all.
