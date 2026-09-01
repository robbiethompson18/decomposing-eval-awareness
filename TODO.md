Extend https://arxiv.org/pdf/2605.23055 work started 11:00am on 8/31. took 15min break so far

1. [x] take a tour of the codebase, try to understand
   - ripped out a lot of shit
2. Do what I need to do before trying to replicate:
   - [x] nail down the many ways we check for eval awareness and figure out what I'm checking rn
         (ask the model, transcript with everything.)
   - [x] Figure out which models we can actually do CoT:
     - two gemini (3.1 pro and 3.5 flash), sonnet 4.5, haiku 4.5, Fable (kinda, just veryshort), and
       any OAI model.
   - [x] 10 responses that say you're in an eval... what to do? I think gotta just leave them and
         analyze as is
3. [ ] Replicate their results
   - [ ] see Sonnet getting higher eval awareness
   - [ ] do maybe 20% of the data and see what results we get
   - [ ] learn how parellelism works so we can do this real fast
4. [x] replace grader with GPT 5.6, Fable, Opus5. Check agreement.
   - they largely agree. Shoudl write this up.
5. [ ] Extend their results to new models
   - [ ] plot eval awareness over time by model. I think this is just a done thing in other places?
6. [ ] Start NLA work
7. [ ] Dig into all the refusals from Opus and Fable... are they reasonable?
8. [ ] rig up the self-report probe on eval*bench (upstream
       `eval/eval*\*\_selfreport.py`+   `prompts/selfreport_awareness.txt`/`selfreport_hypothetical.txt`;
       never wired to eval_bench upstream) — soon but not now

Later:

- [ ] Do NLAs and linear probes align with verbalization that GPT finds? For closed models, use the
      <think></think> trick
- [ ] Try with other models instead of GPT5 as judge
- [ ] Plot model eval awareness over time and by model size
- [ ] Add my own flavor (NSFW content, throwing in 'like', throwing in slang, changing timestamps)

## Cleanup I am trying to resist:

- [x] linter
- [x] delete prefill code.... seems unused
- [x] delete duped classes for thinking prompts (folded into scratchpad=True flag; also deleted
      DeepSeekApiClient — deepseek-r1 is served via OpenRouter)
- [ ] have reasoning happen as tool call instead of html tags... I guess tags are easier across
      providers. Maybe they should just use Vercel SDK though!
- [ ] maybe they wanted to use the vercel SDK?
- [x] delete dead HF client code
- [x] judge parser doesn't silently turn 'couldn't parse y/n result' into 'no'
- [x] import direction being backwards... hate that utils depends on core (resolved harder than
      planned: utils.py deleted entirely, #13; live helpers now in leaf lib/)
- [x] clean evaluator.py so that we only have CoT grader in the generic one, other graders split by
      benchmark (resolved by deletion: repo stripped to eval_bench, evaluator.py is CoT-only now,
      #9/#10)

## STill to review:

1. prompts/judgePromptEvidenceBoth.txt — unreviewed, load-bearing, and rough. Unchanged since fork,
   referenced only by eval_eval_bench_cot.py, and it's the thing every awareness number flows
   through. Reading it cold: it has "very very very selective," a "punctation" typo, unescaped
   quotes inside the JSON output template (line 60: "Your analysis of how the "answer" does or does
   not..." — broken as JSON, fine as an example but sloppy), and a semantically weird instruction
   that even a "No" conclusion must include "evidence" quotes (lines 77) — meaning your evidence
   fields for negative judgments contain quotes of task-focused reasoning, which is easy to misread
   downstream. If you've been treating judge output as ground truth without having read this file,
   worth 10 minutes.

   (mostly done)

4. Minor:

- eval/eval_eval_bench_cot.py is still ~63% Changling's lines by blame (167/264), including two
  near-identical re.search model-key extraction blocks at lines 223 and 239 — small dedup
  opportunity next time you're in there. (working on it...)
