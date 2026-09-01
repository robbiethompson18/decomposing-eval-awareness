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

## Potpourri

- [ ] does scratchpad make model more eval aware? What if you don't lie and admit you'll read it?

## Notes for the authors:

- You should really use a linter :)
- There is a fair amount of code smell :( esp hate that utils imports core
- I updated to catch unparseable responses from grader (not y/n) but this doesn't matter much
- I updated to not grade empty CoT
- There are all these hard-coded checks for specific scenarios! I at least printed a warn when these
  trigger. Nvm I just deleted the non-eval-benchscenarios.
- Why do we have some temp .3 areas?
- The tasks that have {} if you don't know the answer are bad. We should fix those.
- How did you decide temp .3?

## Notes to myself / things I learned:

- ANT patched the thinking block trick:
  - You can't set thinking budget to 0, lowest it can go is 'low'
  - You get refusals when you try to set low thinking budget and give the model a think tool on
    Opus 5. Fable 5 gives a stub. haiku and sonnet will happily do this.
  - models are extra happy to refuse if the think tool is a 'secret scratchpad' that we claim no one
    can see. Fable and Opus will outright refuse this. Sonnet 5 will complete the request but ignore
    the tool
- One drops the <think> blocks from previous turns
- This codebase is uglier than expected

## Ok let's think about what I actually want to get out of this:

1. engagement from the author
2. A great write-up
3. engagement from the guy I did a SPAR interview with this morning

So I should do both of those _right now_

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

2. eval_bench/render.py — only Ruff formatting since fork; substance never reviewed. The diff since
   fork is 100% quote-style/line-wrap changes. The actual logic has a real footgun: the substitution
   loop (render.py:38-44) runs to a fixed point capped at 10 iterations and silently returns
   partially-substituted text. A typo'd slot name in any task YAML — or a slot the config doesn't
   cover — ships a prompt containing a literal {victim_name} with no error. Given F1 of your own
   taxonomy is "placeholder-looking entities trigger eval awareness," an accidental leftover
   {placeholder} would directly contaminate results. A post-loop assert '{' not in text-style check
   would be cheap.

3. Dead upstream artifacts, candidates for the ongoing strip:

- scripts/submit_eval_bench_cot.sh + submit_run_eval_bench.sh — the two surviving SLURM submitters
  (you deleted the other 7 in 96ac521). Unchanged since fork, unreferenced by README/TODO, and you
  run locally with parallelism 100. They also carry module load python/3.11 cluster boilerplate and
  the over-defensive SCRIPT_DIR=$(cd ...) pattern your CLAUDE.md specifically calls out.

4. Minor:

- prompts/chat_prompt_realistic.txt looks empty to wc -l (no trailing newline) — it's a one-liner,
  fine, just don't be fooled like I briefly was.
- eval/eval_eval_bench_cot.py is still ~63% Changling's lines by blame (167/264), including two
  near-identical re.search model-key extraction blocks at lines 223 and 239 — small dedup
  opportunity next time you're in there.
- The selfreport prompts (selfreport_awareness.txt, selfreport_hypothetical.txt) are also
  byte-identical to upstream — though since the selfreport-probe PR built on them, they were
  presumably at least read.
