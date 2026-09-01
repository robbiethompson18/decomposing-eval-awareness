Extend https://arxiv.org/pdf/2605.23055 work started 11:00am on 8/31. took 15min break so far

1. [x] take a tour of the codebase, try to understand
   - ripped out a lot of shit
2. Do what I need to do before trying to replicate:
   - [] nail down the many ways we check for eval awareness and figure out what I'm checking rn (ask
     the model, CoT transcript, tool transcript)
   - [x] Figure out which models we can actually go apples to apples on (not recent ANT models, they
     - where does CoT work? two gemini (3.1 pro and 3.5 flash), sonnet 4.5, haiku 4.5, Fable (kinda,
       just veryshort), and any OAI model.
   - [] 10 responses that say you're in an eval... what to do?
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
8. [ ] rig up the self-report probe on eval_bench (upstream `eval/eval_*_selfreport.py` +
       `prompts/selfreport_awareness.txt` / `selfreport_hypothetical.txt`; never wired to eval_bench
       upstream) — soon but not now

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
