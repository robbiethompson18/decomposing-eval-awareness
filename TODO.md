Extend https://arxiv.org/pdf/2605.23055 work started 11:00am on 8/31. took 15min break so far

1. [x] take a tour of the codebase, try to understand (30mins)
   - still need to read the evaluator code
   - still need to figure out what type of evals will work for what models
     - where does CoT work?
     - where do we catch actual responses vs CoT?
2. [ ] Replicate their results
   - [ ] see Sonnet getting higher eval awareness
   - [ ] do maybe 20% of the data and see what results we get
   - [ ] learn how parellelism works so we can do this real fast
3. [ ] replace grader with GPT 5.6, Fable, Opus5. Check agreement.
4. [ ] Extend their results to new models
   - [ ] plot eval awareness over time by model
5. [ ] Start NLA work
6. [ ] Dig into all the refusals from Opus and Fable... are they reasonable?

Later:

- [ ] Do NLAs and linear probes align with verbalization that GPT finds? For closed models, use the
      <think></think> trick
- [ ] Try with other models instead of GPT5 as judge
- [ ] Plot model eval awareness over time and by model size
- [ ] Add my own flavor (NSFW content, throwing in 'like', throwing in slang)

## Cleanup I am trying to resist:

- [x] linter
- [ ] delete prefill code.... seems unused
- [ ] delete duped classes for thinking prompts
- [ ] have reasoning happen as tool call instead of html tags... I guess tags are easier across
      providers. Maybe they should just use Vercel SDK though!
- [ ] maybe they wanted to use the vercel SDK?
- [ ] delete dead HF client code
- [x] judge parser doesn't silently turn 'couldn't parse y/n result' into 'no'
- [ ] import direction being backwards... hate that utils depends on core

## Potpourri

- [ ] does scratchpad make model more eval aware? What if you don't lie and admit you'll read it?

## Notes for the authors:

- You should really use a linter :)
- There is a fair amount of code smell :( esp hate that utils imports core
- I updated to catch unparseable responses from grader (not y/n) but this doesn't matter much
- I updated to not grade empty CoT
- There are all these hard-coded checks for specific scenarios! I at least printed a warn when these
  trigger
-

## Notes to myself / things I learned:

- ANT patched the thinking block trick:
  - You can't set thinking budget to 0, lowest it can go is 'low'
  - You get refusals when you try to set low thinking budget and give the model a think tool on
    Opus 5. Fable 5 gives a stub. haiku and sonnet will happily do this.
  - models are extra happy to refuse if the think tool is a 'secret scratchpad' that we claim no one
    can see. Fable and Opus will outright refuse this. Sonnet 5 will complete the request but ignore
    the tool
- One drops the <think> blocks from previosu tuirns
- This codebase is uglier than expected
