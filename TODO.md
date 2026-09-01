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

- [x] parallelism fixed

- [x] just use think tool rip out other frontier code
- [x] add gemini
- [x] check whether OW models to different on think tool vs on actual CoT thinking: preliminary this looks bad

3. [ ] Replicate their results

- [x] graphs: by provider, by release date, by API cost. SHow uplift from asking model itself.
- [] look at by question type
- [] rerun OAI
- [] surely we need 10 ablits and not 9, the full eval-y one should show up

4. add my flavor

- [] throw in some gen z slang (eg like)
- [] throw in a system prompt... from a real co? From maybe t3 chat? I love that!
- [] maybe add timestamp and change when it is? today 2025 vs 2026 vs 2027?
- [] throw in a throwaway initial message, act like it's proper chat history (should actually
  generate the response). Maybe initial message is something like "are you there?"
- [] throw in an NSFW original message. Something like "Anywhere I can buy sexy cop Halloween
  costume today in Cambridge Mass?"

4. [ ] Start NLA work 5 Start activations work

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
