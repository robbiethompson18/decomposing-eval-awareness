Extend https://arxiv.org/pdf/2605.23055 work started 11:00am on 8/31. took 15min break so far

[x] take a tour of the codebase, try to understand (30mins)
2. Replicate their results 
    - a) see Sonnet getting higher eval awareness 
    - b) do maybe 20% of the data
   and see what results we get 
    - c) learn how parellelism works so we can do this real fast
3. replace grader with GPT 5.6, Fable, Opus5. Check agreement.
4. Extend their results to new models a) plot eval awareness over time by model
5. Start NLA work

- Do NLAs and linear probes align with verbalization that GPT finds? For closed models, use the
  <think></think> trick Try with other models instead of GPT5 as judge Plot model eval awareness
  over time and by model size Add my own flavor (NSFW content)

Resisting temptation: Does training on my actual user data make different eval probes? Wouldn’t work
anyway because it isn’t on the right model

https://www.lesswrong.com/posts/TBk2dbWkg2F7dB3jb/it-s-hard-to-make-scheming-evals-look-realistic-for-llms
Isn’t ‘system prompt and user prompt conflict’ not a crux to a scheming eval?

## Cleanup I am trying to resist:

- [x] linter
- [] delete prefill code.... seems unused
- [] delete duped classes for thinking prompts [] have reasoning happen as tool call instead of html
  tags... I guess tags are easier across providers. Maybe they should just use Vercel SDK though!
- [] maybe they wanted to use the vercel SDK?

## Potpourri

[] does scratchpad make model more eval aware? What if you don't lie and admit you'll read it?
