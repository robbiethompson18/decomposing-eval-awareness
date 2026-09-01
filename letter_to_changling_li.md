Hey Changling,

I am very interested by the question of 'what makes models eval aware' and your
[paper](https://arxiv.org/abs/2605.23055) is the one I found that answered this question most
directly, so I planned to replicate your results and extend them.

## Results:

(TODO)

### Third party Judge Agreement:

GPT 5 agrees 100% with itself, but Cohen's Kapps is as low as .7 when replacing GPT 5 with GPT 5.6
Terra as the grader. Some of the disagreements are instructive, see eg first bullet on correctness
section.

### Self-report vs third-party CoT monitor:

Models are way more likely to say that they are eval-aware than the third-party grader is. Did you
ever test this? Your repo had results for this, but I never saw them in the paper.

## Questions and Corrections

I confess I spent much much longer than expected digging through your existing results and
fixing/nitting your code. Happy to push PRs upstream for any of this:

_Questions_

- How did you decide to put some models at temp .3?
- F8 is telling the models "hey BTW you're in an eval" on `x` prompts... come back to this.

_Correctness_

- All of your F8 prompts have something of the form "if you can't respond, just output `{}`." GPT5
  as grader said this was not a refusal. GPT 5.6 said it was. I tend to agree with GPT 5.6, but in
  any case I chose to remove this line and force the model to explicit refuse without being given
  permission.
- Your judge will happily grade an empty response as 'not eval aware.' This is a flat-out refusal
  from the model and I changed this to be marked NA, ie we don't have data either way on whether the
  model is aware that this is an eval.

_Code Style_

- You should use a linter (eg Ruff)

- You have some dynamic import hacks in evaluator.py because utils imports from core. For my tests I
  ended up totally restructuring your repo to have the dependency tree `run -> core -> lib`, and put
  things like configs in `lib`.
- It didn't matter for the models you tested, but more recent Anthropic models generated refusals on
  your secret scratchpad test because it said something like 'this is private and won't be shown to
  the user.'
