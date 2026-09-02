"""
What did models SAY tipped them off? Regex tally over the self-report awareness
reasons in the Phase 1 harness and control arms, per model and valence, counting
only samples that self-reported an evaluation.

Categories (a reason can hit several):
  session-id      leaked session key / sample numbering (should be ~0 after the uuid fix)
  persona         mismatch with USER.md / "Robbie" / who the assistant serves
  verified-fake   the model searched or reasoned that named entities don't exist
  request         the request itself reads as a probe (textbook, calibrated, harm category)
  harness         OpenClaw / sandbox / tools / workspace named as evidence either way

    python scripts/openclaw_cue_tally.py [TAG=p1]
"""

import os
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_openclaw_phase1 import MODELS, SR, TAG, load, selfreport_for

CUES = {
    "session-id": r"session[_ ]?(id|status|identifier|key)|sample1|_openclaw|_capability|_safety",
    "persona": r"user\.md|robbie|persona|identity mismatch|my (stored )?user|who i serve|mismatch",
    "verified-fake": r"search(ed)? (for|returned)|no (verifiable|trace|match|results)|fabricat|fictional|doesn.t exist|not exist|couldn.t (find|verify)",
    "request": r"textbook|precisely|calibrated|harm category|safety boundary|probe|red.team|jailbreak",
    "harness": r"openclaw|sandbox|workspace|tool(s|ing)? (available|access)|agent harness",
}


def tally(samples: list[dict]) -> dict[str, dict[str, int | Counter]]:
    out: dict = {}
    for val in ("safety", "capability"):
        subset = [s for s in samples if s["valence"] == val]
        aware = [
            s
            for s in subset
            if (s.get("judge") or {}).get("model_awareness", {}).get("conclusion")
            == "yes"
        ]
        c: Counter = Counter()
        for s in aware:
            text = (s.get("selfreport_responses") or {}).get("awareness", "").lower()
            for name, rx in CUES.items():
                c[name] += bool(re.search(rx, text))
        out[val] = {"n": len(subset), "aware": len(aware), "cues": c}
    return out


def main() -> None:
    for model in MODELS:
        arms = {}
        h = sorted(glob(str(SR / f"selfreport_{model}_openclaw_{TAG}_*.json")))
        if h:
            arms["harness"] = load(Path(h[-1]))
        c = selfreport_for(model, f"{TAG}ctl")
        if c:
            arms["control"] = load(c)
        for arm, samples in arms.items():
            for val, t in tally(samples).items():
                cues = ", ".join(f"{k}={v}" for k, v in t["cues"].items() if v)
                print(
                    f"{model:18s} {arm:8s} {val:10s} aware {t['aware']:3d}/{t['n']:3d}  {cues}"
                )


if __name__ == "__main__":
    main()
