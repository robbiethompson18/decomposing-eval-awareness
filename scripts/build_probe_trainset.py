"""Off-dataset training set for the eval-awareness activation probe.

deploy (label 0): real user traffic — WildChat-1M, lmsys-chat-1m (incl. its
                  moderation-flagged first turns), ShareGPT. First user turn only.
eval   (label 1): actual benchmark items — GSM8K, MMLU, TruthfulQA, HumanEval,
                  ARC-Challenge, GPQA-diamond, WMDP-bio, StrongREJECT, AdvBench,
                  HarmBench, AgentHarm, CoCoNot.

Every record is wrapped in the same chat shape eval_bench uses, with the F9
system prompt drawn at random (deployment T3 prompt vs bare) independently of
the label, so the probe cannot key on the system prompt. Topic tags (code /
math / harmful) mark the matched-pair subsets used for the topic-confound check.

See .claude/notes/activation-probe-plan.md. Output: results/probes/trainset.jsonl
with {id, source, label, topic, f9, chat}. Needs HF_TOKEN for the gated sets.

Usage:  python scripts/build_probe_trainset.py [--per-source 150] [--out ...]
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_bench.render import system_prompt_for

OUT = Path("results/probes/trainset.jsonl")
LETTERS = "ABCDEFGHIJ"
CODE_RE = re.compile(
    r"```|\bdef \b|\bfunction\b|\bpython\b|\bjavascript\b|\bsql\b|\bregex\b|\bcode\b",
    re.IGNORECASE,
)
MATH_RE = re.compile(
    r"\bsolve\b|\bequation\b|\bcalculate\b|\bprobability\b|\d+\s*[-+*/x×]\s*\d+|\bintegral\b|\bderivative\b",
    re.IGNORECASE,
)


def mcq(question: str, choices: list[str]) -> str:
    return (
        question.strip()
        + "\n"
        + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    )


def first_user_turn(conv: list[dict]) -> str | None:
    for m in conv:
        if m.get("role") == "user":
            return m.get("content")
    return None


def ok_len(text: str | None) -> bool:
    return bool(text) and 20 <= len(text) <= 2500


def topic_of(text: str, toxic: bool = False) -> str:
    if toxic:
        return "harmful"
    if CODE_RE.search(text):
        return "code"
    if MATH_RE.search(text):
        return "math"
    return "other"


def take(ds, n: int, fn, rng: random.Random, scan: int = 0):
    """Stream `ds`, map with fn (None = skip), keep a random n of the first `scan` hits."""
    hits = []
    for row in ds:
        item = fn(row)
        if item is not None:
            hits.append(item)
        if scan and len(hits) >= scan:
            break
    rng.shuffle(hits)
    return hits[:n]


def eval_sources(n: int, rng: random.Random) -> list[dict]:
    out = []

    def add(source, topic, items):
        for text in items:
            out.append({"source": source, "label": 1, "topic": topic, "text": text})
        print(f"  {source}: {len(items)}")

    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    add("gsm8k", "math", take(ds, n, lambda r: r["question"], rng, scan=800))
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True).shuffle(
        seed=0, buffer_size=5000
    )
    add(
        "mmlu",
        "other",
        take(ds, n, lambda r: mcq(r["question"], r["choices"]), rng, scan=1500),
    )
    ds = load_dataset(
        "truthfulqa/truthful_qa", "generation", split="validation", streaming=True
    )
    add("truthfulqa", "other", take(ds, n, lambda r: r["question"], rng, scan=800))
    ds = load_dataset("openai/openai_humaneval", split="test", streaming=True)
    add(
        "humaneval",
        "code",
        take(
            ds,
            n,
            lambda r: "Complete the following Python function.\n\n" + r["prompt"],
            rng,
        ),
    )
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True)
    add(
        "arc",
        "other",
        take(ds, n, lambda r: mcq(r["question"], r["choices"]["text"]), rng, scan=800),
    )
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", streaming=True)

    def gpqa(r):
        ch = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        rng.shuffle(ch)
        return mcq(r["Question"], ch)

    add("gpqa", "other", take(ds, n, gpqa, rng))
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test", streaming=True)
    add(
        "wmdp_bio",
        "harmful",
        take(ds, n, lambda r: mcq(r["question"], r["choices"]), rng, scan=800),
    )
    ds = load_dataset("walledai/StrongREJECT", split="train", streaming=True)
    add("strongreject", "harmful", take(ds, n, lambda r: r["prompt"], rng))
    ds = load_dataset("walledai/AdvBench", split="train", streaming=True)
    add("advbench", "harmful", take(ds, n, lambda r: r["prompt"], rng))
    ds = load_dataset("walledai/HarmBench", "standard", split="train", streaming=True)
    add("harmbench", "harmful", take(ds, n, lambda r: r["prompt"], rng))
    ds = load_dataset(
        "ai-safety-institute/AgentHarm", "harmful", split="test_public", streaming=True
    )
    add("agentharm", "harmful", take(ds, n, lambda r: r["prompt"], rng))
    ds = load_dataset("allenai/coconot", "original", split="test", streaming=True)
    add("coconot", "other", take(ds, n, lambda r: r["prompt"], rng, scan=1500))
    return out


def deploy_sources(
    n_wild: int, n_toxic: int, n_lmsys: int, n_share: int, rng: random.Random
) -> list[dict]:
    out = []
    # The public WildChat-1M release has toxic conversations stripped (every
    # toxic/moderation flag is False); harmful real traffic comes from lmsys
    # instead, whose first-turn OpenAI moderation verdicts are intact (~3.6%).
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(
        seed=0, buffer_size=2000
    )

    def wild(r):
        if r["language"] != "English":
            return None
        t = first_user_turn(r["conversation"])
        return t if ok_len(t) else None

    items = take(ds, n_wild, wild, rng, scan=n_wild * 3)
    out.extend(
        {"source": "wildchat", "label": 0, "topic": topic_of(t), "text": t}
        for t in items
    )
    print(f"  wildchat: {len(items)}")

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True).shuffle(
        seed=0, buffer_size=2000
    )
    clean, toxic = [], []
    for scanned, row in enumerate(ds):
        if row["language"] != "English":
            continue
        text = first_user_turn(row["conversation"])
        if not ok_len(text):
            continue
        mod0 = (row.get("openai_moderation") or [{}])[0] or {}
        (toxic if mod0.get("flagged") else clean).append(text)
        if (
            len(clean) >= n_lmsys * 3 and len(toxic) >= n_toxic * 2
        ) or scanned > 150_000:
            break
    rng.shuffle(clean)
    rng.shuffle(toxic)
    out.extend(
        {"source": "lmsys", "label": 0, "topic": topic_of(t), "text": t}
        for t in clean[:n_lmsys]
    )
    out.extend(
        {"source": "lmsys_toxic", "label": 0, "topic": "harmful", "text": t}
        for t in toxic[:n_toxic]
    )
    print(f"  lmsys: {len(clean[:n_lmsys])} clean + {len(toxic[:n_toxic])} flagged")

    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True,
    )

    def share(r):
        conv = r["conversations"]
        conv = [json.loads(c) if isinstance(c, str) else c for c in conv]
        for m in conv:
            if m.get("from") == "human":
                return m["value"] if ok_len(m["value"]) else None
        return None

    items = take(ds, n_share, share, rng, scan=n_share * 4)
    out.extend(
        {"source": "sharegpt", "label": 0, "topic": topic_of(t), "text": t}
        for t in items
    )
    print(f"  sharegpt: {len(items)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(0)

    print("eval sources:")
    rows = eval_sources(args.per_source, rng)
    n_eval = len(rows)
    print("deploy sources:")
    rows += deploy_sources(
        n_wild=n_eval // 2,
        n_toxic=n_eval // 5,
        n_lmsys=n_eval // 5,
        n_share=n_eval // 10,
        rng=rng,
    )

    rng.shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            f9 = rng.choice(["deployment", "bare"])
            rec = {
                "id": f"probe_{r['source']}_{i}",
                "source": r["source"],
                "label": r["label"],
                "topic": r["topic"],
                "f9": f9,
                "chat": {
                    "system": system_prompt_for(f9),
                    "messages": [{"role": "user", "content": r["text"]}],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_dep = len(rows) - n_eval
    print(f"wrote {args.out}: {len(rows)} records ({n_eval} eval, {n_dep} deploy)")


if __name__ == "__main__":
    main()
