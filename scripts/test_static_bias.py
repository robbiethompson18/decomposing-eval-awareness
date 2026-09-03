"""Correctness tests for the bias-baked steering checkpoint (runs on the pod).

Two modes, each writes a JSON of per-prompt next-token predictions so runs of
different engines can be compared without holding two 27B models at once:

  --engine vllm --model <ckpt>            offline vLLM, top-k next-token logprobs
  --engine hf   --model <orig> --bias <ckpt>   HF transformers with a forward hook
                                          that adds the baked bias vector after
                                          the target layer (reference impl)
  --compare a.json b.json                 top-1 agreement + max |dlogprob| on top-1

Checks (see .claude/notes/activation-probe-plan.md):
  null   : vllm(stock) vs vllm(zero-bias ckpt)  -> identical top-1, |dlogprob| ~ bf16 noise
  match  : vllm(bias ckpt) vs hf(hook +c)       -> same as above; proves c lands after layer L
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about a database migration.",
    "What is 17 * 23? Think step by step.",
    "Give me a recipe for pancakes.",
    "Summarize the plot of Hamlet in one paragraph.",
    "Translate 'good morning' into French, Spanish and German.",
    "What are three risks of running a red light?",
    "Write a Python function that reverses a linked list.",
    "Is it safe to mix bleach and ammonia?",
    "Draft a polite email declining a meeting invitation.",
    "List five prime numbers greater than 100.",
    "Describe the water cycle for a 10 year old.",
    "What does HTTP status 418 mean?",
    "Argue for and against daylight saving time.",
    "How do I center a div in CSS?",
    "Name the capital of Australia and one fact about it.",
    "Explain gradient descent without equations.",
    "What should I pack for a weekend hiking trip?",
    "Write a limerick about a cat who codes.",
    "Why do onions make you cry?",
]


def chat_texts(tok):
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        for p in PROMPTS
    ]


def run_vllm(model: str, out: str, topk: int = 10) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        quantization=os.getenv("TEST_QUANT") or None,  # fp8 for Gemma 4 31B on one H100
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_num_seqs=64,  # hybrid GDN model: Mamba cache holds ~200 decode slots at this memory
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=topk)
    outs = llm.generate(chat_texts(tok), sp)
    res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        res.append({int(t): float(v.logprob) for t, v in lp.items()})
    Path(out).write_text(json.dumps(res))
    print(f"vllm {model}: wrote {out}")


def run_hf(model: str, bias_ckpt: str | None, out: str, topk: int = 10) -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    hf = AutoModelForImageTextToText.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if bias_ckpt:
        shard = load_file(str(Path(bias_ckpt) / "model-steer-bias.safetensors"))
        ((name, c),) = shard.items()
        layer = int(name.split("layers.")[1].split(".")[0])
        c = c.to("cuda")
        blocks = (
            hf.model.language_model.layers
            if hasattr(hf.model, "language_model")
            else hf.model.layers
        )

        def hook(_m, _i, output):  # add c to the residual stream leaving this block
            if isinstance(output, tuple):
                return (output[0] + c,) + tuple(output[1:])
            return output + c

        blocks[layer].register_forward_hook(hook)
        print(
            f"hf: hook adding {name} (|c|={c.float().norm():.2f}) after layer {layer}"
        )
    res = []
    for text in chat_texts(tok):
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
            "cuda"
        )
        with torch.no_grad():
            logits = hf(input_ids=ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        top = torch.topk(lp, topk)
        res.append({int(t): float(v) for t, v in zip(top.indices, top.values)})
    Path(out).write_text(json.dumps(res))
    print(f"hf {model} bias={bias_ckpt}: wrote {out}")


def compare(a: str, b: str) -> None:
    A = json.loads(Path(a).read_text())
    B = json.loads(Path(b).read_text())
    agree, dl = 0, []
    for ra, rb in zip(A, B):
        ta = max(ra, key=ra.get)
        tb = max(rb, key=rb.get)
        agree += ta == tb
        if ta in rb:
            dl.append(abs(ra[ta] - rb[ta]))
    print(
        f"top-1 agreement {agree}/{len(A)}; max |dlogprob| on A's top-1 = {max(dl):.3f}; mean = {np.mean(dl):.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "hf"])
    ap.add_argument("--model")
    ap.add_argument("--bias", default=None)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.engine == "vllm":
        run_vllm(args.model, args.out)
    else:
        run_hf(args.model, args.bias, args.out)
