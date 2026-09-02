"""Pooled residual-stream features for every record of an eval_bench response file.

Runs ON THE GPU POD (HF transformers, no vLLM). Teacher-forces the stored
transcript (rendered chat + the model's own <think>...</think>response) through
the model once and pools the hidden state at every LAYER_STRIDE-th layer over
four spans:

  prompt_last   last prompt token (the position the Aug-2026 eval-awareness
                paper probes; arXiv:2608.21766)
  prompt_mean   mean over all prompt tokens (system + user)
  cot_mean      mean over the generated reasoning tokens (inside <think>)
  answer_mean   mean over the generated visible-answer tokens

Output: an .npz with feats[position] of shape (n_records, n_layers, hidden)
in float16, plus ids/layers so the probe script can join back to the
response + judge files by record id.

Usage (on pod):
  python extract_activations.py --model /workspace/qwen27b \
      --responses responses_qwen3.8-27b_probe_<ts>.json --out acts_qwen27b.npz
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

THINK_END = "</think>"
MAX_TOKENS = 12000  # truncate the (rare) monster CoTs so a batch never OOMs


def _messages(chat: dict) -> list[dict]:
    msgs = []
    if chat.get("system"):
        msgs.append({"role": "system", "content": chat["system"]})
    msgs.extend(chat["messages"])
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # Qwen3.8 ships as a VL checkpoint (Qwen3_5ForConditionalGeneration); the
    # text-only forward with input_ids still returns per-layer hidden states.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    layers = sorted(set(list(range(0, n_layers, args.layer_stride)) + [n_layers - 1]))
    hidden = text_cfg.hidden_size
    print(f"model loaded: {n_layers} layers, hidden={hidden}, probing {layers}")

    data = json.loads(Path(args.responses).read_text())
    records = data["samples"] if isinstance(data, dict) else data
    records = [r for r in records if not str(r["response"]).startswith("ERROR")]
    if args.limit:
        records = records[: args.limit]
    positions = ["prompt_last", "prompt_mean", "cot_mean", "answer_mean"]
    feats = {
        p: np.zeros((len(records), len(layers), hidden), dtype=np.float16)
        for p in positions
    }
    ids, n_tok = [], []
    think_end_ids = tok(THINK_END, add_special_tokens=False).input_ids

    t0 = time.time()
    for i, r in enumerate(records):
        chat = r["chat"] if isinstance(r["chat"], dict) else json.loads(r["chat"])
        prompt_ids = tok.apply_chat_template(
            _messages(chat), add_generation_prompt=True, tokenize=True
        )
        if isinstance(prompt_ids, dict):  # newer transformers return a BatchEncoding
            prompt_ids = prompt_ids["input_ids"]
        resp_ids = tok(r["response"], add_special_tokens=False).input_ids
        ids_all = (list(prompt_ids) + resp_ids)[:MAX_TOKENS]
        n_p = len(prompt_ids)
        # CoT span = response tokens up to and including the </think> token(s)
        resp_part = ids_all[n_p:]
        cot_end = len(resp_part)
        for j in range(len(resp_part) - len(think_end_ids) + 1):
            if resp_part[j : j + len(think_end_ids)] == think_end_ids:
                cot_end = j + len(think_end_ids)
                break
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids_all], device="cuda"),
                output_hidden_states=True,
            )
        hs = (
            out.hidden_states
        )  # tuple(n_layers+1) of (1, T, hidden); [k+1] is after block k
        for li, layer in enumerate(layers):
            h = hs[layer + 1][0].float()
            feats["prompt_last"][i, li] = h[n_p - 1].cpu().numpy()
            feats["prompt_mean"][i, li] = h[:n_p].mean(0).cpu().numpy()
            if cot_end > 0:
                feats["cot_mean"][i, li] = h[n_p : n_p + cot_end].mean(0).cpu().numpy()
            if len(resp_part) > cot_end:
                feats["answer_mean"][i, li] = h[n_p + cot_end :].mean(0).cpu().numpy()
        ids.append(r["id"])
        n_tok.append(len(ids_all))
        del out, hs
        if i % 50 == 0:
            el = time.time() - t0
            print(
                f"{i}/{len(records)} tokens={len(ids_all)} elapsed={el:.0f}s eta={el / (i + 1) * (len(records) - i):.0f}s",
                flush=True,
            )

    np.savez(
        args.out,
        ids=np.array(ids),
        layers=np.array(layers),
        n_tokens=np.array(n_tok),
        **feats,
    )
    print(f"wrote {args.out}: {len(ids)} records in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
