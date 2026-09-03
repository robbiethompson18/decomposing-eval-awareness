"""Generate one response per probe-trainset prompt from a self-hosted vLLM model.

The train set (results/probes/trainset.jsonl) is prompt-only; BDV' (the
benchmark-vs-chat direction read at the last *response* token) needs the
model's own answer to every prompt. Output has the run_eval_bench response-file
shape ({"samples": [{id, chat, response, ...}]}) so extract_activations.py can
teacher-force it unchanged; the response is "<think>reasoning</think>answer".

Usage:
  VLLM_BASE_URL=http://localhost:8000/v1 python scripts/gen_trainset_responses.py \
      --trainset results/probes/trainset.jsonl --out results/probes/trainset_responses_qwen27b.json
Resumes: existing ids in --out are skipped; saves every 100.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI


async def one(client, model, rec, sem, max_tokens):
    chat = rec["chat"]
    msgs = (
        [{"role": "system", "content": chat["system"]}] if chat.get("system") else []
    ) + chat["messages"]
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=0.3
            )
            m = r.choices[0].message
            reasoning = (
                getattr(m, "reasoning_content", None)
                or getattr(m, "reasoning", None)
                or ""
            )
            text = f"<think>{reasoning}</think>{m.content or ''}".strip()
        except Exception as e:  # noqa: BLE001 -- keep the run going; ERROR rows are dropped downstream
            text = f"ERROR calling vLLM: {e}"
    return {**rec, "response": text}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", default="results/probes/trainset.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3.8-27B"))
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    recs = [
        json.loads(line) for line in Path(args.trainset).read_text().split("\n") if line
    ]
    out = Path(args.out)
    data = (
        json.loads(out.read_text())
        if out.exists()
        else {"model": args.model, "samples": []}
    )
    data["samples"] = [
        s for s in data["samples"] if not str(s["response"]).startswith("ERROR")
    ]
    done = {s["id"] for s in data["samples"]}
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(todo)} to generate ({len(done)} cached)")

    client = AsyncOpenAI(
        api_key="EMPTY", base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(one(client, args.model, r, sem, args.max_tokens))
        for r in todo
    ]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        data["samples"].append(await fut)
        if i % 100 == 0 or i == len(tasks):
            out.write_text(json.dumps(data, ensure_ascii=False))
            print(f"{i}/{len(tasks)} saved", flush=True)
    n_err = sum(str(s["response"]).startswith("ERROR") for s in data["samples"])
    print(f"done: {len(data['samples'])} samples, {n_err} errors -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
