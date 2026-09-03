"""Client-side throughput benchmark: N prompts through an OpenAI-compatible vLLM server.

Reports wall time, completion tokens, tokens/s. Run once per server variant
(stock / bias-baked / vllm-lens unsteered / vllm-lens steered) with the same
prompts, concurrency and max_tokens so the numbers are comparable.

Usage: python scripts/bench_steer_throughput.py --base-url http://localhost:8000/v1 \
    --prompts results/probes/prompts/stack_a.jsonl --n 100 --concurrency 64 --max-tokens 2000 \
    [--steer-xargs results/probes/payloads/steer_data_L28_a+0.50.json] --label stock
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--steer-xargs", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="results/probes/throughput.jsonl")
    args = ap.parse_args()

    recs = [
        json.loads(line) for line in Path(args.prompts).read_text().split("\n") if line
    ][: args.n]
    extra = (
        {"vllm_xargs": json.loads(Path(args.steer_xargs).read_text())}
        if args.steer_xargs
        else {}
    )
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url, timeout=1800)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r):
        msgs = (
            [{"role": "system", "content": r["chat"]["system"]}]
            if r["chat"].get("system")
            else []
        ) + r["chat"]["messages"]
        async with sem:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=msgs,
                max_tokens=args.max_tokens,
                temperature=0.3,
                extra_body=extra,
            )
        return resp.usage.completion_tokens, resp.usage.prompt_tokens

    t0 = time.time()
    usage = await asyncio.gather(*(one(r) for r in recs))
    wall = time.time() - t0
    comp = sum(u[0] for u in usage)
    prompt = sum(u[1] for u in usage)
    row = {
        "label": args.label,
        "n": len(recs),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 1),
        "completion_tokens": comp,
        "prompt_tokens": prompt,
        "gen_tok_per_s": round(comp / wall, 1),
    }
    print(json.dumps(row))
    return args.out, row


if __name__ == "__main__":
    out_path, result = asyncio.run(main())
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
