"""Bake a constant steering vector into a Qwen3.8-27B checkpoint as an MLP bias.

c = alpha * m * v_hat, the arXiv:2608.21766 recipe (m = mean residual norm at the
layer over the probe training prompts, v_hat a unit direction). Writes a new
checkpoint directory that symlinks every original file except the safetensors
index, adds one shard holding ``<layer>.mlp.down_proj.bias = c``, and rewrites
the index to include it. Serve it with stock vLLM plus the vllm_static_bias
plugin (STATIC_BIAS_MODULES=layers.<L>.mlp). alpha=0 gives the null-bias
control for the correctness test.

Usage (pod):
  python bake_steer_bias.py --model /workspace/qwen27b --directions steer_directions.npz \
      --vector data --alpha 0.5 --layer 28 --out /workspace/ckpt_data_a0.5
  --vector both adds the data and framing vectors (alpha each).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument(
        "--vector", choices=["data", "framing", "both", "zero"], required=True
    )
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # Find this layer's down_proj weight name in the checkpoint and derive the bias name.
    cands = [
        k for k in weight_map if k.endswith(f"layers.{args.layer}.mlp.down_proj.weight")
    ]
    assert len(cands) == 1, cands
    bias_name = cands[0][: -len("weight")] + "bias"

    d = np.load(args.directions)
    layers = [int(v) for v in d["layers"]]
    li = layers.index(args.layer)
    m = float(d["train_mean_norm"][li])
    # |c| in absolute units, for alpha scaled relative to that layer's mean norm.
    # Gemma 4's residual norms are ~4x Qwen's at mid-depth (212 vs 50 at L28), so
    # the same alpha is a proportionally equal push, not an equal absolute one.
    vecs = {"data": d["data_unit"][li], "framing": d["framing_unit"][args.layer]}
    if args.vector == "zero":
        c = np.zeros_like(vecs["data"])
    elif args.vector == "both":
        c = args.alpha * m * (vecs["data"] + vecs["framing"])
    else:
        c = args.alpha * m * vecs[args.vector]
    bias = torch.tensor(c, dtype=torch.float32).to(torch.bfloat16)

    shard = "model-steer-bias.safetensors"
    save_file(
        {bias_name: bias.contiguous()}, str(out / shard), metadata={"format": "pt"}
    )
    weight_map[bias_name] = shard
    index["metadata"] = dict(index.get("metadata", {}))
    index["metadata"]["steer_bias"] = json.dumps(
        {
            "vector": args.vector,
            "alpha": args.alpha,
            "layer": args.layer,
            "mean_norm": m,
            "norm_c": float(np.linalg.norm(c)),
        }
    )
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for f in src.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        dst = out / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    print(
        f"wrote {out}: {bias_name} (|c|={np.linalg.norm(c):.2f}, m={m:.1f}, alpha={args.alpha}, vector={args.vector})"
    )


if __name__ == "__main__":
    main()
