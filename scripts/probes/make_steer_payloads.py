"""Serialize steering vectors into the `vllm_xargs` JSON the vllm-lens server expects.

One file per (direction, alpha): steer_<direction>_L<layer>_a<alpha>.json containing
{"apply_steering_vectors": "<json list of one serialized SteeringVector>"}.
norm_match=True means h' = h + alpha * ||h|| * v_hat at every token of the layer
(prompt and generated), which is the arXiv:2608.21766 recipe with the per-token
norm in place of the calibration-set mean norm.

Runs on the pod (needs vllm_lens). Usage:
  python make_steer_payloads.py --directions steer_directions.npz --layer 28 \
      --alphas -1,-0.5,0.5,1 --out-dir payloads
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vllm_lens import SteeringVector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", default="-1,-0.5,0.5,1")
    ap.add_argument("--out-dir", default="payloads")
    args = ap.parse_args()

    d = np.load(args.directions)
    layers = [int(v) for v in d["layers"]]
    dirs = {
        "framing": d["framing_unit"][args.layer],
        "data": d["data_unit"][layers.index(args.layer)],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, vec in dirs.items():
        for alpha in [float(a) for a in args.alphas.split(",")]:
            sv = SteeringVector(
                activations=torch.tensor(vec, dtype=torch.float32)[None, :],
                layer_indices=[args.layer],
                scale=alpha,
                norm_match=True,
            )
            payload = {"apply_steering_vectors": json.dumps([sv.model_dump()])}
            path = out / f"steer_{name}_L{args.layer}_a{alpha:+.2f}.json"
            path.write_text(json.dumps(payload))
            print(path, len(payload["apply_steering_vectors"]), "bytes")


if __name__ == "__main__":
    main()
