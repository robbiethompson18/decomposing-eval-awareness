"""vLLM general plugin: add a bias to selected MLP down_proj modules.

Activation steering as a *constant* vector c added to the residual stream after
layer L equals giving that layer's MLP output projection a bias of c (the MLP
output is what gets added to the residual). vLLM's Qwen3.5 dense MLP
(``Qwen2MoeMLP``, reused by ``qwen3_next.py``) builds ``down_proj`` with
``bias=False``; this plugin rebuilds it with ``bias=True`` for the modules named
in ``STATIC_BIAS_MODULES`` (comma-separated suffixes of the vLLM module prefix,
e.g. ``layers.28.mlp``), before weights load, so a checkpoint carrying
``...layers.28.mlp.down_proj.bias`` is picked up by the normal weight loader.

Nothing else changes: no hooks, no eager mode, CUDA graphs and the V2 model
runner stay on. Build the checkpoint with scripts/bake_steer_bias.py. Loaded in
every vLLM process via the ``vllm.general_plugins`` entry point; a no-op unless
``STATIC_BIAS_MODULES`` is set.
"""

import inspect
import logging
import os

log = logging.getLogger("vllm_static_bias")


def register() -> None:
    targets = [
        s.strip() for s in os.getenv("STATIC_BIAS_MODULES", "").split(",") if s.strip()
    ]
    if not targets:
        return
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.models import qwen2_moe

    orig_init = qwen2_moe.Qwen2MoeMLP.__init__
    sig = inspect.signature(orig_init)

    def patched_init(self, *args, **kwargs):
        # Signature-agnostic: vLLM main and 0.28 differ (`disable_tp` vs
        # `is_sequence_parallel`), so bind whatever this version takes.
        orig_init(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        prefix = a.get("prefix", "")
        if not any(prefix.endswith(t) for t in targets):
            return
        self.down_proj = RowParallelLinear(
            a["intermediate_size"],
            a["hidden_size"],
            bias=True,
            quant_config=a.get("quant_config"),
            reduce_results=a.get("reduce_results", True),
            disable_tp=bool(
                a.get("disable_tp", False) or a.get("is_sequence_parallel", False)
            ),
            prefix=f"{prefix}.down_proj",
        )
        log.warning("vllm_static_bias: %s.down_proj rebuilt with bias=True", prefix)

    qwen2_moe.Qwen2MoeMLP.__init__ = patched_init
    log.warning("vllm_static_bias: active for module suffixes %s", targets)
