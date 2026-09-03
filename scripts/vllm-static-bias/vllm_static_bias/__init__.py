"""vLLM general plugin: add a bias to selected MLP down_proj modules.

Activation steering as a *constant* vector c added to the residual stream after
layer L equals giving that layer's MLP output projection a bias of c (the MLP
output is what gets added to the residual). vLLM's dense MLPs
(``Qwen2MoeMLP`` for Qwen3.5/Qwen3-Next, ``Gemma4MLP`` for Gemma 4) build ``down_proj``
with ``bias=False``; this plugin rebuilds it with ``bias=True`` for the modules named
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

    # Dense MLP classes whose down_proj we may need to give a bias. Each is
    # patched the same way: run the original __init__, then, if the module
    # prefix matches a target, rebuild down_proj with bias=True.
    mlp_classes = []
    from vllm.model_executor.models import qwen2_moe

    mlp_classes.append(
        qwen2_moe.Qwen2MoeMLP
    )  # Qwen3.5 / Qwen3-Next dense layers reuse this
    try:
        from vllm.model_executor.models import gemma4

        mlp_classes.append(gemma4.Gemma4MLP)
    except ImportError:
        pass

    for cls in mlp_classes:
        _patch(cls, targets, RowParallelLinear)
    log.warning(
        "vllm_static_bias: active for module suffixes %s on %s",
        targets,
        [c.__name__ for c in mlp_classes],
    )


def _patch(cls, targets, RowParallelLinear) -> None:
    orig_init = cls.__init__
    sig = inspect.signature(orig_init)

    def patched_init(self, *args, **kwargs):
        # Signature-agnostic: bind whatever this vLLM version's constructor takes
        # (Qwen2MoeMLP has reduce_results/is_sequence_parallel/disable_tp,
        # Gemma4MLP has neither).
        orig_init(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        prefix = a.get("prefix", "")
        if not any(prefix.endswith(t) for t in targets):
            return
        extra = {}
        if "reduce_results" in a:
            extra["reduce_results"] = a["reduce_results"]
        if "disable_tp" in a or "is_sequence_parallel" in a:
            extra["disable_tp"] = bool(
                a.get("disable_tp", False) or a.get("is_sequence_parallel", False)
            )
        self.down_proj = RowParallelLinear(
            a["intermediate_size"],
            a["hidden_size"],
            bias=True,
            quant_config=a.get("quant_config"),
            prefix=f"{prefix}.down_proj",
            **extra,
        )
        log.warning("vllm_static_bias: %s.down_proj rebuilt with bias=True", prefix)

    cls.__init__ = patched_init
