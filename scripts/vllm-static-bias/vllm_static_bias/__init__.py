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

    def patched_init(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        quant_config=None,
        reduce_results=True,
        expert_gate=None,
        is_sequence_parallel=False,
        disable_tp=False,
        prefix="",
    ):
        orig_init(
            self,
            hidden_size,
            intermediate_size,
            hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results,
            expert_gate=expert_gate,
            is_sequence_parallel=is_sequence_parallel,
            disable_tp=disable_tp,
            prefix=prefix,
        )
        if any(prefix.endswith(t) for t in targets):
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=True,
                quant_config=quant_config,
                reduce_results=reduce_results,
                disable_tp=disable_tp or is_sequence_parallel,
                prefix=f"{prefix}.down_proj",
            )
            log.warning("vllm_static_bias: %s.down_proj rebuilt with bias=True", prefix)

    qwen2_moe.Qwen2MoeMLP.__init__ = patched_init
    log.warning("vllm_static_bias: active for module suffixes %s", targets)
