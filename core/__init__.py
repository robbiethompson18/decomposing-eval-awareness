"""
Core evaluation modules.

Kept intentionally minimal: the eval_bench scripts import from the
submodules directly (core.evaluator, core.metrics), so this only
re-exports the live surface.
"""

from .evaluator import (
    create_judge_client,
    evaluate_batch_async,
    evaluate_response_async,
    evaluate_sample,
)
from .metrics import (
    bucket_conclusion,
    bucket_task_conclusion,
    extract_conclusion,
)
from .target import (
    AnthropicApiClient,
    OpenAITargetClient,
    OpenRouterApiClient,
    create_openai_client,
)

__all__ = [
    "AnthropicApiClient",
    "OpenAITargetClient",
    "OpenRouterApiClient",
    "bucket_conclusion",
    "bucket_task_conclusion",
    "create_judge_client",
    "create_openai_client",
    "evaluate_batch_async",
    "evaluate_response_async",
    "evaluate_sample",
    "extract_conclusion",
]
