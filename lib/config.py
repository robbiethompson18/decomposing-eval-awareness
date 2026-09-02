"""Model registries and environment-based API configuration."""

import os
from typing import Any

# We should think harder about how we want to change default model configs - Robbie

# Model configs: model_key -> model_id
# Used to map user-friendly keys to actual API model identifiers
DEFAULT_ANTHROPIC_MODELS: dict[str, str] = {
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-opus-5": "claude-opus-5",
    "claude-fable-5": "claude-fable-5",
}

DEFAULT_OPENAI_MODELS: dict[str, str] = {
    "gpt-5": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
}

# Gemini model configs: model_key -> (model_id, thinking_budget floor, force_first_think).
# Only the two models validated as reliable think-tool CoT extractors
# (see .claude/notes/cot-extraction-model-roster.md). thinkingBudget 128 is the
# mandatory minimum for 3.1-pro (rejects 0); 0 turns thinking fully off for
# 3.5-flash. Starving native thinking pushes the CoT into visible think-tool calls.
# force_first_think forces a round-1 think call (toolConfig mode ANY): needed for
# 3.5-flash, which otherwise skips the tool on easy tasks; must stay False for
# 3.1-pro, where ANY at the starved budget makes it spam think calls until
# MAX_TOKENS (hundreds of calls, no answer) while plain AUTO complies fine.
DEFAULT_GEMINI_CONFIGS: dict[str, tuple[str, int, bool]] = {
    "gemini-3.1-pro": ("gemini-3.1-pro-preview", 128, False),
    "gemini-3.5-flash": ("gemini-3.5-flash", 0, True),
}

# OpenRouter model configs: model_key -> (model_id, temperature)
# temperature=None means model doesn't support it (reasoning models)
# Note: claude-sonnet-4.5 and gpt-5 use native APIs, not OpenRouter
DEFAULT_OPENROUTER_CONFIGS: dict[str, tuple[str, float | None]] = {
    "claude-sonnet-4.5": ("anthropic/claude-sonnet-4.5", 0.3),
    "claude-opus-4.5": ("anthropic/claude-opus-4.5", 0.3),
    "gpt-5": ("openai/gpt-5", None),  # GPT-5 doesn't support temperature
    "qwen3-30b": ("qwen/qwen3-30b-a3b-thinking-2507", 0.3),
    "qwen3-235b": ("qwen/qwen3-235b-a22b-thinking-2507", 0.3),
    "deepseek-r1": ("deepseek/deepseek-r1", 0.3),  # DeepSeek R1 reasoning model
    "kimi-k2.5": ("moonshotai/kimi-k2.5", 0.3),  # Moonshot Kimi K2.5 reasoning model
    "glm-5": ("z-ai/glm-5", 0.3),  # Z.ai GLM-5 reasoning model
    "gpt-oss-120b": (
        "openai/gpt-oss-120b",
        0.3,
    ),  # OpenAI open-weight MoE reasoning model
    "gpt-oss-20b": (
        "openai/gpt-oss-20b",
        0.3,
    ),  # OpenAI open-weight small MoE reasoning model
    # Tier-1 current-generation open-weight reasoning models (added 2026-08-31),
    # one per lab; all have published HF weights (checked via OpenRouter metadata)
    "qwen3.8-2.4t": ("qwen/qwen3.8-2.4t-a95b", 0.3),  # Alibaba flagship, Aug 2026
    "deepseek-v4-pro": ("deepseek/deepseek-v4-pro", 0.3),  # DeepSeek flagship, Apr 2026
    "kimi-k3": ("moonshotai/kimi-k3", 0.3),  # Moonshot flagship, Jul 2026
    "glm-5.3": ("z-ai/glm-5.3", 0.3),  # Z.ai flagship, Aug 2026
    "minimax-m3": ("minimax/minimax-m3", 0.3),  # MiniMax flagship, May 2026
}


def get_api_keys_config() -> dict[str, Any]:
    """Get common API keys and model defaults from environment variables.

    Returns dict with keys for Anthropic, OpenAI, and OpenRouter.
    Each key uses os.getenv with no default — users must set environment variables.
    """
    return {
        # Anthropic
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        "ANTHROPIC_API_BASE": os.getenv(
            "ANTHROPIC_API_BASE", "https://api.anthropic.com"
        ),
        # OpenAI
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_TARGET_MODEL": os.getenv("OPENAI_TARGET_MODEL", "gpt-5"),
        # OpenRouter
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        # Gemini
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    }
