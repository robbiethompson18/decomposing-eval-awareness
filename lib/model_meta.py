"""Static metadata for target models (leaf module — imports nothing from this repo).

Keyed by the model_key used in result filenames, which sometimes differs from
the target-client slug (e.g. "gemini-3.1-pro" here vs "gemini-3.1-pro-preview";
this is also why lib.config's display-name map can't be reused).

provider = coarse grouping used for plot colors; lab = fine-grained attribution.
release: month-precision dates use day=15. usd_out = $ per Mtok output tokens —
prevailing OpenRouter price as of 2026-09-01 (non-batch), except sonnet/haiku
4.5 and gpt-5/5.1 which are first-party list prices (OR matches).
None = unknown when last edited. #@robbie fill in the None release dates.
"""

from datetime import date

MODEL_META: dict[str, dict] = {
    "claude-sonnet-4.5": {
        "display": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "lab": "Anthropic",
        "release": date(2025, 9, 29),
        "usd_out": 15.0,
    },
    "claude-haiku-4.5": {
        "display": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "lab": "Anthropic",
        "release": date(2025, 10, 1),
        "usd_out": 5.0,
    },
    "gpt-5": {
        "display": "GPT-5",
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": date(2025, 8, 7),
        "usd_out": 10.0,
    },
    "gpt-5.1": {
        "display": "GPT-5.1",
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": date(2025, 11, 15),
        "usd_out": 10.0,
    },
    "gpt-5.2": {
        "display": "GPT-5.2",
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": 14.0,
    },
    "gpt-5.4": {
        "display": "GPT-5.4",
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": 15.0,
    },
    "gpt-5.6-sol": {
        "display": "GPT-5.6",
        "provider": "OpenAI",
        "lab": "OpenAI",
        "release": None,
        "usd_out": 10.0,
    },
    "gemini-3.1-pro": {
        "display": "Gemini 3.1 Pro",
        "provider": "Google",
        "lab": "Google",
        "release": None,
        "usd_out": 12.0,
    },
    "gemini-3.5-flash": {
        "display": "Gemini 3.5 Flash",
        "provider": "Google",
        "lab": "Google",
        "release": None,
        "usd_out": 9.0,
    },
    "qwen3.8-2.4t": {
        "display": "Qwen3.8",
        "provider": "Open-weight",
        "lab": "Alibaba",
        "release": date(2026, 8, 15),
        "usd_out": 6.0,
    },
    "deepseek-v4-pro": {
        "display": "DeepSeek V4 Pro",
        "provider": "Open-weight",
        "lab": "DeepSeek",
        "release": date(2026, 4, 15),
        "usd_out": 3.2,
    },
    "kimi-k3": {
        "display": "Kimi K3",
        "provider": "Open-weight",
        "lab": "Moonshot",
        "release": date(2026, 7, 15),
        "usd_out": 15.0,
    },
    "glm-5.3": {
        "display": "GLM-5.3",
        "provider": "Open-weight",
        "lab": "Z.ai",
        "release": date(2026, 8, 15),
        "usd_out": 4.4,
    },
    "minimax-m3": {
        "display": "MiniMax M3",
        "provider": "Open-weight",
        "lab": "MiniMax",
        "release": date(2026, 5, 15),
        "usd_out": 1.2,
    },
    # The paper's (arXiv:2605.23055) open-weight roster, kept for the matched
    # replication runs (plots/plot_stacking.py). Display names follow the paper's
    # figures. usd_out unknown; #@robbie fill in if these end up on a cost chart.
    "deepseek-r1": {
        "display": "DeepSeek R1",
        "provider": "Open-weight",
        "lab": "DeepSeek",
        "release": date(2025, 1, 20),
        "usd_out": None,
    },
    "kimi-k2.5": {
        "display": "Kimi K2.5",
        "provider": "Open-weight",
        "lab": "Moonshot",
        "release": None,
        "usd_out": None,
    },
    "glm-5": {
        "display": "GLM-5",
        "provider": "Open-weight",
        "lab": "Z.ai",
        "release": None,
        "usd_out": None,
    },
    "qwen3-235b": {
        "display": "Qwen3 T-235B",
        "provider": "Open-weight",
        "lab": "Alibaba",
        "release": date(2025, 7, 25),
        "usd_out": None,
    },
    "qwen3-30b": {
        "display": "Qwen3 T-30B",
        "provider": "Open-weight",
        "lab": "Alibaba",
        "release": date(2025, 7, 30),
        "usd_out": None,
    },
    "gpt-oss-120b": {
        "display": "GPT-oss 120B",
        "provider": "Open-weight",
        "lab": "OpenAI",
        "release": date(2025, 8, 5),
        "usd_out": None,
    },
    "gpt-oss-20b": {
        "display": "GPT-oss 20B",
        "provider": "Open-weight",
        "lab": "OpenAI",
        "release": date(2025, 8, 5),
        "usd_out": None,
    },
}


def display_name(model_key: str) -> str:
    return MODEL_META.get(model_key, {}).get("display", model_key)
