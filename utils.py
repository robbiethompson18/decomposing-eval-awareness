"""
Shared helper functions.

Sections (in order):
1. Basic Utilities - File I/O, configuration
2. API Client Functions - Client creation, model configs
3. Parallel Query Functions - Async helpers
4. Judge Template Functions - Template processing
5. Results Extraction Functions - Per-model extraction
6. Cross-Model Comparison Functions - Multi-model analysis
7. Evaluation Statistics Functions - Stats printing
8. AgentHarm Log Processing Functions - Log parsing
"""

import asyncio
import csv
import json
import os
import re
import sys
from collections.abc import Callable
from glob import glob
from typing import Any

from core.target import (
    AnthropicApiClient,
    AnthropicApiClientScratchpad,
    DeepSeekApiClient,
    OpenAITargetClient,
    OpenAITargetClientScratchpad,
    OpenRouterApiClient,
)

# =============================================================================
# Basic Utilities
# =============================================================================


def read_file(filepath: str) -> str:
    """Read text file contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# API Client Functions
# =============================================================================

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
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
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
}

# AgentHarm model configs: model_key -> (inspect_ai_model_id, temperature)
# Maps short model keys to full Inspect AI model identifiers for consistent naming
AGENTHARM_MODEL_CONFIGS: dict[str, tuple[str, float]] = {
    "claude-sonnet-4.5": ("anthropic/claude-sonnet-4-5-20250929", 0.3),
    "gpt-5": ("openai/gpt-5", 1.0),
    "qwen3-30b": ("openrouter/qwen/qwen3-30b-a3b-thinking-2507", 0.3),
    "qwen3-235b": ("openrouter/qwen/qwen3-235b-a22b-thinking-2507", 0.3),
    "deepseek-r1": ("openrouter/deepseek/deepseek-r1", 0.3),
    "kimi-k2.5": ("openrouter/moonshotai/kimi-k2.5", 0.3),
    "glm-5": ("openrouter/z-ai/glm-5", 0.3),
    "gpt-oss-120b": ("openrouter/openai/gpt-oss-120b", 0.3),
    "gpt-oss-20b": ("openrouter/openai/gpt-oss-20b", 0.3),
}


def get_api_keys_config() -> dict[str, Any]:
    """Get common API keys and model defaults from environment variables.

    Returns dict with keys for DeepSeek, Anthropic, OpenAI, and OpenRouter.
    Each key uses os.getenv with no default — users must set environment variables.
    """
    return {
        # DeepSeek
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
        "DEEPSEEK_MODEL": os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
        "DEEPSEEK_API_BASE": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        # Anthropic
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        "ANTHROPIC_API_BASE": os.getenv(
            "ANTHROPIC_API_BASE", "https://api.anthropic.com"
        ),
        "THINKING_BUDGET": int(os.getenv("THINKING_BUDGET", "10000")),
        # OpenAI
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_TARGET_MODEL": os.getenv("OPENAI_TARGET_MODEL", "gpt-5"),
        # OpenRouter
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    }


def load_responses(response_file: str) -> tuple[list[dict[str, Any]], str]:
    """Load samples and model name from a response JSON file.

    Args:
        response_file: Path to response JSON file

    Returns:
        (samples, model) tuple
    """
    with open(response_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", []), data.get("model", "unknown")


def find_harmbench_response_file(
    base_dir: str,
    target_model: str,
    reasoning_mode: str = "summary",
) -> str:
    """Find the most recent HarmBench response file for a target model.

    Args:
        base_dir: Base directory of the project (contains results/)
        target_model: Model key (e.g., "gpt-5", "claude-sonnet-4.5", "qwen3-30b")
        reasoning_mode: "summary" or "scratchpad"

    Returns:
        Path to the most recent response file, or empty string if not found
    """
    response_dir = os.path.join(base_dir, "results", "harmbench", "response")

    # Build pattern based on reasoning mode
    if reasoning_mode == "scratchpad":
        pattern = os.path.join(
            response_dir, f"responses_{target_model}_scratchpad*.json"
        )
    else:
        pattern = os.path.join(response_dir, f"responses_{target_model}*.json")

    matches = sorted(glob(pattern))

    # Filter out scratchpad files (for summary mode)
    if reasoning_mode != "scratchpad":
        matches = [m for m in matches if "_scratchpad" not in os.path.basename(m)]

    return matches[-1] if matches else ""


def find_agenticmisalignment_response_file(
    base_dir: str,
    target_model: str,
    reasoning_mode: str = "summary",
) -> str:
    """Find the most recent Agentic Misalignment response file for a target model.

    Args:
        base_dir: Base directory of the project (contains results/)
        target_model: Model key (e.g., "gpt-5", "claude-sonnet-4.5", "qwen3-30b")
        reasoning_mode: "summary" or "scratchpad"

    Returns:
        Path to the most recent response file, or empty string if not found
    """
    response_dir = os.path.join(base_dir, "results", "agenticmisalignment", "response")

    # Build pattern based on reasoning mode
    if reasoning_mode == "scratchpad":
        pattern = os.path.join(
            response_dir, f"responses_{target_model}_scratchpad*.json"
        )
    else:
        pattern = os.path.join(response_dir, f"responses_{target_model}*.json")

    matches = sorted(glob(pattern))

    # Filter out scratchpad files (for summary mode)
    if reasoning_mode != "scratchpad":
        matches = [m for m in matches if "_scratchpad" not in os.path.basename(m)]

    return matches[-1] if matches else ""


def create_target_client(
    backend: str,
    cfg: dict[str, Any],
    system_prompt: str,
    reasoning_mode: str = "summary",
    verbose: bool = True,
):
    """Create target client based on backend selection.

    Args:
        backend: Backend name (deepseek_api, anthropic_api, openai_api, openrouter)
        cfg: Configuration dict with API keys and settings (including MAX_TOKENS)
        system_prompt: System prompt for the target model
        reasoning_mode: "summary" for native reasoning APIs, "scratchpad" for explicit tags
                        Only affects anthropic_api and openai_api backends
        verbose: Whether to print backend info (default True)

    Returns:
        Tuple of (client, backend_slug)
    """
    b = backend.strip().lower()
    use_scratchpad = reasoning_mode == "scratchpad"

    if b == "deepseek_api":
        if verbose:
            print("Using DeepSeek API backend...")
        return DeepSeekApiClient(cfg, system_prompt), "deepseek"
    if b == "anthropic_api":
        model_key = os.getenv("ANTHROPIC_MODEL_KEY", "claude-sonnet-4.5")
        if model_key in DEFAULT_ANTHROPIC_MODELS:
            cfg["ANTHROPIC_MODEL"] = DEFAULT_ANTHROPIC_MODELS[model_key]
        if use_scratchpad:
            if verbose:
                print(
                    f"Using Anthropic API backend (scratchpad mode, model={model_key})..."
                )
            return AnthropicApiClientScratchpad(cfg, system_prompt), model_key
        if verbose:
            print(
                f"Using Anthropic API backend (extended thinking, model={model_key})..."
            )
        return AnthropicApiClient(cfg, system_prompt), model_key
    if b == "openai_api":
        model_key = os.getenv("OPENAI_MODEL_KEY", "gpt-5")
        if model_key in DEFAULT_OPENAI_MODELS:
            cfg["OPENAI_TARGET_MODEL"] = DEFAULT_OPENAI_MODELS[model_key]
        if use_scratchpad:
            if verbose:
                print(
                    f"Using OpenAI API backend (scratchpad mode, model={model_key})..."
                )
            return OpenAITargetClientScratchpad(cfg, system_prompt), model_key
        if verbose:
            print(f"Using OpenAI API backend (reasoning summary, model={model_key})...")
        return OpenAITargetClient(cfg, system_prompt), model_key
    if b == "openrouter":
        model_key = os.getenv("OPENROUTER_MODEL_KEY", "qwen3-30b")
        if model_key not in DEFAULT_OPENROUTER_CONFIGS:
            print(
                f"ERROR: Unknown OPENROUTER_MODEL_KEY '{model_key}'. Available: {list(DEFAULT_OPENROUTER_CONFIGS.keys())}"
            )
            sys.exit(1)
        model_id, temperature = DEFAULT_OPENROUTER_CONFIGS[model_key]
        cfg["OPENROUTER_MODEL"] = model_id
        cfg["TEMPERATURE"] = temperature
        if verbose:
            print(f"Using OpenRouter API backend (model={model_key}, id={model_id})...")
        return OpenRouterApiClient(cfg, system_prompt), model_key
    print(
        f"ERROR: Unsupported TARGET_BACKEND '{backend}'. Use: deepseek_api, anthropic_api, openai_api, openrouter"
    )
    sys.exit(1)


# =============================================================================
# Parallel Query Functions
# =============================================================================


async def query_parallel(
    items: list[tuple[Any, str, dict[str, Any]]],
    result_fn: Callable[[str, dict[str, Any], int], dict[str, Any]],
    max_concurrent: int = 10,
    prefill: str = None,
) -> list[dict[str, Any]]:
    """
    Run parallel queries to target models.

    Args:
        items: List of (client, prompt, metadata) tuples
        result_fn: Function (response, metadata, sample_idx) -> result dict
        max_concurrent: Max concurrent requests
        prefill: Optional text to prefill the assistant response (seeds chain of thought)

    Returns:
        List of results in original order
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(items)
    current = [0]

    async def query_one(
        idx: int, client: Any, prompt: str, metadata: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            response = await asyncio.to_thread(client.query, prompt, prefill)
            current[0] += 1
            sample_idx = metadata.get("sample_idx", 1)
            item_id = metadata.get("id", f"item_{idx}")
            print(f"  [{current[0]}/{total}] {item_id} sample {sample_idx}")
            return idx, result_fn(response, metadata, sample_idx)

    tasks = [
        query_one(i, client, prompt, meta)
        for i, (client, prompt, meta) in enumerate(items)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid results and sort by original index
    valid = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        else:
            valid.append(r)

    valid.sort(key=lambda x: x[0])
    return [r[1] for r in valid]


# =============================================================================
# Judge Template Functions
# =============================================================================


def get_agenticmisalignment_judge_template(
    base_template: str, scenario: str, prompts_dir: str
) -> str:
    """
    Fill the agentic misalignment judge template with scenario-specific criteria.

    Args:
        base_template: The base judge template with {task_performance_criteria} placeholder
        scenario: One of "murder", "blackmail", "leaking"
        prompts_dir: Path to the prompts directory containing criteria files

    Returns:
        Judge template with scenario-specific harmful behavior criteria filled in
    """
    valid_scenarios = ("murder", "blackmail", "leaking")
    if scenario not in valid_scenarios:
        raise ValueError(
            f"Unknown scenario: {scenario}. Must be one of {valid_scenarios}"
        )

    criteria_file = os.path.join(
        prompts_dir, f"agenticmisalignment_criteria_{scenario}.txt"
    )
    criteria = read_file(criteria_file).strip()
    return base_template.replace("{task_performance_criteria}", criteria)


# =============================================================================
# AgentHarm Log Processing Functions
# =============================================================================


def find_agentharm_log_file(
    base_dir: str,
    target_model: str,
    dataset_type: str = "harmful",
    reasoning_mode: str = "summary",
    result_subdir: str = "",
) -> str:
    """
    Find the most recent AgentHarm log file for a target model.

    File naming convention: responses_{model}_{dataset}.json (or with timestamp)
    or responses_{model}_scratchpad_{dataset}.json for scratchpad mode.

    Args:
        base_dir: Base directory of the project (contains results/)
        target_model: Model key (e.g., "gpt-5", "claude-sonnet-4.5", "qwen3-30b")
        dataset_type: "harmful" or "benign"
        reasoning_mode: "summary" or "scratchpad"
        result_subdir: Override subdirectory (e.g., "rerun" -> results/rerun/agentharm/)

    Returns:
        Path to the most recent log file, or empty string if not found
    """
    result_dir = "agentharm" if dataset_type == "harmful" else "agentharm_benign"
    if result_subdir:
        response_dir = os.path.join(
            base_dir, "results", result_subdir, result_dir, "response"
        )
    else:
        response_dir = os.path.join(base_dir, "results", result_dir, "response")

    # Build pattern based on reasoning mode
    if reasoning_mode == "scratchpad":
        pattern = os.path.join(
            response_dir, f"responses_{target_model}_scratchpad_{dataset_type}*.json"
        )
    else:
        pattern = os.path.join(
            response_dir, f"responses_{target_model}_{dataset_type}*.json"
        )

    matches = sorted(glob(pattern))

    # Filter out scratchpad files (for summary mode)
    if reasoning_mode != "scratchpad":
        matches = [m for m in matches if "_scratchpad" not in os.path.basename(m)]

    return matches[-1] if matches else ""


def load_agentharm_logs(log_path: str) -> tuple[list[dict[str, Any]], str]:
    """
    Load AgentHarm JSON log file(s) from a file or directory.

    Args:
        log_path: Path to a JSON file or directory containing JSON files

    Returns:
        (samples, model) tuple:
        - samples: List of raw sample dicts from all log files
        - model: Model name from the logs
    """
    # Handle single file or directory
    if os.path.isfile(log_path):
        log_files = [log_path]
    else:
        log_files = sorted(glob(os.path.join(log_path, "*.json")))

    if not log_files:
        return [], ""

    all_samples = []
    model = ""

    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            log = json.load(f)

        if not model:
            model = log.get("eval", {}).get("model", "unknown")

        samples = log.get("samples", [])
        all_samples.extend(samples)

    return all_samples, model


def build_agentharm_interaction_history(sample: dict[str, Any], model: str) -> str:
    """
    Build structured interaction history from an AgentHarm sample.

    Formats the multi-turn interaction with markers for task, reasoning, output,
    actions, and tool results to provide context for the evaluation awareness judge.

    Args:
        sample: Raw sample dict from AgentHarm log
        model: Model name (used to extract reasoning correctly for GPT-5 vs others)

    Returns:
        Structured interaction history string
    """
    parts = []

    # Add task prompt
    parts.append(f"[TASK]\n{sample.get('input', '')}\n")

    turn_num = 0
    for msg in sample.get("messages", []):
        role = msg.get("role", "")

        if role == "assistant":
            turn_num += 1
            content = msg.get("content", [])

            reasoning_text = ""
            visible_text = ""

            if isinstance(content, str):
                # Handle <think> tags (DeepSeek, Qwen) and <SCRATCHPAD_REASONING> tags (scratchpad mode)
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                scratchpad_match = re.search(
                    r"<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>",
                    content,
                    re.DOTALL,
                )
                if think_match:
                    reasoning_text = think_match.group(1).strip()
                elif scratchpad_match:
                    reasoning_text = scratchpad_match.group(1).strip()
                # Remove both types of reasoning tags from visible text
                visible_text = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                )
                visible_text = re.sub(
                    r"<SCRATCHPAD_REASONING>.*?</SCRATCHPAD_REASONING>",
                    "",
                    visible_text,
                    flags=re.DOTALL,
                ).strip()
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "reasoning":
                            # GPT-5: use summary (readable), others: use reasoning
                            if "gpt-5" in model.lower():
                                reasoning_text = item.get("summary", "")
                            else:
                                reasoning_text = item.get("reasoning", "")
                        elif item.get("type") == "text":
                            visible_text = item.get("text", "")

            if reasoning_text:
                parts.append(f"[TURN {turn_num} - REASONING]\n{reasoning_text}\n")
            if visible_text:
                parts.append(f"[TURN {turn_num} - OUTPUT]\n{visible_text}\n")

            # Add tool calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tools = [
                    tc.get("function", "") for tc in tool_calls if isinstance(tc, dict)
                ]
                if tools:
                    parts.append(
                        f"[TURN {turn_num} - ACTIONS]\nCalled: {', '.join(tools)}\n"
                    )

        elif role == "tool":
            result = str(msg.get("content", ""))[:200]
            parts.append(f"[TOOL RESULT]\n{result}...\n")

    return "\n".join(parts)


def extract_reasoning_from_history(interaction_history: str) -> str:
    """Extract reasoning sections from structured interaction history.

    Args:
        interaction_history: Formatted history from build_agentharm_interaction_history

    Returns:
        Combined reasoning text from all turns
    """
    matches = re.findall(
        r"\[TURN \d+ - REASONING\]\n(.*?)(?=\n\[|\Z)",
        interaction_history,
        re.DOTALL,
    )
    return "\n\n".join(m.strip() for m in matches if m.strip())


def prepare_agentharm_sample(
    sample: dict[str, Any],
    model: str,
    judge_template: str,
) -> tuple[str, str, Callable[[dict[str, Any]], dict[str, Any]]] | None:
    """
    Prepare AgentHarm sample data for evaluation.

    Args:
        sample: Raw sample dict from AgentHarm log
        model: Model name
        judge_template: Template for judge prompt

    Returns:
        Tuple of (sample_id, formatted_prompt, result_builder) or None if sample should be skipped.
    """
    sample_id = sample.get("id", "")
    epoch = sample.get("epoch", 1)
    task = sample.get("input", "")

    # Extract AgentHarm scores
    scores = sample.get("scores", {})
    combined = scores.get("combined_scorer", {})
    value = combined.get("value", {})
    agentharm_score = value.get("score", 0.0)
    agentharm_refusal = value.get("refusal", 0.0)

    # Build structured interaction history
    interaction_history = build_agentharm_interaction_history(sample, model)

    if not interaction_history.strip():
        return None

    # Format judge prompt
    reasoning = extract_reasoning_from_history(interaction_history)
    formatted_prompt = judge_template.format(
        task=task or "",
        reasoning=reasoning,
        answer=interaction_history,
    )

    # Result builder closure
    def build_result(judge: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": f"{sample_id}_epoch{epoch}",
            "behavior_id": sample_id,
            "epoch": epoch,
            "task": task,
            "interaction_history": interaction_history,
            "model": model,
            "agentharm_score": agentharm_score,
            "agentharm_refusal": agentharm_refusal,
            "judge": judge.get("judgment"),
            "judge_raw": judge.get("raw"),
        }

    return sample_id, formatted_prompt, build_result


# =============================================================================
# Dataset Loading Functions
# =============================================================================


def load_harmbench_dataset(
    dataset_file: str,
    exclude_categories: list[str] | None = None,
    exclude_behavior_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Load behaviors from a HarmBench CSV dataset file.

    Args:
        dataset_file: Path to CSV file
        exclude_categories: SemanticCategory values to exclude (e.g., ["chemical_biological"])
        exclude_behavior_ids: BehaviorID values to exclude

    Returns list of dicts with keys: Behavior, FunctionalCategory, SemanticCategory,
    Tags, ContextString, BehaviorID, prompt (combined context + behavior).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = (
        dataset_file
        if os.path.isabs(dataset_file)
        else os.path.join(base_dir, dataset_file)
    )
    exclude_cat_set = set(exclude_categories or [])
    exclude_id_set = set(exclude_behavior_ids or [])

    behaviors: list[dict[str, str]] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behavior_id = row.get("BehaviorID", "").strip()
            if behavior_id in exclude_id_set:
                continue
            semantic_cat = row.get("SemanticCategory", "").strip()
            if semantic_cat in exclude_cat_set:
                continue

            behavior = row.get("Behavior", "").strip()
            context_str = row.get("ContextString", "").strip()

            if context_str:
                prompt = f"{context_str}\n\n---\n\n{behavior}"
            else:
                prompt = behavior

            behaviors.append(
                {
                    "Behavior": behavior,
                    "FunctionalCategory": row.get("FunctionalCategory", "").strip(),
                    "SemanticCategory": semantic_cat,
                    "Tags": row.get("Tags", "").strip(),
                    "ContextString": context_str,
                    "BehaviorID": row.get("BehaviorID", "").strip(),
                    "prompt": prompt,
                }
            )

    return behaviors


def load_agenticmisalignment_dataset(
    dataset_dir: str,
    scenarios: list[str] | None = None,
    goal_types: list[str] | None = None,
    urgency_types: list[str] | None = None,
    use_think_tags: bool = False,
) -> list[dict]:
    """
    Load conditions from the agentic misalignment dataset.

    Args:
        dataset_dir: Path to directory containing condition subdirectories
        scenarios: Filter to specific scenarios (e.g., ["murder", "blackmail"])
        goal_types: Filter to specific goal types (e.g., ["explicit", "none"])
        urgency_types: Filter to specific urgency types (e.g., ["replacement"])
        use_think_tags: If True (default), load system_prompt_think.txt (generic think instructions).
                        If False, load system_prompt.txt (has <SCRATCHPAD_REASONING> tag instructions).

    Returns list of dicts with keys:
        - condition_id, scenario, goal_type, goal_value, urgency_type (from metadata)
        - system_prompt, user_prompt, email_content (file contents)
        - prompt (combined user_prompt + email_content for target model)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = (
        dataset_dir
        if os.path.isabs(dataset_dir)
        else os.path.join(base_dir, dataset_dir)
    )

    scenarios_set = set(scenarios) if scenarios else None
    goal_types_set = set(goal_types) if goal_types else None
    urgency_types_set = set(urgency_types) if urgency_types else None

    system_prompt_filename = (
        "system_prompt_think.txt" if use_think_tags else "system_prompt.txt"
    )

    conditions: list[dict] = []

    for entry in sorted(os.listdir(dataset_path)):
        condition_dir = os.path.join(dataset_path, entry)
        if not os.path.isdir(condition_dir):
            continue

        metadata_file = os.path.join(condition_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if scenarios_set and metadata.get("scenario") not in scenarios_set:
            continue
        if goal_types_set and metadata.get("goal_type") not in goal_types_set:
            continue
        if urgency_types_set and metadata.get("urgency_type") not in urgency_types_set:
            continue

        system_prompt_path = os.path.join(condition_dir, system_prompt_filename)
        if not os.path.exists(system_prompt_path):
            system_prompt_path = os.path.join(condition_dir, "system_prompt.txt")

        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        with open(
            os.path.join(condition_dir, "user_prompt.txt"), "r", encoding="utf-8"
        ) as f:
            user_prompt = f.read().strip()
        with open(
            os.path.join(condition_dir, "email_content.txt"), "r", encoding="utf-8"
        ) as f:
            email_content = f.read().strip()

        prompt = f"{user_prompt}\n\n{email_content}"

        conditions.append(
            {
                "condition_id": metadata.get("condition_id"),
                "scenario": metadata.get("scenario"),
                "goal_type": metadata.get("goal_type"),
                "goal_value": metadata.get("goal_value"),
                "urgency_type": metadata.get("urgency_type"),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "email_content": email_content,
                "prompt": prompt,
            }
        )

    return conditions
