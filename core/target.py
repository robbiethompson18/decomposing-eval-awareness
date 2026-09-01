"""
Target model query functions and API clients.
"""

import os
import re
import sys
from typing import Any

from lib.config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_OPENAI_MODELS,
    DEFAULT_OPENROUTER_CONFIGS,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None  # type: ignore


# ============================================================================
# API Clients
# ============================================================================


def create_openai_client(api_key: str, base_url: str | None = None) -> "OpenAI":
    """
    Create an OpenAI client (also works for OpenAI-compatible APIs like OpenRouter).

    Args:
        api_key: API key
        base_url: Optional custom base URL for compatible APIs

    Returns:
        OpenAI client instance
    """
    if OpenAI is None:
        raise ImportError("openai package is required")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _convert_scratchpad_to_think(content: str) -> str:
    """Convert <SCRATCHPAD_REASONING> tags to <think> tags for consistency."""
    pattern = r"<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        reasoning = match.group(1).strip()
        # Get everything after the closing tag as the answer
        answer = content[match.end() :].strip()
        return f"<think>{reasoning}</think>{answer}"

    # No scratchpad tags found - return as-is (model may not have used them)
    return content


# Claude 5 family rejects thinking.type=enabled/budget_tokens; requires adaptive thinking.
# Note: these models return an EMPTY thinking block (encrypted signature only), so
# summary-mode responses from them contain no <think> content for the judge.
ADAPTIVE_THINKING_MODELS = {"claude-opus-5", "claude-fable-5"}


class AnthropicApiClient:
    """Anthropic API client for Claude models.

    Two modes:
    - default: Extended Thinking API, returns summarized thinking as <think> content
    - scratchpad=True: standard Messages API; the system prompt instructs the model
      to reason in <SCRATCHPAD_REASONING> tags, which are converted to <think>
    """

    def __init__(
        self, cfg: dict[str, Any], system_prompt: str, scratchpad: bool = False
    ) -> None:
        if requests is None:
            raise ImportError("requests package required for Anthropic API")
        self.system_prompt = system_prompt
        self.api_key = cfg["ANTHROPIC_API_KEY"]
        self.model = cfg["ANTHROPIC_MODEL"]
        self.api_base = cfg["ANTHROPIC_API_BASE"]
        self.scratchpad = scratchpad
        # Extended thinking requires budget_tokens < max_tokens
        # Default to 16000 max_tokens with 10000 thinking budget
        self.max_tokens = cfg.get("MAX_TOKENS", 16000)
        self.thinking_budget = cfg.get("THINKING_BUDGET", 10000)
        # Ensure thinking budget is valid (must be < max_tokens, minimum 1024)
        if self.thinking_budget >= self.max_tokens:
            # Allocate ~60% of max_tokens to thinking, rest to output
            self.thinking_budget = max(1024, int(self.max_tokens * 0.6))

    def query(self, prompt: str) -> str:
        return self.query_messages([{"role": "user", "content": prompt}])

    def query_messages(self, messages: list[dict[str, str]]) -> str:
        url = self.api_base.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.scratchpad:
            # No extended thinking - model reasons in visible <SCRATCHPAD_REASONING> tags
            payload["temperature"] = 0.3
            # Claude 5 family defaults to adaptive thinking; force it all the way down
            # so reasoning happens in the visible <SCRATCHPAD_REASONING> tags, not natively.
            # (temperature is deprecated for these models, so drop it.)
            if self.model in ADAPTIVE_THINKING_MODELS:
                payload["output_config"] = {"effort": "low"}
                del payload["temperature"]
        # Extended thinking is not compatible with temperature, so we omit it
        elif self.model in ADAPTIVE_THINKING_MODELS:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": "high"}
        else:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            content_blocks = data.get("content") or []

            # Extract thinking and text content
            thinking_text = ""
            answer_text = ""

            for c in content_blocks:
                if isinstance(c, dict):
                    if c.get("type") == "thinking":
                        # Summarized thinking from Claude 4 models
                        thinking_text = c.get("thinking", "")
                    elif c.get("type") == "text":
                        answer_text += c.get("text", "")

            if self.scratchpad:
                # Convert <SCRATCHPAD_REASONING> to <think> for consistency
                return _convert_scratchpad_to_think(answer_text).strip()

            # Format in same <think> structure as other models
            if thinking_text:
                return f"<think>{thinking_text}</think>{answer_text}".strip()
            return answer_text.strip()
        except Exception as e:
            return f"ERROR calling Anthropic API: {e}"


class OpenAITargetClient:
    """OpenAI API client for GPT-5 models.

    Two modes:
    - default: Responses API with human-readable reasoning summaries as <think> content
    - scratchpad=True: standard Chat Completions API; GPT-5 may use internal reasoning
      instead of the explicit <SCRATCHPAD_REASONING> tags the system prompt asks for
    """

    # TODO: force reasoning to as low as possible (0 for A\, minimal for GPT) to encourage use of the scratchpad

    def __init__(
        self, cfg: dict[str, Any], system_prompt: str, scratchpad: bool = False
    ) -> None:
        from openai import OpenAI

        self.system_prompt = system_prompt
        self.model = cfg["OPENAI_TARGET_MODEL"]
        self.max_tokens = cfg.get("MAX_TOKENS", 7000)
        self.scratchpad = scratchpad
        self.client = OpenAI(api_key=cfg["OPENAI_API_KEY"])

    def query(self, prompt: str) -> str:
        return self.query_messages([{"role": "user", "content": prompt}])

    def query_messages(self, messages: list[dict[str, str]]) -> str:
        try:
            if self.scratchpad:
                # Standard Chat Completions API (not Responses API)
                # GPT-5 requires: max_completion_tokens (not max_tokens), no temperature param
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *messages,
                    ],
                    max_completion_tokens=self.max_tokens,
                )
                choice = resp.choices[0] if resp.choices else None
                content = (choice.message.content or "") if choice else ""

                # Convert <SCRATCHPAD_REASONING> to <think> for consistency (if present)
                return _convert_scratchpad_to_think(content).strip()

            # Use Responses API with reasoning summary for GPT-5
            # This provides human-readable reasoning summaries instead of encrypted reasoning
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": self.system_prompt},
                    *messages,
                ],
                reasoning={
                    "effort": "high",
                    "summary": "detailed",
                },
                max_output_tokens=self.max_tokens,
            )

            # Extract reasoning summary and answer from response
            reasoning_summary = ""
            answer = ""

            for item in resp.output:
                if item.type == "reasoning":
                    # Get summary text from reasoning item
                    if hasattr(item, "summary") and item.summary:
                        reasoning_summary = "\n".join(
                            s.text for s in item.summary if hasattr(s, "text")
                        )
                elif item.type == "message":
                    # Get answer text from message item
                    for content in item.content:
                        if hasattr(content, "text"):
                            answer += content.text

            # Format in same <think> structure as other models
            if reasoning_summary:
                return f"<think>{reasoning_summary}</think>{answer}".strip()
            return answer.strip()
        except Exception as e:
            return f"ERROR calling OpenAI API target: {e}"


class OpenRouterApiClient:
    """OpenRouter API client (OpenAI-compatible) for various models.

    Supports multiple models via cfg["OPENROUTER_MODEL"].
    Temperature is optional - set cfg["TEMPERATURE"] = None for reasoning models that don't support it.
    Native reasoning mode is always enabled for better reasoning extraction.
    """

    def __init__(self, cfg: dict[str, Any], system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.model = cfg["OPENROUTER_MODEL"]
        self.max_tokens = cfg.get("MAX_TOKENS", 2048)
        self.temperature = cfg.get(
            "TEMPERATURE", 0.3
        )  # None = don't set (for reasoning models)
        self.client = create_openai_client(
            cfg["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"
        )

    def query(self, prompt: str) -> str:
        return self.query_messages([{"role": "user", "content": prompt}])

    def query_messages(self, messages: list[dict[str, str]]) -> str:
        text, _meta = self.query_messages_meta(messages)
        return text

    def query_messages_meta(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        """Query and also return per-call metadata (serving provider, token usage)."""
        try:
            # Build request kwargs - temperature is optional for reasoning models
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    *messages,
                ],
                "max_tokens": self.max_tokens,
                "stream": False,
                "extra_body": {"reasoning": {"effort": "high"}},
            }

            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0] if resp.choices else None
            content = (choice.message.content or "") if choice else ""

            # `reasoning` can exist but be None (provider returned no reasoning);
            # without `or ""` that f-strings into a literal "<think>None</think>"
            reasoning = (
                (getattr(choice.message, "reasoning", None) or "") if choice else ""
            )
            provider = getattr(resp, "provider", None)
            usage = getattr(resp, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            meta = {
                "provider": provider,
                "reasoning_chars": len(reasoning),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "finish_reason": getattr(choice, "finish_reason", None)
                if choice
                else None,
            }
            print(
                f"    [OpenRouter] API call successful (model={self.model}, "
                f"provider={provider}, reasoning_chars={len(reasoning)}, "
                f"reasoning_tokens={meta['reasoning_tokens']}, content_len={len(content)})"
            )
            return f"<think>{reasoning}</think>{content}".strip(), meta
        except Exception as e:
            return f"ERROR calling OpenRouter API: {e}", {"error": str(e)}


# ============================================================================
# Client Factory
# ============================================================================


def create_target_client(
    backend: str,
    cfg: dict[str, Any],
    system_prompt: str,
    reasoning_mode: str = "summary",
    verbose: bool = True,
):
    """Create target client based on backend selection.

    Args:
        backend: Backend name (anthropic_api, openai_api, openrouter)
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

    if b == "anthropic_api":
        model_key = os.getenv("ANTHROPIC_MODEL_KEY", "claude-sonnet-4.5")
        if model_key in DEFAULT_ANTHROPIC_MODELS:
            cfg["ANTHROPIC_MODEL"] = DEFAULT_ANTHROPIC_MODELS[model_key]
        if verbose:
            mode = "scratchpad mode" if use_scratchpad else "extended thinking"
            print(f"Using Anthropic API backend ({mode}, model={model_key})...")
        client = AnthropicApiClient(cfg, system_prompt, scratchpad=use_scratchpad)
        return client, model_key
    if b == "openai_api":
        model_key = os.getenv("OPENAI_MODEL_KEY", "gpt-5")
        if model_key in DEFAULT_OPENAI_MODELS:
            cfg["OPENAI_TARGET_MODEL"] = DEFAULT_OPENAI_MODELS[model_key]
        if verbose:
            mode = "scratchpad mode" if use_scratchpad else "reasoning summary"
            print(f"Using OpenAI API backend ({mode}, model={model_key})...")
        client = OpenAITargetClient(cfg, system_prompt, scratchpad=use_scratchpad)
        return client, model_key
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
        f"ERROR: Unsupported TARGET_BACKEND '{backend}'. Use: anthropic_api, openai_api, openrouter"
    )
    sys.exit(1)
