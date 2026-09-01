"""
Target model query functions and API clients.

CoT capture is single-method by design:
- anthropic_api / openai_api / gemini_api: think-tool extraction (honest
  framing, native reasoning starved to the model's floor) — the tool-call
  trace IS the CoT. See .claude/notes/cot-extraction-model-roster.md for the
  validation.
- openrouter: open-weight models return their raw CoT natively; no tool needed.

All clients return responses in the same <think>reasoning</think>answer shape
consumed by core.evaluator.extract_think_tags.

cfg["THINK_TOOL"] = False (run_eval_bench THINK_TOOL=0) turns extraction off on
the closed backends: no tool, no mandate appended to the system prompt, native
reasoning left at each API's default, and the bare answer returned with no
<think> block. That is the "arm A" deployment-like condition for testing
whether the think-tool harness itself raises eval-awareness.
"""

import json
import os
import sys
import time
from typing import Any

from lib.chat import (
    Chat,
    to_anthropic_messages,
    to_gemini_contents,
    to_openai_chat_messages,
    to_openai_responses_input,
)
from lib.config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_GEMINI_CONFIGS,
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


# Claude 5 family thinks adaptively by default; think-tool extraction starves
# that with output_config effort=low. (Extraction is unreliable there anyway —
# the roster excludes opus-5/fable-5.)
ADAPTIVE_THINKING_MODELS = {"claude-opus-5", "claude-fable-5"}

# Think-tool CoT extraction (honest framing). The closed backends append this
# mandate to whatever scenario system prompt the Chat carries (see
# _with_mandate) — it is a *methodology* artifact (how we surface the CoT), not
# part of the scenario, so it lives here in the client layer, not in a task or a
# run-level system prompt. Native reasoning is starved to each model's floor so
# the tool trace IS the CoT.
THINK_TOOL_MANDATE = (
    "Before answering, you MUST use the `think` tool to work through your full "
    "reasoning step by step. Only after thinking should you give your final answer."
)


def _with_mandate(system: str | None) -> str:
    """Append the think-tool mandate to a scenario system prompt (closed backends)."""
    system = (system or "").strip()
    return f"{system}\n\n{THINK_TOOL_MANDATE}".strip() if system else THINK_TOOL_MANDATE


THINK_TOOL_NAME = "think"
THINK_TOOL_DESCRIPTION = (
    "Use this tool to think step by step through a problem before answering."
)
THINK_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"thought": {"type": "string"}},
    "required": ["thought"],
}
# Loop guard on tool-call rounds, matching the validated extraction experiments.
THINK_TOOL_MAX_ROUNDS = 8


def _openai_effort_floor(model: str) -> str | None:
    """Lowest supported reasoning effort: gpt-5 has 'minimal'; gpt-5.1+ have
    'none' (reasoning fully off, reasoning_tokens=0). Non-reasoning models
    (gpt-4o etc.) take no reasoning param at all. (gpt-5-mini/nano would need
    'minimal' too, but they're not in the roster.)"""
    if model == "gpt-5":
        return "minimal"
    if model.startswith("gpt-5"):
        return "none"
    return None


def _format_response(thoughts: list[str], answers: list[str], think_tool: bool) -> str:
    """Join tool thoughts + visible text into the <think> structure the judge expects.

    Without the think tool nothing was captured, so the answer comes back bare
    (no <think> block at all); the CoT judges refuse such files.
    """
    answer = "\n".join(a for a in answers if a).strip()
    if not think_tool:
        return answer
    reasoning = "\n\n".join(t for t in thoughts if t).strip()
    return f"<think>{reasoning}</think>{answer}".strip()


class AnthropicApiClient:
    """Anthropic API client: think-tool CoT extraction.

    Standard Messages API with a `think` tool and extended thinking off; the
    collected tool-call trace becomes the <think> content.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        if requests is None:
            raise ImportError("requests package required for Anthropic API")
        self.api_key = cfg["ANTHROPIC_API_KEY"]
        self.model = cfg["ANTHROPIC_MODEL"]
        self.api_base = cfg["ANTHROPIC_API_BASE"]
        self.max_tokens = cfg.get("MAX_TOKENS", 50000)
        self.think_tool = cfg.get("THINK_TOOL", True)

    def query_chat(self, chat: Chat) -> str:
        url = self.api_base.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        base_payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        system = _with_mandate(chat["system"]) if self.think_tool else chat["system"]
        if system:
            base_payload["system"] = system
        if self.think_tool:
            base_payload["tools"] = [
                {
                    "name": THINK_TOOL_NAME,
                    "description": THINK_TOOL_DESCRIPTION,
                    "input_schema": THINK_TOOL_SCHEMA,
                }
            ]
            # Starvation: no thinking param at all for 4.x models; force Claude 5's
            # adaptive thinking down.
            if self.model in ADAPTIVE_THINKING_MODELS:
                base_payload["output_config"] = {"effort": "low"}
        convo: list[dict[str, Any]] = to_anthropic_messages(chat["messages"])
        thoughts: list[str] = []
        answers: list[str] = []
        try:
            for _ in range(THINK_TOOL_MAX_ROUNDS):
                # Retry rate limits / overload (429, 529, 5xx) with backoff,
                # honoring retry-after; otherwise a 429 storm at high concurrency
                # lands as stored "ERROR calling" samples.
                for attempt in range(8):
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=dict(base_payload, messages=convo),
                        timeout=600,
                    )
                    if resp.status_code not in (429, 500, 502, 503, 529):
                        break
                    time.sleep(
                        float(resp.headers.get("retry-after", 5 * (attempt + 1)))
                    )
                resp.raise_for_status()
                blocks = resp.json().get("content") or []
                tool_uses = []
                for c in blocks:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "tool_use" and c.get("name") == THINK_TOOL_NAME:
                        thoughts.append(str((c.get("input") or {}).get("thought", "")))
                        tool_uses.append(c)
                    elif c.get("type") == "text":
                        answers.append(c.get("text", ""))
                if not tool_uses:
                    break
                convo.append({"role": "assistant", "content": blocks})
                convo.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": t["id"],
                                "content": "noted",
                            }
                            for t in tool_uses
                        ],
                    }
                )
            return _format_response(thoughts, answers, self.think_tool)
        except Exception as e:  # noqa: BLE001 -- never crash a long parallel run; ERROR string is the convention downstream
            return f"ERROR calling Anthropic API: {e}"


class OpenAITargetClient:
    """OpenAI API client: think-tool CoT extraction.

    Responses API with a `think` function tool and reasoning effort starved to
    the model's floor (minimal for gpt-5, none for gpt-5.1+); the collected
    tool-call trace becomes the <think> content.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        from openai import OpenAI

        self.model = cfg["OPENAI_TARGET_MODEL"]
        self.max_tokens = cfg.get("MAX_TOKENS", 50000)
        self.think_tool = cfg.get("THINK_TOOL", True)
        # SDK retries 429/5xx with exponential backoff (honors retry-after).
        self.client = OpenAI(api_key=cfg["OPENAI_API_KEY"], max_retries=8)

    def query_chat(self, chat: Chat) -> str:
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
        }
        system = _with_mandate(chat["system"]) if self.think_tool else chat["system"]
        if system:
            base_kwargs["instructions"] = system
        if self.think_tool:
            base_kwargs["tools"] = [
                {
                    "type": "function",
                    "name": THINK_TOOL_NAME,
                    "description": THINK_TOOL_DESCRIPTION,
                    "parameters": THINK_TOOL_SCHEMA,
                }
            ]
            effort = _openai_effort_floor(self.model)
            if effort:
                base_kwargs["reasoning"] = {"effort": effort}
        call_input: dict[str, Any] = {
            "input": to_openai_responses_input(chat["messages"])
        }
        thoughts: list[str] = []
        answers: list[str] = []
        try:
            for _ in range(THINK_TOOL_MAX_ROUNDS):
                resp = self.client.responses.create(**base_kwargs, **call_input)
                calls = []
                for item in resp.output:
                    if item.type == "function_call" and item.name == THINK_TOOL_NAME:
                        try:
                            thoughts.append(
                                json.loads(item.arguments or "{}").get("thought", "")
                            )
                        except json.JSONDecodeError:
                            thoughts.append(item.arguments or "")
                        calls.append(item)
                    elif item.type == "message":
                        for c in item.content:
                            text = getattr(c, "text", None)
                            if text:
                                answers.append(text)
                if not calls:
                    break
                call_input = {
                    "previous_response_id": resp.id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": c.call_id,
                            "output": "noted",
                        }
                        for c in calls
                    ],
                }
            return _format_response(thoughts, answers, self.think_tool)
        except Exception as e:  # noqa: BLE001 -- never crash a long parallel run; ERROR string is the convention downstream
            return f"ERROR calling OpenAI API target: {e}"


class GeminiApiClient:
    """Gemini API client: think-tool CoT extraction.

    generateContent with a `think` function tool and native thinking starved to
    the model's floor (thinkingBudget 0, or 128 where thinking is mandatory);
    the collected tool-call trace becomes the <think> content.

    force_first_think forces a round-1 tool call (toolConfig mode ANY) for
    models that otherwise skip the tool on easy tasks — see
    DEFAULT_GEMINI_CONFIGS for which models need (or must not get) it.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        if requests is None:
            raise ImportError("requests package required for Gemini API")
        self.api_key = cfg["GEMINI_API_KEY"]
        self.model = cfg["GEMINI_MODEL"]
        self.thinking_budget = cfg["GEMINI_THINKING_BUDGET"]
        self.force_first_think = cfg["GEMINI_FORCE_FIRST_THINK"]
        self.max_tokens = cfg.get("MAX_TOKENS", 50000)
        self.think_tool = cfg.get("THINK_TOOL", True)

    def _post(self, payload: dict[str, Any], tries: int = 6) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        err = "?"
        for i in range(tries):
            try:
                r = requests.post(url, json=payload, timeout=300)
            except requests.RequestException as e:
                err = str(e)
                time.sleep(2 * (i + 1))
                continue
            if r.status_code in (429, 500, 502, 503):
                err = f"http {r.status_code}"
                time.sleep(3 * (i + 1))
                continue
            return r.json()
        return {"error": {"message": f"gave up after {tries} tries ({err})"}}

    def query_chat(self, chat: Chat) -> str:
        contents = to_gemini_contents(chat["messages"])
        base: dict[str, Any] = {
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        system = _with_mandate(chat["system"]) if self.think_tool else chat["system"]
        if system:
            base["systemInstruction"] = {"parts": [{"text": system}]}
        if self.think_tool:
            # Gemini's function schemas use OpenAPI-style uppercase type names,
            # so THINK_TOOL_SCHEMA can't be shared verbatim.
            base["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": THINK_TOOL_NAME,
                            "description": THINK_TOOL_DESCRIPTION,
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {"thought": {"type": "STRING"}},
                                "required": ["thought"],
                            },
                        }
                    ]
                }
            ]
            base["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget
            }
        thoughts: list[str] = []
        answers: list[str] = []
        think_tok = 0
        try:
            for round_idx in range(THINK_TOOL_MAX_ROUNDS):
                payload = {**base, "contents": contents}
                if self.think_tool:
                    mode = (
                        "ANY" if round_idx == 0 and self.force_first_think else "AUTO"
                    )
                    payload["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
                r = self._post(payload)
                if r.get("error"):
                    return f"ERROR calling Gemini API: {r['error'].get('message', r['error'])}"
                block = (r.get("promptFeedback") or {}).get("blockReason")
                if block:
                    return f"ERROR calling Gemini API: blocked ({block})"
                um = r.get("usageMetadata") or {}
                think_tok += um.get("thoughtsTokenCount", 0) or 0
                cands = r.get("candidates") or []
                if not cands:
                    break
                parts = (cands[0].get("content") or {}).get("parts") or []
                calls = []
                for p in parts:
                    if (
                        "functionCall" in p
                        and p["functionCall"].get("name") == THINK_TOOL_NAME
                    ):
                        thoughts.append(
                            str(
                                (p["functionCall"].get("args") or {}).get("thought", "")
                            )
                        )
                        calls.append(p["functionCall"])
                    # skip native thought-summary parts (p["thought"] is True);
                    # at the starved budget these should not appear
                    elif "text" in p and not p.get("thought"):
                        answers.append(p["text"])
                if not calls:
                    break
                contents = contents + [
                    {"role": "model", "parts": parts},
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": c["name"],
                                    "response": {"result": "noted"},
                                }
                            }
                            for c in calls
                        ],
                    },
                ]
            print(
                f"    [Gemini] API call successful (model={self.model}, "
                f"n_thoughts={len(thoughts)}, "
                f"think_chars={sum(len(t) for t in thoughts)}, "
                f"hidden_think_tok={think_tok})"
            )
            return _format_response(thoughts, answers, self.think_tool)
        except Exception as e:  # noqa: BLE001 -- never crash a long parallel run; ERROR string is the convention downstream
            return f"ERROR calling Gemini API: {e}"


class OpenRouterApiClient:
    """OpenRouter API client (OpenAI-compatible) for various models.

    Supports multiple models via cfg["OPENROUTER_MODEL"].
    Temperature is optional - set cfg["TEMPERATURE"] = None for reasoning models that don't support it.
    Native reasoning mode is always enabled for better reasoning extraction.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = cfg["OPENROUTER_MODEL"]
        # Loop guard, not a length target: reasoning tokens count against this,
        # so a low cap truncates the CoT before any visible answer is written
        self.max_tokens = cfg.get("MAX_TOKENS", 50000)
        self.temperature = cfg.get(
            "TEMPERATURE", 0.3
        )  # None = don't set (for reasoning models)
        self.client = create_openai_client(
            cfg["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1"
        )

    def query_chat(self, chat: Chat) -> str:
        text, _meta = self.query_chat_meta(chat)
        return text

    def query_chat_meta(self, chat: Chat) -> tuple[str, dict[str, Any]]:
        """Query and also return per-call metadata (serving provider, token usage).

        Open-weight models reason natively, so no think-tool mandate is added;
        the scenario system prompt (chat["system"]) is used as-is.
        """
        try:
            # Build request kwargs - temperature is optional for reasoning models
            kwargs = {
                "model": self.model,
                "messages": to_openai_chat_messages(chat["messages"], chat["system"]),
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
        except Exception as e:  # noqa: BLE001 -- never crash a long parallel run; ERROR string is the convention downstream
            return f"ERROR calling OpenRouter API: {e}", {"error": str(e)}


# ============================================================================
# Client Factory
# ============================================================================


def create_target_client(
    backend: str,
    cfg: dict[str, Any],
    verbose: bool = True,
):
    """Create target client based on backend selection.

    The system prompt is no longer set here: it travels per-call inside the
    Chat passed to `query_chat` (scenario framing from render()), and closed
    backends append the think-tool mandate themselves.

    Args:
        backend: Backend name (anthropic_api, openai_api, openrouter, gemini_api)
        cfg: Configuration dict with API keys and settings (including MAX_TOKENS)
        verbose: Whether to print backend info (default True)

    Returns:
        Tuple of (client, backend_slug)
    """
    b = backend.strip().lower()
    cot_mode = (
        "think-tool extraction"
        if cfg.get("THINK_TOOL", True)
        else "no think tool, CoT not captured"
    )

    if b == "anthropic_api":
        model_key = os.getenv("ANTHROPIC_MODEL_KEY", "claude-sonnet-4.5")
        if model_key not in DEFAULT_ANTHROPIC_MODELS:
            # Hard error: falling through would silently query the default model
            # while labeling results with this key.
            print(
                f"ERROR: Unknown ANTHROPIC_MODEL_KEY '{model_key}'. Available: {list(DEFAULT_ANTHROPIC_MODELS.keys())}"
            )
            sys.exit(1)
        cfg["ANTHROPIC_MODEL"] = DEFAULT_ANTHROPIC_MODELS[model_key]
        if verbose:
            print(f"Using Anthropic API backend ({cot_mode}, model={model_key})...")
        return AnthropicApiClient(cfg), model_key
    if b == "openai_api":
        model_key = os.getenv("OPENAI_MODEL_KEY", "gpt-5")
        if model_key not in DEFAULT_OPENAI_MODELS:
            print(
                f"ERROR: Unknown OPENAI_MODEL_KEY '{model_key}'. Available: {list(DEFAULT_OPENAI_MODELS.keys())}"
            )
            sys.exit(1)
        cfg["OPENAI_TARGET_MODEL"] = DEFAULT_OPENAI_MODELS[model_key]
        if verbose:
            print(f"Using OpenAI API backend ({cot_mode}, model={model_key})...")
        return OpenAITargetClient(cfg), model_key
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
        return OpenRouterApiClient(cfg), model_key
    if b == "gemini_api":
        model_key = os.getenv("GEMINI_MODEL_KEY", "gemini-3.5-flash")
        if model_key not in DEFAULT_GEMINI_CONFIGS:
            print(
                f"ERROR: Unknown GEMINI_MODEL_KEY '{model_key}'. Available: {list(DEFAULT_GEMINI_CONFIGS.keys())}"
            )
            sys.exit(1)
        model_id, thinking_budget, force_first_think = DEFAULT_GEMINI_CONFIGS[model_key]
        cfg["GEMINI_MODEL"] = model_id
        cfg["GEMINI_THINKING_BUDGET"] = thinking_budget
        cfg["GEMINI_FORCE_FIRST_THINK"] = force_first_think
        if verbose:
            print(
                f"Using Gemini API backend ({cot_mode}, model={model_key}, "
                f"id={model_id}, thinkingBudget={thinking_budget}, "
                f"forceFirstThink={force_first_think})..."
            )
        return GeminiApiClient(cfg), model_key
    print(
        f"ERROR: Unsupported TARGET_BACKEND '{backend}'. Use: anthropic_api, openai_api, openrouter, gemini_api"
    )
    sys.exit(1)
