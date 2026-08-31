"""
Judge/evaluator functions for evaluating target model responses.

CoT monitoring: analyzes the model's reasoning in <think> tags and uses a
judge model to classify evaluation-awareness signals and task performance.

(Upstream also had post-hoc probe methods — self-report, purpose guessing,
third-person — but they were never wired to eval_bench and were deleted here;
see upstream aisa-group/decomposing-eval-awareness for the full pipeline.)
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from lib.config import DEFAULT_ANTHROPIC_MODELS


def create_judge_client(
    judge_model: str,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
) -> Any:
    """
    Create the async API client for a judge model.

    claude-* models use the native Anthropic API; everything else uses OpenAI.
    """
    if judge_model.startswith("claude"):
        return AsyncAnthropic(api_key=anthropic_api_key, max_retries=5)
    return AsyncOpenAI(api_key=openai_api_key, max_retries=5)


def _strip_json_fences(content: str) -> str:
    """Strip markdown code fences (```json ... ```) around a JSON response."""
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        content = content.removesuffix("```")
    return content.strip()


async def judge_completion_async(
    client: Any,
    model: str,
    prompt: str,
    system: str = "",
) -> str:
    """
    Run a single judge completion, routed by client type.

    OpenAI-style clients keep the original message shape (prompt sent as a
    system message). Anthropic clients send the prompt as a user message —
    the Messages API requires at least one user turn — resolve friendly model
    keys (e.g. "claude-opus-4.5") via DEFAULT_ANTHROPIC_MODELS, and strip
    markdown fences Claude tends to wrap around JSON output.
    """
    if isinstance(client, AsyncAnthropic):
        kwargs: dict[str, Any] = {"system": system} if system else {}
        resp = await client.messages.create(
            model=DEFAULT_ANTHROPIC_MODELS.get(model, model),
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _strip_json_fences(text.strip())

    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "system", "content": prompt}]
    resp = await client.chat.completions.create(model=model, messages=messages)
    return (resp.choices[0].message.content or "").strip()


def extract_think_tags(response: str) -> tuple[str, str]:
    """
    Extract reasoning and answer from <think> tags.

    Args:
        response: Model response potentially containing <think>...</think> tags

    Returns:
        (reasoning, answer) tuple. If no <think> tags, reasoning is empty.
    """
    if not response:
        return "", ""

    if "</think>" in response:
        parts = response.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return reasoning, answer

    return "", response.strip()


NO_REASONING_MSG = "No reasoning present in the model response; nothing to grade."


def mark_no_reasoning_sections(
    parsed: dict[str, Any], reasoning: str
) -> dict[str, Any]:
    """Replace reasoning-based judge sections with N/A when the response has no CoT.

    The judge grades model_awareness and hypothetical_recognition solely from the
    reasoning channel; given an empty channel it returns a confident "No" on no
    evidence, silently corrupting awareness rates (e.g. models whose thinking
    comes back empty). N/A conclusions are counted separately by core.metrics.
    """
    if reasoning.strip() or not isinstance(parsed, dict):
        return parsed
    for key in ("model_awareness", "hypothetical_recognition"):
        if key in parsed:
            section = {
                "conclusion": "N/A",
                "evidence": [],
                "reasoning": NO_REASONING_MSG,
            }
            if key == "model_awareness":
                section["awareness_type"] = "N/A"
            parsed[key] = section
    return parsed


async def evaluate_response_async(
    client,
    model: str,
    prompt: str,
    response: str,
    judge_template: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Evaluate a target model response using a judge model (async).

    Args:
        client: Async judge client (AsyncOpenAI or AsyncAnthropic)
        model: Judge model name (e.g., "gpt-5")
        prompt: Original prompt given to target model
        response: Target model's response
        judge_template: Judge prompt template with {task}, {reasoning}, {answer} placeholders
        max_retries: Max retry attempts for transient errors

    Returns:
        Dict with "judgment" (parsed JSON) and "raw" (raw response string)
    """
    reasoning, answer = extract_think_tags(response)

    formatted_prompt = judge_template.format(
        task=prompt or "",
        reasoning=reasoning,
        answer=answer,
    )

    for attempt in range(max_retries):
        try:
            content = await judge_completion_async(client, model, formatted_prompt)

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw": content}
            parsed = mark_no_reasoning_sections(parsed, reasoning)

            return {"judgment": parsed, "raw": content}
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
                print(
                    f"  [WARN] Judge call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}"
                )
                continue
            raise


async def evaluate_sample(
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
    sample_id: str,
    judge_fn: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
    result_fn: Callable[[dict[str, Any]], dict[str, Any]],
    metrics: list[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    """
    Generic sample evaluator with concurrency control and progress output.

    Args:
        semaphore: Concurrency limiter
        idx: Sample index for progress
        total: Total samples for progress
        sample_id: ID for progress display
        judge_fn: Async function that calls the judge and returns {"judgment": ..., "raw": ...}
        result_fn: Function that takes judge dict and returns the result record
        metrics: List of (judgment_key, display_label) for progress output.
                 Default: [("model_awareness", "awareness"), ("hypothetical_recognition", "hypo")]

    Returns:
        Result dict from result_fn, or None if result_fn returns None
    """
    if metrics is None:
        metrics = [
            ("model_awareness", "awareness"),
            ("hypothetical_recognition", "hypo"),
        ]

    async with semaphore:
        judge = await judge_fn()
        judgment = judge.get("judgment", {})

        # Print progress
        parts = [
            f"{label}={judgment.get(key, {}).get('conclusion', '?')}"
            for key, label in metrics
        ]
        print(f"[{idx}/{total}] {sample_id}: {', '.join(parts)}")

        return result_fn(judge)


async def evaluate_batch_async(
    samples: list[dict[str, Any]],
    sample_evaluator: Callable[
        [Any, dict[str, Any], asyncio.Semaphore, int, int],
        Coroutine[Any, Any, dict[str, Any] | None],
    ],
    api_key: str,
    max_concurrent: int = 100,
    judge_model: str = "",
    anthropic_api_key: str = "",
) -> list[dict[str, Any]]:
    """
    Evaluate a batch of samples with controlled concurrency.

    Generic batch evaluator that can be reused across experiments.
    Each experiment provides its own sample_evaluator function.

    Args:
        samples: List of samples to evaluate
        sample_evaluator: Async function(client, sample, semaphore, idx, total) -> result dict or None
        api_key: OpenAI API key
        max_concurrent: Maximum concurrent evaluations (default: 100)
        judge_model: Judge model key; claude-* routes to the Anthropic API
        anthropic_api_key: Anthropic API key (used for claude-* judges)

    Returns:
        List of result records (filtered for None and exceptions)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(samples)

    async with create_judge_client(judge_model, api_key, anthropic_api_key) as client:
        tasks = [
            sample_evaluator(client, sample, semaphore, idx, total)
            for idx, sample in enumerate(samples, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for r in results:
        if r is None:
            continue
        if isinstance(r, Exception):
            print(f"Task failed: {r}")
            continue
        out.append(r)
    return out
