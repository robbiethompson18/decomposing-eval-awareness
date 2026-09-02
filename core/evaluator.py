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
from lib.retry import MAX_RETRIES


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
        return AsyncAnthropic(api_key=anthropic_api_key, max_retries=MAX_RETRIES)
    return AsyncOpenAI(api_key=openai_api_key, max_retries=MAX_RETRIES)


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
NO_EVIDENCE_MSG = "No evidence in the channels shown to the judge; nothing to grade."

# Which channels the judge was actually shown, per grading mode. The awareness /
# hypothetical sections are forced to N/A only when EVERY shown channel is empty —
# otherwise the judge would return a confident "no" on no evidence, silently
# deflating the rates. Physical masking (tiered grader) makes "shown" real: the
# output tier never sees reasoning, so an empty answer there is N/A even if a CoT
# exists in the transcript, because the judge could not use it.
_NA_MODES = {
    "reasoning": ("reasoning",),  # legacy CoT-only grader
    "answer": ("answer",),  # output-only tier
    "both": ("reasoning", "answer"),  # output+CoT tier
}


def mark_na_when_no_evidence(
    parsed: dict[str, Any], reasoning: str, answer: str, mode: str = "reasoning"
) -> dict[str, Any]:
    """N/A the reasoning-based sections when the judge's shown channels are all empty.

    mode selects which channels the judge saw: "reasoning" (legacy CoT-only),
    "answer" (output tier), or "both" (output+CoT tier). N/A conclusions are
    counted separately by core.metrics.
    """
    if not isinstance(parsed, dict):
        return parsed
    channels = {"reasoning": reasoning, "answer": answer}
    shown = _NA_MODES.get(mode, _NA_MODES["reasoning"])
    if any(channels[c].strip() for c in shown):
        return parsed
    for key in ("model_awareness", "hypothetical_recognition"):
        if key in parsed:
            section = {
                "conclusion": "N/A",
                "evidence": [],
                "reasoning": NO_EVIDENCE_MSG,
            }
            if key == "model_awareness":
                section["awareness_type"] = "N/A"
            parsed[key] = section
    return parsed


def mark_no_reasoning_sections(
    parsed: dict[str, Any], reasoning: str
) -> dict[str, Any]:
    """Back-compat shim for the legacy CoT-only grader (N/A when reasoning empty)."""
    return mark_na_when_no_evidence(parsed, reasoning, answer="", mode="reasoning")


async def evaluate_response_async(
    client,
    model: str,
    prompt: str,
    response: str,
    judge_template: str,
    max_retries: int = 5,
    na_mode: str = "reasoning",
) -> dict[str, Any]:
    """
    Evaluate a target model response using a judge model (async).

    Args:
        client: Async judge client (AsyncOpenAI or AsyncAnthropic)
        model: Judge model name (e.g., "gpt-5")
        prompt: Original prompt given to target model
        response: Target model's response
        judge_template: Judge prompt template with {task}, {reasoning}, {answer} placeholders
            (a template that omits {reasoning} — the output-only tier — just ignores it)
        max_retries: Max retry attempts for transient errors
        na_mode: which channels the judge is shown, for the empty-evidence N/A rule —
            "reasoning" (legacy CoT-only), "answer" (output tier), "both" (output+CoT tier)

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
            parsed = mark_na_when_no_evidence(parsed, reasoning, answer, na_mode)

            return {"judgment": parsed, "raw": content}
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
                print(
                    f"  [WARN] Judge call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}"
                )
                continue
            raise


def judge_failed_na(metrics: list[tuple[str, str]], err: str) -> dict[str, Any]:
    """Verdict for a sample the judge could not grade at all: every metric N/A,
    with the error as the reasoning so it stays distinguishable from a
    no-evidence N/A."""
    reason = f"judge call failed: {err}"
    return {key: {"conclusion": "N/A", "reasoning": reason} for key, _ in metrics}


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
        try:
            judge = await judge_fn()
        except Exception as e:  # noqa: BLE001
            # Retries exhausted — in practice OpenAI's cyber content filter
            # rejecting the transcript. Mark N/A and move on; a dropped row
            # would silently bias the rates on the most harmful items.
            err = f"{type(e).__name__}: {e}"
            print(
                f"[{idx}/{total}] {sample_id}: judge failed ({type(e).__name__}); marked N/A"
            )
            return result_fn(
                {"judgment": judge_failed_na(metrics, err), "raw": f"ERROR: {err}"}
            )
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
    max_concurrent: int = 500,
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
        max_concurrent: Maximum concurrent evaluations (default: 500)
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
