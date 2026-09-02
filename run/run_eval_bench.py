"""
Run eval_bench tasks across all models.

For each task (safety + capability variant), runs:
  - 1 baseline config (all factors at naturalistic level)
  - 12 single-factor ablations (one factor varied, rest naturalistic)
  - 2 anchor configs off the OFAT grid:
      all_varied          every factor at its varied level (the grid's ceiling)
      harmbench_verbatim  the original HarmBench behavior text, safety arm only
                          (see eval_bench/harmbench.py)

Each config is sampled SAMPLES_PER_CONFIG times per model.
Results are saved incrementally after each task, so runs can be resumed.

Usage:
  # Run a single model via OpenRouter (most models):
  TARGET_BACKEND=openrouter OPENROUTER_MODEL_KEY=kimi-k2.5 SAMPLES_PER_CONFIG=3 python run/run_eval_bench.py

  # Run Claude via native Anthropic API:
  TARGET_BACKEND=anthropic_api ANTHROPIC_MODEL_KEY=claude-sonnet-4.5 SAMPLES_PER_CONFIG=3 python run/run_eval_bench.py

  # Run GPT-5 via native OpenAI API:
  TARGET_BACKEND=openai_api OPENAI_MODEL_KEY=gpt-5 SAMPLES_PER_CONFIG=3 python run/run_eval_bench.py

  # Run Gemini via native API:
  TARGET_BACKEND=gemini_api GEMINI_MODEL_KEY=gemini-3.5-flash SAMPLES_PER_CONFIG=3 python run/run_eval_bench.py

  # Resume a crashed run (auto-detects existing file and skips completed samples):
  Just re-run the same command. The script finds the latest matching output file and resumes.

Environment variables:
  TARGET_BACKEND        Backend to use (openrouter, anthropic_api, openai_api, gemini_api)
  OPENROUTER_MODEL_KEY  Model key for OpenRouter (default: qwen3-30b)
  ANTHROPIC_MODEL_KEY   Model key for Anthropic (default: claude-sonnet-4.5)
  OPENAI_MODEL_KEY      Model key for OpenAI (default: gpt-5)
  GEMINI_MODEL_KEY      Model key for Gemini (default: gemini-3.5-flash)
  SAMPLES_PER_CONFIG    Samples per (task, config) pair (default: 3)
  MAX_CONCURRENT        Max parallel API requests (default: 500)
  MAX_TOKENS            Max tokens for model response (default: 50000)
  TASK_FILTER           Comma-separated task names to run (default: all tasks)
  VALENCE_FILTER        Comma-separated valences to run, e.g. "capability" (default: all)
  CONFIG_FILTER         Comma-separated config names to run, e.g. "F6" or "baseline,F1" or
                        "all_varied,harmbench_verbatim" (default: all)
  RUN_TAG               Extra filename suffix, e.g. "f11test" -> responses_<model>_f11test_<ts>.json.
                        Runs resume by sample id, so after a prompt/factor change a tag is how you
                        get a FRESH file instead of resuming into (and skipping) stale samples.
  THINK_TOOL            "0" disables think-tool CoT extraction on the closed backends
                        (no think tool, no mandate, native reasoning at API default, no
                        <think> block in the response). Default: "1".
  F9_BASELINE           F9 level used as the baseline system prompt: deployment
                        (default), bare, or none (no system prompt at all).

  Arm A of the think-tool-confound test (no harness at all, self-report only):
    THINK_TOOL=0 F9_BASELINE=none CONFIG_FILTER=baseline TARGET_BACKEND=... python run/run_eval_bench.py
  Its output file is suffixed _no_think_tool_f9none so it never resumes into an
  arm-B (think-tool) file. See .claude/notes/think-tool-harness-vs-selfreport.md.
"""

import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.target import create_target_client
from eval_bench.harmbench import (
    HARMBENCH_VERBATIM,
    harmbench_factor_config,
    harmbench_verbatim_chat,
    load_harmbench_subset,
)
from eval_bench.render import F9_LEVELS, render
from lib.config import get_api_keys_config
from lib.parallel import query_parallel
from lib.time_utils import now_pacific

BASE_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = BASE_DIR / "eval_bench" / "tasks"
RESULTS_DIR = BASE_DIR / "results" / "eval_bench" / "response"

BASELINE_CONFIG = {
    "F1": "realistic",
    "F2": "realistic",
    "F3": "casual",
    "F4": "realistic",
    "F5": "realistic",
    "F6": "realistic",
    "F7": "prose",
    "F8": "realistic",
    # F9 (deployment framing) sets the system prompt. Baseline = a realistic
    # product system prompt; the ablation swaps in the bland benchmark one.
    "F9": "deployment",
    # F10 (deployment date) sets the {TIMESTAMP} in the deployment prompt.
    # Baseline = the real current time; the ablation shifts it back one year.
    "F10": "current",
    # F11 (conversation history). Baseline = the task arrives as turn 2, after a
    # throwaway "are you there?" exchange; the ablation sends it as turn 1.
    "F11": "warm",
    # F12 (tool affordances). Baseline = a web_search tool in the request, as
    # every real chat product ships (still a stub that returns an error on
    # every call — see eval_bench/render.py); the ablation sends no tools.
    "F12": "search",
}

FACTOR_VARIED_LEVELS = {
    "F1": "varied",
    "F2": "varied",
    "F3": "formal",
    "F4": "varied",
    "F5": "varied",
    "F6": "varied",
    "F7": "template",
    "F8": "varied",
    "F9": "bare",
    "F10": "stale",
    "F11": "cold",
    "F12": "none",
}


ALL_VARIED = "all_varied"


def build_configs(f9_baseline: str = BASELINE_CONFIG["F9"]) -> list[tuple]:
    """Build 15 configs: baseline + 12 single-factor ablations + 2 anchors.

    The anchors are off the OFAT grid. `all_varied` is the grid's far corner;
    `harmbench_verbatim` is not a render at all (its Chat and factor_config are
    built per task from the vendored HarmBench row, safety arm only), so its
    entry here carries the standard-behavior factor vector as a placeholder for
    config_list and is special-cased in the item loop.

    f9_baseline overrides the baseline system-prompt level (F9_BASELINE env);
    the F9 ablation still flips to `bare`, whatever the baseline is.
    """
    baseline = dict(BASELINE_CONFIG, F9=f9_baseline)
    configs = [("baseline", baseline)]
    for factor, varied_level in FACTOR_VARIED_LEVELS.items():
        cfg = dict(baseline)
        cfg[factor] = varied_level
        configs.append((factor, cfg))
    configs.append((ALL_VARIED, dict(FACTOR_VARIED_LEVELS)))
    configs.append((HARMBENCH_VERBATIM, harmbench_factor_config({"ContextString": ""})))
    return configs


def load_all_tasks() -> list[dict[str, Any]]:
    """Load all task YAML files from eval_bench/tasks/."""
    tasks = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        for yaml_file in sorted(task_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                template = yaml.safe_load(f)
            tasks.append(
                {
                    "task_name": task_dir.name,
                    "valence": yaml_file.stem,
                    "template": template,
                    "task_id": template.get(
                        "task_id", f"{task_dir.name}_{yaml_file.stem}"
                    ),
                }
            )
    return tasks


def get_output_path(model_slug: str, mode_suffix: str) -> Path:
    """Get the output file path for a model, reusing existing file if present.

    Closed backends carry a _think_tool suffix so their files stay distinct
    from pre-collapse summary/scratchpad results; OpenRouter (native CoT) keeps
    the unsuffixed naming, so its existing result files remain resumable.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Match the timestamp directly after the suffix: a plain glob would let an
    # unsuffixed (native) run resume into a think_tool file, or a
    # `_no_think_tool` (F9=deployment) run resume into `_no_think_tool_f9none`.
    stamped = re.compile(
        rf"responses_{re.escape(model_slug)}{re.escape(mode_suffix)}_\d{{8}}_\d{{6}}\.json$"
    )
    existing = sorted(
        p for p in glob.glob(str(RESULTS_DIR / "responses_*.json")) if stamped.search(p)
    )
    if existing:
        return Path(existing[-1])
    timestamp = now_pacific().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"responses_{model_slug}{mode_suffix}_{timestamp}.json"


def load_existing_results(out_file: Path) -> tuple[dict[str, Any], set[str]]:
    """Load existing results file and return (data_dict, set_of_completed_ids).

    Samples whose response is an "ERROR calling ..." string (API failure) are
    dropped so a resume regenerates them instead of treating them as done.
    """
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        kept = [
            s
            for s in samples
            if not str(s.get("response", "")).startswith("ERROR calling")
        ]
        if len(kept) < len(samples):
            print(
                f"Dropping {len(samples) - len(kept)} errored samples for regeneration"
            )
        data["samples"] = kept
        return data, {s["id"] for s in kept}
    return None, set()


def save_results(out_file: Path, output_data: dict[str, Any]) -> None:
    """Atomically save results (write to temp then rename)."""
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    tmp.rename(out_file)


def main() -> None:
    samples_per_config = int(os.getenv("SAMPLES_PER_CONFIG", "3"))
    think_tool = os.getenv("THINK_TOOL", "1") != "0"
    f9_baseline = os.getenv("F9_BASELINE", BASELINE_CONFIG["F9"])
    if f9_baseline not in F9_LEVELS:
        print(f"ERROR: Unknown F9_BASELINE '{f9_baseline}'. Available: {F9_LEVELS}")
        sys.exit(1)

    cfg = {
        "MAX_TOKENS": int(os.getenv("MAX_TOKENS", "50000")),
        "TARGET_BACKEND": os.getenv("TARGET_BACKEND", "openrouter"),
        "MAX_CONCURRENT": int(os.getenv("MAX_CONCURRENT", "500")),
        "THINK_TOOL": think_tool,
        **get_api_keys_config(),
    }

    backend = cfg["TARGET_BACKEND"]
    # The system prompt is no longer set at the run level: it travels per-sample
    # inside each task's Chat (render() picks it from the F9 factor), and closed
    # backends append the think-tool mandate unless THINK_TOOL=0.
    client, model_slug = create_target_client(backend, cfg)

    tasks = load_all_tasks()
    configs = build_configs(f9_baseline)
    harmbench_rows = load_harmbench_subset()

    task_filter = os.getenv("TASK_FILTER", "")
    valence_filter = os.getenv("VALENCE_FILTER", "")
    config_filter = os.getenv("CONFIG_FILTER", "")
    if task_filter:
        allowed_tasks = {t.strip() for t in task_filter.split(",")}
        tasks = [t for t in tasks if t["task_name"] in allowed_tasks]
    if valence_filter:
        allowed_valences = {v.strip() for v in valence_filter.split(",")}
        tasks = [t for t in tasks if t["valence"] in allowed_valences]
    if config_filter:
        allowed_configs = {c.strip() for c in config_filter.split(",")}
        configs = [(name, c) for name, c in configs if name in allowed_configs]

    n_task_sets = len({t["task_name"] for t in tasks})

    # Determine output file and load existing progress. The suffix encodes
    # every run-level choice that changes what a sample id means, so a run
    # never resumes into a file generated under a different harness.
    if backend == "openrouter":
        reasoning_method, mode_suffix = "native", ""
    elif think_tool:
        reasoning_method, mode_suffix = "think_tool", "_think_tool"
    else:
        reasoning_method, mode_suffix = "none", "_no_think_tool"
    if f9_baseline != BASELINE_CONFIG["F9"]:
        mode_suffix += f"_f9{f9_baseline}"
    if os.getenv("RUN_TAG"):
        mode_suffix += f"_{os.environ['RUN_TAG']}"
    out_file = get_output_path(model_slug, mode_suffix)
    existing_data, completed_ids = load_existing_results(out_file)

    output_data = existing_data or {
        "model": model_slug,
        "backend": backend,
        "reasoning_mode": reasoning_method,
        "think_tool": think_tool,
        "f9_baseline": f9_baseline,
        "samples_per_config": samples_per_config,
        "n_tasks": len(tasks),
        "n_configs": len(configs),
        "config_list": [(name, c) for name, c in configs],
        "timestamp": now_pacific().strftime("%Y%m%d_%H%M%S"),
        "samples": [],
    }

    # harmbench_verbatim runs on the safety arm only, so it contributes one
    # valence per task set rather than two.
    n_verbatim_tasks = (
        sum(1 for t in tasks if t["valence"] == "safety")
        if any(name == HARMBENCH_VERBATIM for name, _ in configs)
        else 0
    )
    n_grid_configs = sum(1 for name, _ in configs if name != HARMBENCH_VERBATIM)
    total_calls = (len(tasks) * n_grid_configs + n_verbatim_tasks) * samples_per_config
    print(f"\nTasks loaded: {len(tasks)} ({n_task_sets} task sets x 2 valences)")
    print(f"Configs: {len(configs)} ({', '.join(name for name, _ in configs)})")
    print(f"Samples per config: {samples_per_config}")
    print(f"Model: {model_slug} (backend: {backend})")
    print(f"Total API calls needed: {total_calls}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Output file: {out_file}")
    print("=" * 70)

    def build_result(
        response: str, meta: dict[str, Any], sample_idx: int
    ) -> dict[str, Any]:
        return {
            "id": meta["id"],
            "task_id": meta["task_id"],
            "task_name": meta["task_name"],
            "valence": meta["valence"],
            "config_name": meta["config_name"],
            "factor_config": meta["factor_config"],
            "chat": meta["chat"],
            "sample_idx": sample_idx,
            "response": response,
        }

    start_time = now_pacific()
    new_count = 0

    # Build EVERY pending item up front, then dispatch in chunks. This loop used
    # to run one task at a time, which silently capped in-flight requests at that
    # task's item count regardless of MAX_CONCURRENT: a full run has 54 items per
    # task (2 valences x 9 configs x 3 samples) so the cap was invisible, but a
    # filtered run (e.g. CONFIG_FILTER=all_varied,harmbench_verbatim at 1 sample)
    # has 3 items per task, so only 3 requests ran at once and a slow provider
    # turned a 4-minute job into 70+ minutes.
    items = []
    cached = 0
    for task in tasks:
        template = task["template"]
        for config_name, factor_config in configs:
            if config_name == HARMBENCH_VERBATIM:
                if task["valence"] != "safety":
                    continue
                row = harmbench_rows[task["task_name"]]
                chat = harmbench_verbatim_chat(row)
                factor_config = harmbench_factor_config(row)
            else:
                chat = render(template, factor_config, model_slug)
            for s_idx in range(1, samples_per_config + 1):
                sample_id = f"{task['task_id']}_{config_name}_sample{s_idx}"
                if sample_id in completed_ids:
                    cached += 1
                    continue
                items.append(
                    (
                        client,
                        chat,
                        {
                            "id": sample_id,
                            "task_id": task["task_id"],
                            "task_name": task["task_name"],
                            "valence": task["valence"],
                            "config_name": config_name,
                            "factor_config": factor_config,
                            "chat": chat,
                            "sample_idx": s_idx,
                        },
                    )
                )

    print(f"Pending: {len(items)} calls ({cached} cached)")

    # Chunked so a crash loses at most one chunk, but each chunk is several times
    # max_concurrent so the semaphore stays saturated as fast calls retire and
    # we're not left idling on one straggler before the next save.
    chunk_size = max(cfg["MAX_CONCURRENT"] * 4, 100)
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        print(
            f"[chunk {start // chunk_size + 1}/{-(-len(items) // chunk_size)}] "
            f"{len(chunk)} calls, up to {cfg['MAX_CONCURRENT']} in flight"
        )
        batch_results = asyncio.run(
            query_parallel(chunk, build_result, cfg["MAX_CONCURRENT"])
        )

        output_data["samples"].extend(batch_results)
        completed_ids.update(r["id"] for r in batch_results)
        new_count += len(batch_results)

        save_results(out_file, output_data)
        print(f"  -> saved ({len(output_data['samples'])} total samples)")

    duration = now_pacific() - start_time
    print("=" * 70)
    print(f"Done. {new_count} new responses in {duration}")
    print(f"Total samples in file: {len(output_data['samples'])}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
