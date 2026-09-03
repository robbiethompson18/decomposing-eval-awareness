"""
Run eval_bench tasks across all models.

For each task (safety + capability variant), runs:
  - 1 baseline config (all factors at naturalistic level)
  - 13 single-factor ablations (one factor flipped, rest at baseline)
  - 7 anchor configs off the OFAT grid:
      bare_harness        F1-F8 realistic, every harness feature off (F9 bare,
                          F11 cold, F12 none, F13 clean): a plain benchmark harness
      bare+F9/F11/F12/F13 bare_harness with exactly one harness feature switched back on
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

  Arm A of the think-tool-confound test (no harness, self-report only):
    THINK_TOOL=0 CONFIG_FILTER=baseline TARGET_BACKEND=... python run/run_eval_bench.py
  Its output file is suffixed _no_think_tool so it never resumes into an
  arm-B (think-tool) file. See .claude/notes/think-tool-harness-vs-selfreport.md.

Factor-stacking design (replicates the paper's Figure 7, arXiv:2605.23055):
  STACKING=1            Replace the OFAT grid with per-task random factor combinations:
                        STACKING_TASKS task sets (default 25) are sampled once (seeded), and
                        for each k = 0..8 factors varied, each task gets STACKING_COMBOS
                        (default 10) random subsets of F1–F8 — or every subset when there
                        are fewer than that (1 at k=0, 8 at k=1, 8 at k=7, 1 at k=8), i.e.
                        the paper's exact design. F9–F13 are pinned to the levels the paper
                        ran under (bare system prompt, cold, no tools, clean); see
                        PAPER_BASE_CONFIG. Output file gets a `_stacking` suffix.
  STACKING_TASKS        Task sets to sample (default: 25)
  STACKING_COMBOS       Random combos per task per k, where the space allows (default: 10)
  STACKING_SEED         Seed for task + combo sampling (default: 0; same across models)

    STACKING=1 TARGET_BACKEND=openrouter OPENROUTER_MODEL_KEY=qwen3-30b SAMPLES_PER_CONFIG=1 \\
      python run/run_eval_bench.py
"""

import asyncio
import glob
import json
import os
import random
import re
import sys
from itertools import combinations
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
from eval_bench.render import render
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
    # F13 (typos). Baseline = two hand-authored typing slips in the user
    # message, as real messages have; the ablation is the copy-edited text
    # every benchmark item has. Typos are {TYPO_*} slots in every shell, so
    # they compose with F1–F8 (eval_bench/typos.py).
    "F13": "typos",
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
    "F13": "clean",
}


ALL_VARIED = "all_varied"

# The paper's eight factors. F9–F13 are this repo's additions and are held at the
# level the paper implicitly ran under (its harness had no product system prompt,
# no prior turn, no tools, no typos) whenever we replicate a paper figure.
PAPER_FACTORS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
PAPER_BASE_CONFIG = {
    **{f: BASELINE_CONFIG[f] for f in PAPER_FACTORS},
    "F9": "bare",
    "F10": "current",
    "F11": "cold",
    "F12": "none",
    "F13": "clean",
}

STACKING_PREFIX = "S"  # config names: S{k}_{F1..F8 bitmask}, e.g. S3_10100100


def n_varied(factor_config: dict[str, str]) -> int:
    """How many of the paper's eight factors sit at their varied level."""
    return sum(
        1 for f in PAPER_FACTORS if factor_config.get(f) == FACTOR_VARIED_LEVELS[f]
    )


def build_stacking_configs(
    task_names: list[str], n_tasks: int, n_combos: int, seed: int
) -> dict[str, list[tuple[str, dict[str, str]]]]:
    """Per-task factor-stacking configs, the paper's Figure 7 design.

    Samples `n_tasks` task sets, then for every k in 0..8 gives each task
    `n_combos` random k-subsets of F1–F8 (every subset when C(8,k) <= n_combos).
    Everything is seeded so every model sees the same tasks and the same
    combinations. Combos are drawn per task, as in the paper, and are shared
    across a task's two valences so safety/capability stay paired.
    """
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(task_names), n_tasks))
    per_task: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for task_name in chosen:
        task_rng = random.Random(f"{seed}:{task_name}")
        configs = []
        for k in range(len(PAPER_FACTORS) + 1):
            subsets = list(combinations(PAPER_FACTORS, k))
            if len(subsets) > n_combos:
                subsets = task_rng.sample(subsets, n_combos)
            for subset in subsets:
                cfg = dict(PAPER_BASE_CONFIG)
                for f in subset:
                    cfg[f] = FACTOR_VARIED_LEVELS[f]
                mask = "".join("1" if f in subset else "0" for f in PAPER_FACTORS)
                configs.append((f"{STACKING_PREFIX}{k}_{mask}", cfg))
        per_task[task_name] = configs
    return per_task


# The run-level "harness features": everything the harness adds around the
# per-question content. `bare_harness` is F1-F8 at their realistic levels with
# all four switched off (bare system prompt, so no F10 timestamp either; task as
# turn 1; no search tool; copy-edited text) — the same tasks as sent from a
# plain benchmark harness. `bare+F<n>` switches exactly one harness feature back on,
# so bare_harness / bare+F* / baseline is a second OFAT grid that prices each
# harness feature from the bare side and the bundle against `baseline` (all on).
HARNESS_FACTORS = ("F9", "F11", "F12", "F13")
BARE_HARNESS = "bare_harness"


def harness_configs() -> list[tuple]:
    bare = dict(
        BASELINE_CONFIG, **{f: FACTOR_VARIED_LEVELS[f] for f in HARNESS_FACTORS}
    )
    configs = [(BARE_HARNESS, bare)]
    for factor in HARNESS_FACTORS:
        configs.append(
            (f"bare+{factor}", dict(bare, **{factor: BASELINE_CONFIG[factor]}))
        )
    return configs


def build_configs() -> list[tuple]:
    """Build 21 configs: baseline + 13 single-factor ablations + 7 anchors.

    The anchors are off the OFAT grid. `bare_harness` and the four `bare+F*`
    cells are the harness grid (see HARNESS_FACTORS); `all_varied` is the
    grid's far corner; `harmbench_verbatim` is not a render at all (its Chat
    and factor_config are built per task from the vendored HarmBench row,
    safety arm only), so its entry here carries the standard-behavior factor
    vector as a placeholder for config_list and is special-cased in the item
    loop.
    """
    configs = [("baseline", dict(BASELINE_CONFIG))]
    for factor, varied_level in FACTOR_VARIED_LEVELS.items():
        cfg = dict(BASELINE_CONFIG)
        cfg[factor] = varied_level
        configs.append((factor, cfg))
    configs.extend(harness_configs())
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
    # unsuffixed (native) run resume into a think_tool file, or an untagged
    # `_no_think_tool` run resume into a RUN_TAG'd `_no_think_tool_<tag>` file.
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
    configs = build_configs()
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

    # Factor-stacking design (paper Figure 7): configs are per task set, drawn
    # once (seeded) so every model runs the same cells. Task/valence filters
    # still apply on top; CONFIG_FILTER does not (the cells aren't named ahead).
    stacking = os.getenv("STACKING", "0") == "1"
    stacking_configs: dict[str, list[tuple[str, dict[str, str]]]] = {}
    if stacking:
        stacking_configs = build_stacking_configs(
            sorted({t["task_name"] for t in load_all_tasks()}),
            n_tasks=int(os.getenv("STACKING_TASKS", "25")),
            n_combos=int(os.getenv("STACKING_COMBOS", "10")),
            seed=int(os.getenv("STACKING_SEED", "0")),
        )
        tasks = [t for t in tasks if t["task_name"] in stacking_configs]
        seen: dict[str, dict[str, str]] = {}
        for task_configs in stacking_configs.values():
            seen.update(dict(task_configs))
        configs = sorted(seen.items())

    def configs_for(task_name: str) -> list[tuple[str, dict[str, str]]]:
        return stacking_configs[task_name] if stacking else configs

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
    if stacking:
        mode_suffix += "_stacking"
    if os.getenv("RUN_TAG"):
        mode_suffix += f"_{os.environ['RUN_TAG']}"
    out_file = get_output_path(model_slug, mode_suffix)
    existing_data, completed_ids = load_existing_results(out_file)

    output_data = existing_data or {
        "model": model_slug,
        "backend": backend,
        "reasoning_mode": reasoning_method,
        "think_tool": think_tool,
        "samples_per_config": samples_per_config,
        "n_tasks": len(tasks),
        "n_configs": len(configs),
        "config_list": [(name, c) for name, c in configs],
        "design": "stacking" if stacking else "ofat",
        "timestamp": now_pacific().strftime("%Y%m%d_%H%M%S"),
        "samples": [],
    }
    if stacking:
        output_data.setdefault(
            "stacking",
            {
                "tasks": sorted(stacking_configs),
                "combos_per_k": int(os.getenv("STACKING_COMBOS", "10")),
                "seed": int(os.getenv("STACKING_SEED", "0")),
            },
        )

    # harmbench_verbatim runs on the safety arm only, so it contributes one
    # valence per task set rather than two.
    n_verbatim_tasks = (
        sum(1 for t in tasks if t["valence"] == "safety")
        if any(name == HARMBENCH_VERBATIM for name, _ in configs)
        else 0
    )
    n_grid_cells = sum(
        1
        for t in tasks
        for name, _ in configs_for(t["task_name"])
        if name != HARMBENCH_VERBATIM
    )
    total_calls = (n_grid_cells + n_verbatim_tasks) * samples_per_config
    print(f"\nTasks loaded: {len(tasks)} ({n_task_sets} task sets x 2 valences)")
    if stacking:
        per_task = len(next(iter(stacking_configs.values())))
        print(f"Design: factor stacking, {per_task} configs per task set")
    else:
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
            # Count of F1–F8 at their varied level (plots/plot_stacking.py's x axis).
            "n_varied": n_varied(meta["factor_config"]),
            "chat": meta["chat"],
            "sample_idx": sample_idx,
            "response": response,
            # Serving provider (OpenRouter only; None on native backends) so a
            # routing change between cells or runs is visible in the data.
            "provider": meta.get("call_meta", {}).get("provider"),
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
        for config_name, factor_config in configs_for(task["task_name"]):
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

    # One dispatch for everything, saving as results stream in. This used to run
    # in chunks of 4x max_concurrent with a save after each: one hung request
    # (600 s timeout x 8 retries on the SDK clients) then held the whole chunk's
    # completed-but-unsaved calls hostage — a 2,000-call Sonnet chunk sat at
    # 1999/2000 for 25+ minutes. Now a straggler delays only itself, and a
    # crash or kill loses at most SAVE_EVERY completed calls.
    save_every = int(os.getenv("SAVE_EVERY", "100"))
    since_save = [0]

    def collect(result: dict[str, Any]) -> None:
        output_data["samples"].append(result)
        completed_ids.add(result["id"])
        since_save[0] += 1
        if since_save[0] >= save_every:
            since_save[0] = 0
            save_results(out_file, output_data)
            print(f"  -> saved ({len(output_data['samples'])} total samples)")

    print(f"Dispatching {len(items)} calls, up to {cfg['MAX_CONCURRENT']} in flight")
    if items:
        batch_results = asyncio.run(
            query_parallel(
                items, build_result, cfg["MAX_CONCURRENT"], on_result=collect
            )
        )
        new_count = len(batch_results)
        save_results(out_file, output_data)
        print(f"  -> saved ({len(output_data['samples'])} total samples)")

    duration = now_pacific() - start_time
    print("=" * 70)
    print(f"Done. {new_count} new responses in {duration}")
    print(f"Total samples in file: {len(output_data['samples'])}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
