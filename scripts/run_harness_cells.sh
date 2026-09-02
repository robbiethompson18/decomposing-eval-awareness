#!/bin/bash
# One lane of the harness-trappings experiment for one model: generate the six
# harness cells (bare_harness, bare+F9/F11/F12/F13, baseline), then the
# self-report probe, then the CoT judge. Run from the repo root, one lane per
# model in parallel:
#
#   scripts/run_harness_cells.sh anthropic_api claude-sonnet-4.5
#   scripts/run_harness_cells.sh openrouter kimi-k3
#
# Env: SAMPLES_PER_CONFIG (1), MAX_CONCURRENT (200, target + self-report),
# JUDGE_CONCURRENT (60: keep the sum over lanes under ~500 on the Anthropic
# key, see CLAUDE.md), RUN_TAG (harness), VALENCE_FILTER, TASK_FILTER.
set -euo pipefail

backend=$1
model=$2
case "$backend" in
  anthropic_api) keyvar=ANTHROPIC_MODEL_KEY ;;
  gemini_api) keyvar=GEMINI_MODEL_KEY ;;
  openrouter) keyvar=OPENROUTER_MODEL_KEY ;;
  openai_api) keyvar=OPENAI_MODEL_KEY ;;
  *) echo "unknown backend $backend"; exit 1 ;;
esac

export TARGET_BACKEND=$backend
export "$keyvar=$model"
export CONFIG_FILTER=bare_harness,bare+F9,bare+F11,bare+F12,bare+F13,baseline
export SAMPLES_PER_CONFIG=${SAMPLES_PER_CONFIG:-1}
export MAX_CONCURRENT=${MAX_CONCURRENT:-200}
export RUN_TAG=${RUN_TAG:-harness}

echo "=== [$model] generate"
.venv/bin/python run/run_eval_bench.py

file=$(ls -t results/eval_bench/response/responses_"$model"*_"$RUN_TAG"_*.json | head -1)
echo "=== [$model] response file: $file"
echo "=== [$model] ERROR rows: $(grep -c '"response": "ERROR calling' "$file" || true)"

echo "=== [$model] self-report"
RESPONSE_FILE=$file .venv/bin/python eval/eval_eval_bench_selfreport.py

echo "=== [$model] CoT judge"
RESPONSE_FILE=$file MAX_CONCURRENT=${JUDGE_CONCURRENT:-60} .venv/bin/python eval/eval_eval_bench_cot.py
echo "=== [$model] done"
