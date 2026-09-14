#!/bin/bash
# One model of the F10 test run. Usage: run_f10.sh <backend> <model_key>
_MC=$MAX_CONCURRENT; _SPC=$SAMPLES_PER_CONFIG
set -a; source /Users/robbie/repos/decomposing-eval-awareness-14/.envrc; set +a
MAX_CONCURRENT=${_MC:-$MAX_CONCURRENT}; SAMPLES_PER_CONFIG=${_SPC:-$SAMPLES_PER_CONFIG}
cd /Users/robbie/repos/decomposing-eval-awareness-14
export SAMPLES_PER_CONFIG=${SAMPLES_PER_CONFIG:-1}
export MAX_CONCURRENT=${MAX_CONCURRENT:-500}
export CONFIG_FILTER=baseline,F10
export TARGET_BACKEND=$1
case "$1" in
  anthropic_api) export ANTHROPIC_MODEL_KEY=$2;;
  openai_api)    export OPENAI_MODEL_KEY=$2; export OPENAI_API_KEY="${OPENAI_API_KEY_PERSONAL:-$OPENAI_API_KEY}";;
  gemini_api)    export GEMINI_MODEL_KEY=$2;;
  openrouter)    export OPENROUTER_MODEL_KEY=$2;;
esac
exec .venv/bin/python run/run_eval_bench.py
