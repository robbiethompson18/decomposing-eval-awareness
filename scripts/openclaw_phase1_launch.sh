#!/usr/bin/env zsh
# Phase 1 of the OpenClaw realism experiment (.claude/notes/openclaw-realism-scope.md):
# the same 200 baseline requests delivered three ways, all via OpenRouter.
#
#   lane A/B  full harness   run/run_eval_bench_openclaw.py, 7 models, 2 at a time
#             (each = ~10 concurrent OpenClaw processes + Docker sandboxes)
#   lane C    controls       run/run_eval_bench.py CONFIG_FILTER=openclaw_prompt for all 7
#                            (prompt-only) + CONFIG_FILTER=baseline for the two closed models
#                            (bare-API baseline on the same provider route), then the
#                            self-report probe on each of those response files.
#
# Run from the repo root with direnv loaded:  nohup scripts/openclaw_phase1_launch.sh &
# Logs: results/eval_bench/openclaw/logs/<lane>_<model>.log
set -u
cd "$(dirname "$0")/.."
LOG=results/eval_bench/openclaw/logs
mkdir -p $LOG
PY=.venv/bin/python
TAG=${TAG:-p1}

harness() {  # model offset
  MODEL_KEY=$1 WORKER_OFFSET=$2 MAX_CONCURRENT=${HARNESS_CONCURRENCY:-10} RUN_TAG=$TAG \
    $PY run/run_eval_bench_openclaw.py > $LOG/harness_$1.log 2>&1
}

control() {  # model config tag
  TARGET_BACKEND=openrouter OPENROUTER_MODEL_KEY=$1 CONFIG_FILTER=$2 SAMPLES_PER_CONFIG=1 \
    RUN_TAG=$3 MAX_CONCURRENT=${CTL_CONCURRENCY:-30} \
    $PY run/run_eval_bench.py > $LOG/${3}_$1.log 2>&1
  local f
  f=$(ls -t results/eval_bench/response/responses_$1_$3_*.json | head -1)
  RESPONSE_FILE=$f MAX_CONCURRENT=20 $PY eval/eval_eval_bench_selfreport.py > $LOG/${3}_selfreport_$1.log 2>&1
}

( for m in qwen3.8-2.4t deepseek-v4-pro kimi-k3 glm-5.3; do harness $m 0; done ) &
( for m in minimax-m3 claude-sonnet-4.5 gemini-3.5-flash; do harness $m 10; done ) &
(
  for m in qwen3.8-2.4t deepseek-v4-pro kimi-k3 glm-5.3 minimax-m3 claude-sonnet-4.5 gemini-3.5-flash; do
    control $m openclaw_prompt ${TAG}ctl
  done
  for m in claude-sonnet-4.5 gemini-3.5-flash; do
    control $m baseline ${TAG}base
  done
) &
wait
echo "phase 1 launch complete $(date)"
