#!/usr/bin/env zsh
# CoT-judge Phase 1 response files (harness, prompt-only control, OpenRouter
# baseline). Sequential: the judge is claude-sonnet-5 on one Anthropic key and
# 429s above ~50 concurrent. Output cot files are named cot_<model>_<runtag>_<ts>
# so the three arms don't collide with each other or with the main sweep's
# cot_<model>_<ts> files.
#   source .envrc; scripts/openclaw_phase1_judge.sh [file ...]
# With no args: every finished *_openclaw_p1_*, *_p1ctl_*, *_p1base_* response
# file. Pass explicit files to judge a subset (e.g. the controls while the
# harness lanes are still writing their files).
set -u
cd "$(dirname "$0")/.."
TAG=${TAG:-p1}
LOG=results/eval_bench/openclaw/logs
mkdir -p $LOG
if [ $# -gt 0 ]; then files=("$@"); else
  files=(results/eval_bench/response/responses_*_openclaw_${TAG}_*.json(N)
         results/eval_bench/response/responses_*_${TAG}ctl_*.json(N)
         results/eval_bench/response/responses_*_${TAG}base_*.json(N))
fi
for f in $files; do
  base=${f##*/responses_}; base=${base%_2026*}          # e.g. qwen3.8-2.4t_p1ctl
  if ls results/eval_bench/cot/cot_${base}_*.json >/dev/null 2>&1; then
    echo "judged already: $base"; continue
  fi
  echo "judging $f -> cot_${base}_<ts>.json"
  RESPONSE_FILE=$f MODEL_KEY=$base MAX_CONCURRENT=${JUDGE_CONCURRENCY:-40} \
    .venv/bin/python eval/eval_eval_bench_cot.py > $LOG/cot_${base}.log 2>&1
  tail -12 $LOG/cot_${base}.log | grep -E "Overall awareness|N/A\)|Results saved"
done
