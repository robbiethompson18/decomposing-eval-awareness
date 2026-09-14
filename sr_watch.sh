#!/bin/bash
# Wait for a provider lane's generation to finish, then self-report its models.
# Usage: sr_watch.sh <backend> <n_models> <file-glob-pattern...>
cd /Users/robbie/repos/decomposing-eval-awareness-14
backend=$1; n=$2; shift 2
while [ "$(grep -c "DONE  $backend " results_f10/orchestrator.log)" -lt "$n" ]; do sleep 10; done
echo "[$(date +%H:%M:%S)] lane $backend generation complete; starting self-report"
set -a; source /Users/robbie/repos/decomposing-eval-awareness-14/.envrc; set +a
export MAX_CONCURRENT=500  # .envrc sets 100; this run wants 500
export OPENAI_API_KEY="${OPENAI_API_KEY_PERSONAL:-$OPENAI_API_KEY}"
for pat in "$@"; do
  for f in results_f10/eval_bench/response/$pat; do
    [ -e "$f" ] || continue
    m=$(basename "$f" .json)
    echo "[$(date +%H:%M:%S)] SR START $m"
    RESPONSE_FILE="$f" .venv/bin/python -u eval/eval_eval_bench_selfreport.py > "results_f10/logs/sr_$m.log" 2>&1
    echo "[$(date +%H:%M:%S)] SR DONE  $m rc=$?"
  done
done
echo "[$(date +%H:%M:%S)] lane $backend SELFREPORT COMPLETE"
