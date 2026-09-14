#!/bin/bash
# Self-report probe over every F10 response file. Usage: run_selfreport_f10.sh
cd /Users/robbie/repos/decomposing-eval-awareness-14
_MC=$MAX_CONCURRENT
set -a; source /Users/robbie/repos/decomposing-eval-awareness-14/.envrc; set +a
export MAX_CONCURRENT=${_MC:-500}
export OPENAI_API_KEY="${OPENAI_API_KEY_PERSONAL:-$OPENAI_API_KEY}"
LOG=results_f10/logs; mkdir -p $LOG
lane() {
  for f in "$@"; do
    m=$(basename "$f" .json)
    echo "[$(date +%H:%M:%S)] SR START $m"
    RESPONSE_FILE="$f" .venv/bin/python eval/eval_eval_bench_selfreport.py > "$LOG/sr_$m.log" 2>&1
    echo "[$(date +%H:%M:%S)] SR DONE  $m rc=$?"
  done
}
R=results_f10/eval_bench/response
lane $R/*claude*.json &
lane $R/*gemini*.json &
lane $R/*gpt-5*.json &
lane $R/*qwen*.json $R/*deepseek*.json $R/*kimi*.json $R/*glm*.json $R/*minimax*.json &
wait
echo "ALL SELFREPORT COMPLETE"
