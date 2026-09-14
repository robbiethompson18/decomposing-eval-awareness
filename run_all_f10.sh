#!/bin/bash
# F10 test run: baseline (F9=deployment, F10=current) vs F10 (F10=stale)
# 14 models, all 200 tasks, 1 sample per (task, config) = 400 calls/model.
cd /Users/robbie/repos/decomposing-eval-awareness-14
LOG=results_f10/logs; mkdir -p $LOG
lane() {  # lane <backend> <model...>
  local backend=$1; shift
  for m in "$@"; do
    echo "[$(date +%H:%M:%S)] START $backend $m"
    SAMPLES_PER_CONFIG=1 MAX_CONCURRENT=500 ./run_f10.sh "$backend" "$m" > "$LOG/$m.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  $backend $m rc=$?"
  done
}
lane anthropic_api claude-sonnet-4.5 claude-haiku-4.5 &
lane gemini_api    gemini-3.1-pro gemini-3.5-flash &
lane openai_api    gpt-5 gpt-5.1 gpt-5.2 gpt-5.4 gpt-5.6-sol &
lane openrouter    qwen3.8-2.4t deepseek-v4-pro kimi-k3 glm-5.3 minimax-m3 &
wait
echo "ALL LANES COMPLETE"
