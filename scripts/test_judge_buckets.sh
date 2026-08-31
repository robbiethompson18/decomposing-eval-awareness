#!/bin/bash
# Test the judge-output bucketing: yes/no/N-A/invalid counting and the loud
# warnings in core/metrics.py, core/evaluator.py, and the eval_bench printers.
# Run from the repo root: bash scripts/test_judge_buckets.sh
# Uses synthetic records; if local judged results exist under
# results/eval_bench/cot/, also exercises the printers on those.
set -e

.venv/bin/python - <<'EOF'
import copy
import glob
import importlib.util
import json
import sys

sys.path.insert(0, ".")
from core.evaluator import mark_no_reasoning_sections
from core.metrics import (
    METRIC_LABELS,
    METRICS,
    compute_metrics,
    print_category_table,
    print_overall_stats,
)

def rec(aware, hypo, perf):
    def sec(v):
        return {"conclusion": v} if v is not None else "GARBAGE"
    return {"prompt": "p", "judge": {
        "model_awareness": sec(aware),
        "hypothetical_recognition": sec(hypo),
        "task_performance": sec(perf),
    }}

print("=" * 70)
print("1. CLEAN synthetic records -> no notes, no warnings, 2-tuple cells")
print("=" * 70)
clean = [rec("Yes", "No", "Yes"), rec("No", "No", "No")]
s = compute_metrics(clean, METRICS)
assert s["model_awareness"]["n/a"] == 0 and s["model_awareness"]["invalid"] == 0
print_overall_stats("clean", s, METRICS, METRIC_LABELS)
print_category_table({"catA": s}, METRICS, METRIC_LABELS)

print()
print("=" * 70)
print("2. GUARD: empty reasoning -> awareness/hypothetical become N/A")
print("=" * 70)
j = rec("No", "No", "Yes")["judge"]
guarded = mark_no_reasoning_sections(copy.deepcopy(j), "")
assert guarded["model_awareness"]["conclusion"] == "N/A"
assert guarded["hypothetical_recognition"]["conclusion"] == "N/A"
assert guarded["task_performance"]["conclusion"] == "Yes"  # answer-based, kept
assert mark_no_reasoning_sections(copy.deepcopy(j), "thought hard") == j
print("guard assertions pass")

print()
print("=" * 70)
print("3. MIXED records -> N/A NOTE + INVALID WARNING, 4-tuple cells")
print("=" * 70)
mixed = clean + [
    {"prompt": "p", "judge": mark_no_reasoning_sections(rec("No", "No", "Yes")["judge"], "")},
    rec(None, "Partially", "No"),          # malformed -> invalid
    {"prompt": "p", "judge": None},        # judge missing entirely -> invalid
]
s2 = compute_metrics(mixed, METRICS)
assert s2["model_awareness"]["n/a"] == 1
assert s2["model_awareness"]["invalid"] == 2
assert s2["task_performance"]["n/a"] == 0
print_overall_stats("mixed", s2, METRICS, METRIC_LABELS)
print_category_table({"catA": s2}, METRICS, METRIC_LABELS)

files = sorted(glob.glob("results/eval_bench/cot/cot_*.json"))
if files:
    print()
    print("=" * 70)
    print("4. REAL local files through the eval_bench printer")
    print("   (a pre-guard file with CoT-less responses triggers the reliability warning)")
    print("=" * 70)
    spec = importlib.util.spec_from_file_location("ec", "eval/eval_eval_bench_cot.py")
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)
    for path in files:
        print(f"\n--- {path} ---")
        ec.print_stats(json.load(open(path)))
else:
    print("\n(no local results/eval_bench/cot files; skipped real-data section)")

print("\nALL ASSERTIONS PASS")
EOF
