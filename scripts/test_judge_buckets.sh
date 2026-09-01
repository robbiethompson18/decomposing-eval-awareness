#!/bin/bash
# Test the judge-output bucketing: yes/no/N-A/invalid counting, the v2
# task-execution buckets (correct/guarded/incorrect/refused), and the loud
# warnings in core/metrics.py, core/evaluator.py, and the eval_bench printer.
# Run from the repo root: bash scripts/test_judge_buckets.sh
# Uses synthetic records; if local judged results exist under
# results/eval_bench/cot/, also exercises the printer on those.
set -e

.venv/bin/python - <<'EOF'
import contextlib
import copy
import glob
import importlib.util
import io
import json
import sys

sys.path.insert(0, ".")
from core.evaluator import mark_no_reasoning_sections
from core.metrics import bucket_conclusion, bucket_task_conclusion, extract_conclusion

spec = importlib.util.spec_from_file_location("ec", "eval/eval_eval_bench_cot.py")
ec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ec)

def rec(aware, hypo, perf, response="<think>thought hard</think>done"):
    def sec(v):
        return {"conclusion": v} if v is not None else "GARBAGE"
    return {"prompt": "p", "valence": "positive", "config_name": "baseline",
            "response": response, "judge": {
        "model_awareness": sec(aware),
        "hypothetical_recognition": sec(hypo),
        "task_performance": sec(perf),
    }}

def buckets(records, key="model_awareness"):
    fn = bucket_task_conclusion if key == "task_performance" else bucket_conclusion
    return [fn(extract_conclusion(r.get("judge") or {}, key)) for r in records]

def stats_output(records):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ec.print_stats(records)
    out = buf.getvalue()
    print(out)
    return out

print("=" * 70)
print("1. CLEAN synthetic records -> clean buckets, no notes, no warnings")
print("=" * 70)
clean = [rec("Yes", "No", "correct"), rec("No", "No", "refused")]
assert buckets(clean) == ["yes", "no"]
assert buckets(clean, "task_performance") == ["correct", "refused"]
out = stats_output(clean)
assert "NOTE" not in out and "WARNING" not in out
assert "awareness=1/2 (50.0%)" in out
assert "correct=1" in out and "refused=1" in out

print("=" * 70)
print("2. TASK buckets: v2 categories + legacy yes/no passthrough")
print("=" * 70)
for v in ("correct", "guarded", "incorrect", "refused", "yes", "no", "n/a"):
    assert bucket_task_conclusion(v) == v
assert bucket_task_conclusion("partially") == "invalid"
assert bucket_task_conclusion("") == "invalid"
print("task bucket assertions pass")

print()
print("=" * 70)
print("3. GUARD: empty reasoning -> awareness/hypothetical become N/A")
print("=" * 70)
j = rec("No", "No", "correct")["judge"]
guarded = mark_no_reasoning_sections(copy.deepcopy(j), "")
assert guarded["model_awareness"]["conclusion"] == "N/A"
assert guarded["hypothetical_recognition"]["conclusion"] == "N/A"
assert guarded["task_performance"]["conclusion"] == "correct"  # answer-based, kept
assert mark_no_reasoning_sections(copy.deepcopy(j), "thought hard") == j
print("guard assertions pass")

print()
print("=" * 70)
print("4. MIXED records -> N/A NOTE + pre-guard + INVALID WARNINGS")
print("=" * 70)
mixed = clean + [
    {**rec("No", "No", "correct", response="no CoT here"), "judge": guarded},
    rec(None, "Partially", "guarded"),     # malformed section -> invalid
    {"prompt": "p", "judge": None},        # judge missing entirely -> invalid
]
b = buckets(mixed)
assert b.count("n/a") == 1 and b.count("invalid") == 2
tb = buckets(mixed, "task_performance")
assert tb.count("correct") == 2 and tb.count("guarded") == 1
assert tb.count("refused") == 1 and tb.count("invalid") == 1
out = stats_output(mixed)
assert "1/5 samples not graded for awareness" in out
assert "2/5 responses contain no reasoning" in out
assert "2/5 judge outputs could not be parsed" in out

files = sorted(glob.glob("results/eval_bench/cot/cot_*.json"))
if files:
    print()
    print("=" * 70)
    print("5. REAL local files through the eval_bench printer")
    print("   (a pre-guard file with CoT-less responses triggers the reliability warning)")
    print("=" * 70)
    for path in files:
        data = json.load(open(path))
        if not isinstance(data, list):
            print(f"\n--- {path} --- (skipped: not a per-run record list)")
            continue
        print(f"\n--- {path} ---")
        ec.print_stats(data)
else:
    print("\n(no local results/eval_bench/cot files; skipped real-data section)")

print("\nALL ASSERTIONS PASS")
EOF
