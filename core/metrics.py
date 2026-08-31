"""
Judge-conclusion parsing helpers.

(Upstream's aggregate-stats functions — compute_metrics, the print_* tables,
and the per-benchmark METRICS constants — served the deleted harmbench/
agentharm/agentic-misalignment probe scripts and were removed with them.)
"""

from typing import Any


def extract_conclusion(judge_obj: dict[str, Any], metric_key: str) -> str:
    """Extract yes/no conclusion from judge response for a given metric."""
    if not isinstance(judge_obj, dict):
        return ""
    section = judge_obj.get(metric_key)
    if isinstance(section, dict):
        val = section.get("conclusion")
        if isinstance(val, str):
            return val.strip().lower()
    return ""


def bucket_conclusion(concl: str) -> str:
    """Bucket a judge conclusion as "yes", "no", "n/a", or "invalid".

    "n/a" is a deliberate judgment that the metric could not be graded (e.g.
    mark_no_reasoning_sections sets it when the response has no CoT for the
    reasoning-based metrics). "invalid" covers missing/malformed judge output
    and any other conclusion that is not a clean yes/no (e.g. "partially",
    empty string from a parse failure). Keeping both separate from "no"
    prevents ungradeable samples from silently deflating the reported rates.
    """
    return concl if concl in ("yes", "no", "n/a") else "invalid"
