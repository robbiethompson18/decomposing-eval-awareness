"""
Statistics computation and display functions.
"""

from typing import Any

# Default metrics for HarmBench/ToyBench
METRICS = ["model_awareness", "hypothetical_recognition", "task_performance"]
METRIC_LABELS = ["Model Awareness", "Hypothetical Recognition", "Task Performance"]

# Metrics for agentic misalignment (uses harmful_behavior instead of task_performance)
AGENTICMISALIGNMENT_METRICS = [
    "model_awareness",
    "hypothetical_recognition",
    "harmful_behavior",
]
AGENTICMISALIGNMENT_METRIC_LABELS = [
    "Model Awareness",
    "Hypothetical Recognition",
    "Harmful Behavior",
]


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
    """Bucket a judge conclusion as "yes", "no", or "invalid".

    "invalid" covers missing/malformed judge output and any conclusion that is
    not a clean yes/no (e.g. "partially", empty string from a parse failure).
    Keeping it separate from "no" prevents judge failures from silently
    deflating the reported rates.
    """
    return concl if concl in ("yes", "no") else "invalid"


def extract_awareness_type(judge_obj: dict[str, Any]) -> str:
    """Extract awareness_type from model_awareness section.

    Returns:
        "safety", "capability", or "n/a" (normalized to lowercase)
    """
    if not isinstance(judge_obj, dict):
        return "n/a"
    awareness = judge_obj.get("model_awareness")
    if isinstance(awareness, dict):
        awareness_type = awareness.get("awareness_type")
        if isinstance(awareness_type, str):
            return awareness_type.strip().lower()
    return "n/a"


def compute_metrics(
    records: list[dict[str, Any]],
    metrics: list[str] = METRICS,
) -> dict[str, dict[str, Any]]:
    """Compute yes/no/invalid counts and percentages for each metric.

    "invalid" counts judge outputs that could not be parsed as a clean yes/no
    (see bucket_conclusion). Also tracks awareness_type (safety/capability)
    for model_awareness.
    """
    totals = {m: {"yes": 0, "no": 0, "invalid": 0, "total": 0} for m in metrics}

    # Track awareness_type for model_awareness metric
    awareness_type_counts = {"safety": 0, "capability": 0, "n/a": 0}

    for rec in records:
        judge = rec.get("judge") or {}
        for metric in metrics:
            bucket = bucket_conclusion(extract_conclusion(judge, metric))
            totals[metric]["total"] += 1
            totals[metric][bucket] += 1

        # Track awareness_type
        awareness_type = extract_awareness_type(judge)
        if awareness_type in awareness_type_counts:
            awareness_type_counts[awareness_type] += 1
        else:
            awareness_type_counts["n/a"] += 1

    # Compute percentages
    for metric in metrics:
        total = totals[metric]["total"]
        for bucket in ["yes", "no", "invalid"]:
            totals[metric][f"pct_{bucket}"] = (
                (totals[metric][bucket] / total * 100.0) if total > 0 else 0.0
            )

    # Add awareness_type stats to model_awareness metric
    if "model_awareness" in totals:
        total = totals["model_awareness"]["total"]
        totals["model_awareness"]["awareness_type"] = awareness_type_counts
        for atype in ["safety", "capability"]:
            totals["model_awareness"][f"pct_{atype}"] = (
                (awareness_type_counts[atype] / total * 100.0) if total > 0 else 0.0
            )

    return totals


def print_overall_stats(
    tag: str,
    stats: dict[str, dict[str, Any]],
    metrics: list[str] = METRICS,
    metric_labels: list[str] = METRIC_LABELS,
) -> None:
    """Print overall statistics for a set of records.

    Shows Yes/No counts, plus Invalid (unparseable judge output) with a loud
    warning whenever any are present.
    For model_awareness, also shows awareness_type breakdown (safety/capability).
    """
    total = stats[metrics[0]]["total"]
    print(f"\n[{tag}] Total interactions: {total}")

    for metric, label in zip(metrics, metric_labels):
        s = stats[metric]
        line = (
            f"[{tag}] {label}: Yes {s['yes']} ({s['pct_yes']:.1f}%), "
            f"No {s['no']} ({s['pct_no']:.1f}%)"
        )
        invalid = s.get("invalid", 0)
        if invalid:
            line += f", INVALID {invalid} ({s.get('pct_invalid', 0):.1f}%)"
        print(line)
        if invalid:
            print(
                f"[{tag}] !!! WARNING: {invalid}/{s['total']} judge outputs for "
                f"'{label}' could not be parsed as Yes/No — counted as INVALID, "
                f"not No. Inspect the raw judge responses."
            )

        # For model_awareness, show awareness_type breakdown
        if metric == "model_awareness" and "awareness_type" in s:
            at = s["awareness_type"]
            print(
                f"[{tag}]   Awareness Type: Safety {at['safety']} ({s.get('pct_safety', 0):.1f}%), "
                f"Capability {at['capability']} ({s.get('pct_capability', 0):.1f}%)"
            )


def print_category_table(
    category_metrics: dict[str, dict[str, dict[str, Any]]],
    metrics: list[str] = METRICS,
    metric_labels: list[str] = METRIC_LABELS,
) -> None:
    """Print a formatted table of metrics by functional category (for HarmBench).

    Shows Yes/No counts; cells grow a third Invalid count (unparseable judge
    output) whenever any category has one, with a loud warning.
    """
    if not category_metrics:
        print("No categories to display.")
        return

    categories = list(category_metrics.keys())
    has_invalid = any(
        category_metrics[cat][m].get("invalid", 0)
        for cat in categories
        for m in metrics
    )

    # Column widths
    label_width = max(len(lbl) for lbl in metric_labels) + 2
    col_width = max(len(cat) for cat in categories) + 4

    # Header
    header = " ".ljust(label_width) + "".join(
        cat.ljust(col_width) for cat in categories
    )
    print(header)
    print("-" * len(header))

    # Rows with counts (Yes, No[, Invalid])
    if has_invalid:
        print("Cell format: (Yes,No,Invalid)")
    for metric, label in zip(metrics, metric_labels):
        cells = []
        for cat in categories:
            s = category_metrics[cat][metric]
            if has_invalid:
                cells.append(f"({s['yes']},{s['no']},{s.get('invalid', 0)})")
            else:
                cells.append(f"({s['yes']},{s['no']})")
        print(label.ljust(label_width) + "".join(c.ljust(col_width) for c in cells))

    # Percentages breakdown
    print("\nPercentages (Yes% / No%):")
    for cat in categories:
        print(f"  {cat}:")
        for metric, label in zip(metrics, metric_labels):
            s = category_metrics[cat][metric]
            line = f"    {label}: {s['pct_yes']:.1f}% / {s['pct_no']:.1f}%"
            if s.get("invalid", 0):
                line += f" (INVALID: {s.get('pct_invalid', 0):.1f}%)"
            print(line)

    if has_invalid:
        total_invalid = sum(
            category_metrics[cat][m].get("invalid", 0)
            for cat in categories
            for m in metrics
        )
        print(
            f"\n!!! WARNING: {total_invalid} judge outputs across categories could "
            f"not be parsed as Yes/No — counted as INVALID, not No. "
            f"Inspect the raw judge responses."
        )


def print_per_prompt_stats(
    tag: str,
    records: list[dict[str, Any]],
    metrics: list[str] = METRICS,
    metric_labels: list[str] = METRIC_LABELS,
) -> None:
    """Print per-prompt statistics table (for ToyBench).

    Shows Yes/No counts; cells grow a third Invalid count (unparseable judge
    output) whenever any prompt has one, with a loud warning.
    """
    if not records:
        print(f"\n[{tag}] No records.")
        return

    # Group by prompt text
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = rec.get("prompt") or ""
        groups.setdefault(key, []).append(rec)

    print(f"\n[{tag}] Per-prompt statistics ({len(groups)} prompts):")

    # Compute per-prompt counts
    prompt_data = []
    for idx, (prompt_text, recs) in enumerate(groups.items(), 1):
        counts = {m: {"yes": 0, "no": 0, "invalid": 0} for m in metrics}
        for r in recs:
            judge = r.get("judge") or {}
            for metric in metrics:
                bucket = bucket_conclusion(extract_conclusion(judge, metric))
                counts[metric][bucket] += 1

        # Compute awareness percentage for this prompt
        total_samples = len(recs)
        awareness_yes = counts["model_awareness"]["yes"]
        awareness_pct = (
            (awareness_yes / total_samples * 100.0) if total_samples > 0 else 0.0
        )

        prompt_data.append(
            {
                "idx": idx,
                "short": prompt_text[:80] + "..."
                if len(prompt_text) > 80
                else prompt_text,
                "counts": counts,
                "total_samples": total_samples,
                "awareness_pct": awareness_pct,
            }
        )

    # Print table
    headers = [f"Prompt {d['idx']}" for d in prompt_data]
    col_width = max(len(h) for h in headers) + 2
    label_width = max(len(lbl) for lbl in metric_labels) + 2

    header_row = " ".ljust(label_width) + "".join(h.ljust(col_width) for h in headers)
    print(header_row)
    print("-" * len(header_row))

    has_invalid = any(d["counts"][m]["invalid"] for d in prompt_data for m in metrics)
    if has_invalid:
        print(f"[{tag}] Cell format: (Yes,No,Invalid)")
    for metric, label in zip(metrics, metric_labels):
        if has_invalid:
            cells = [
                f"({d['counts'][metric]['yes']},{d['counts'][metric]['no']},{d['counts'][metric]['invalid']})"
                for d in prompt_data
            ]
        else:
            cells = [
                f"({d['counts'][metric]['yes']},{d['counts'][metric]['no']})"
                for d in prompt_data
            ]
        print(label.ljust(label_width) + "".join(c.ljust(col_width) for c in cells))

    if has_invalid:
        total_invalid = sum(
            d["counts"][m]["invalid"] for d in prompt_data for m in metrics
        )
        print(
            f"\n[{tag}] !!! WARNING: {total_invalid} judge outputs could not be "
            f"parsed as Yes/No — counted as INVALID, not No. "
            f"Inspect the raw judge responses."
        )

    # Per-prompt awareness percentages
    print(
        f"\n[{tag}] Per-prompt evaluation awareness (% of samples with awareness=yes):"
    )
    for d in prompt_data:
        print(
            f"  Prompt {d['idx']}: {d['awareness_pct']:.1f}% ({d['counts']['model_awareness']['yes']}/{d['total_samples']} samples)"
        )

    # Summary statistics
    awareness_pcts = [d["awareness_pct"] for d in prompt_data]
    avg_awareness = sum(awareness_pcts) / len(awareness_pcts) if awareness_pcts else 0.0
    prompts_with_any_awareness = sum(
        1 for d in prompt_data if d["counts"]["model_awareness"]["yes"] > 0
    )

    print(f"\n[{tag}] Awareness Summary:")
    print(f"  Average awareness across prompts: {avg_awareness:.1f}%")
    print(
        f"  Prompts with any awareness (>=1 yes): {prompts_with_any_awareness}/{len(prompt_data)} ({prompts_with_any_awareness / len(prompt_data) * 100:.1f}%)"
    )

    # Prompt index reference
    print(f"\n[{tag}] Prompt index reference:")
    for d in prompt_data:
        print(f"  {d['idx']}: {d['short']}")
