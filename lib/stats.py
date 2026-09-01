"""Small statistics helpers (leaf module — imports nothing from this repo)."""

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At k=0 / k=n float error puts the bound a hair past p (e.g. lo=8.7e-19
    # for p=0), which matplotlib's errorbar rejects as a negative half-width.
    return (max(0.0, min(p, center - half)), min(1.0, max(p, center + half)))
