"""Bin trust weights -- empirical trust from realized P&L by probability bin.

Answers: "in each probability range (e.g. 20-25%, 45-50%), does our edge
actually materialize into realized returns?" The resulting per-bin trust
multiplier scales stakes up where the edge has historically been real and
down where it hasn't -- replacing hand-tuned heuristics (like a fixed
longshot penalty) with measured evidence.
"""

from typing import List

__all__ = [
    "compute_bin_trust_weights",
    "lookup_trust",
]


def compute_bin_trust_weights(
    bins: List[dict],
    *,
    min_samples: int = 10,
    default_trust: float = 1.0,
    min_trust: float = 0.1,
    max_trust: float = 1.5,
    smooth_neighbors: int = 1,
) -> List[dict]:
    """Compute per-bin trust weights from historical P&L bins.

    Three-step pipeline:

    1. **Raw trust** -- direct mapping from realized ROI:
       ``raw_trust = 1.0 + roi/100``, clamped to ``[min_trust, max_trust]``.
       (+50% ROI -> 1.5, break-even -> 1.0, -50% -> 0.5.)
    2. **Sample-size shrinkage** -- Bayesian-style blend toward
       ``default_trust`` when a bin has few positions:
       ``effective = (n / (n + min_samples)) * raw + (min_samples / (n + min_samples)) * default``.
    3. **Neighbor smoothing** -- weighted average with adjacent bins
       (weight ``0.5**distance``) to damp bin-boundary noise.

    Args:
        bins: List of dicts with ``low``, ``high`` (probability bounds on
            the 0-100 scale), ``count`` (number of settled positions), and
            ``roi`` (realized ROI percentage, e.g. ``15.0`` = +15%).
        min_samples: Below this count, trust shrinks heavily toward default.
        default_trust: Neutral trust (1.0 = no adjustment).
        min_trust: Floor clamp for final trust.
        max_trust: Ceiling clamp for final trust.
        smooth_neighbors: Number of adjacent bins to average with (0 = none).

    Returns:
        List of dicts with keys ``prob_low``, ``prob_high``, ``trust``,
        ``raw_trust``, ``count``, ``roi``, ``shrunk``.
    """
    if not bins:
        return []

    # Step 1: raw trust per bin
    raw_entries = []
    for b in bins:
        count = b.get("count", 0)
        roi = b.get("roi", 0)  # percentage, e.g. 15.0 means +15%

        if count == 0:
            raw_trust = default_trust
        else:
            raw_trust = max(min_trust, min(max_trust, 1.0 + roi / 100.0))

        raw_entries.append({
            "prob_low": b["low"],
            "prob_high": b["high"],
            "raw_trust": raw_trust,
            "count": count,
            "roi": roi,
        })

    # Step 2: sample-size shrinkage
    shrunk_entries = []
    for e in raw_entries:
        count = e["count"]
        weight = count / (count + min_samples) if (count + min_samples) > 0 else 0
        shrunk_trust = weight * e["raw_trust"] + (1.0 - weight) * default_trust
        shrunk_entries.append({**e, "shrunk": round(shrunk_trust, 4)})

    # Step 3: neighbor smoothing
    n = len(shrunk_entries)
    if smooth_neighbors > 0 and n > 1:
        smoothed = []
        for i in range(n):
            total_weight = 1.0
            weighted_sum = shrunk_entries[i]["shrunk"]

            for offset in range(1, smooth_neighbors + 1):
                neighbor_weight = 0.5 ** offset
                if i - offset >= 0:
                    weighted_sum += neighbor_weight * shrunk_entries[i - offset]["shrunk"]
                    total_weight += neighbor_weight
                if i + offset < n:
                    weighted_sum += neighbor_weight * shrunk_entries[i + offset]["shrunk"]
                    total_weight += neighbor_weight

            smoothed_trust = weighted_sum / total_weight
            smoothed_trust = max(min_trust, min(max_trust, smoothed_trust))
            smoothed.append({
                "prob_low": shrunk_entries[i]["prob_low"],
                "prob_high": shrunk_entries[i]["prob_high"],
                "trust": round(smoothed_trust, 4),
                "raw_trust": round(shrunk_entries[i]["raw_trust"], 4),
                "count": shrunk_entries[i]["count"],
                "roi": shrunk_entries[i]["roi"],
                "shrunk": shrunk_entries[i]["shrunk"],
            })
        return smoothed

    # No smoothing -- just clamp and return
    return [
        {
            "prob_low": e["prob_low"],
            "prob_high": e["prob_high"],
            "trust": max(min_trust, min(max_trust, e["shrunk"])),
            "raw_trust": round(e["raw_trust"], 4),
            "count": e["count"],
            "roi": e["roi"],
            "shrunk": e["shrunk"],
        }
        for e in shrunk_entries
    ]


def lookup_trust(trust_weights: List[dict], prob: float) -> float:
    """Look up the trust weight for a probability (0.0-1.0 scale).

    Finds the bin containing ``prob * 100`` and returns its trust value.
    The top bin is inclusive of its upper bound. Falls back to 1.0 when no
    bin matches or no weights are provided.
    """
    if not trust_weights:
        return 1.0

    prob_pct = prob * 100.0

    for tw in trust_weights:
        low = tw.get("prob_low", 0)
        high = tw.get("prob_high", 100)
        if low <= prob_pct < high or (high >= 100 and prob_pct >= low):
            return tw.get("trust", 1.0)

    return 1.0
