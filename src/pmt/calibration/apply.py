"""Prediction-time application of fitted calibration curves.

Curves are the plain dicts produced by :mod:`pmt.calibration.curves` --
JSON-serializable, storable anywhere, and applied here without any
dependency on the fitting library.
"""

from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "apply_curve",
    "apply_calibration",
    "apply_multiclass_calibration",
]

# Calibrated outputs are clamped away from 0/1 so downstream consumers
# (log-loss, odds conversion, staking) never see degenerate probabilities.
_CLAMP_LO, _CLAMP_HI = 0.01, 0.99


def apply_curve(probs: np.ndarray, curve: Dict) -> np.ndarray:
    """Apply a sampled calibration curve to an array of probabilities.

    Linear interpolation across the curve's ``x``/``y`` sample points
    (vectorized). Inputs outside the sampled range clamp to the endpoints.

    Args:
        probs: Array of raw probabilities (0.0-1.0).
        curve: Output of any ``fit_*_curve`` function.

    Returns:
        Array of calibrated probabilities, clamped to [0.01, 0.99].
    """
    xs = np.asarray(curve["x"], dtype=float)
    ys = np.asarray(curve["y"], dtype=float)
    out = np.interp(np.asarray(probs, dtype=float), xs, ys)
    return np.clip(out, _CLAMP_LO, _CLAMP_HI)


def apply_calibration(
    prob: float,
    bins: Optional[List[Dict[str, float]]] = None,
    curve: Optional[Dict] = None,
) -> float:
    """Apply calibration adjustment to a single probability.

    Prefers the fitted curve (monotonic, smooth) when provided. Falls back
    to linear interpolation between empirical bin midpoints.

    Args:
        prob: Raw model probability (0.0-1.0).
        bins: Output of :func:`pmt.calibration.fit_calibration_bins` (fallback).
        curve: Output of any ``fit_*_curve`` function (preferred).

    Returns:
        Calibrated probability, clamped to [0.01, 0.99]. Returned unchanged
        if neither a curve nor bins are provided.
    """
    # Preferred path: fitted curve
    if curve and curve.get("x") and curve.get("y"):
        return float(apply_curve(np.array([prob]), curve)[0])

    # Fallback: empirical bin interpolation
    if not bins:
        return prob

    preds = [b["pred_mid"] for b in bins]
    actuals = [b["actual_rate"] for b in bins]

    # Clamp to the range covered by calibration data
    if prob <= preds[0]:
        return max(_CLAMP_LO, min(_CLAMP_HI, actuals[0]))
    if prob >= preds[-1]:
        return max(_CLAMP_LO, min(_CLAMP_HI, actuals[-1]))

    # Linear interpolation between adjacent bins
    for i in range(len(preds) - 1):
        if preds[i] <= prob <= preds[i + 1]:
            t = (prob - preds[i]) / (preds[i + 1] - preds[i])
            calibrated = actuals[i] + t * (actuals[i + 1] - actuals[i])
            return max(_CLAMP_LO, min(_CLAMP_HI, calibrated))

    return prob


def apply_multiclass_calibration(
    raw_probs: Dict[str, float],
    curves: Optional[Dict[str, Dict]] = None,
    bins: Optional[Dict[str, List[Dict]]] = None,
) -> Optional[Dict[str, float]]:
    """Apply per-class calibration to mutually exclusive outcomes and renormalize.

    Each class probability is calibrated independently against its own
    curve (each class's calibrator is fit one-vs-rest), then the results
    are renormalized to sum to 1. Typical use: a 3-way market such as
    ``{"home": p, "draw": p, "away": p}``.

    Args:
        raw_probs: ``{class_key: probability}`` on the 0-1 scale.
        curves: ``{class_key: curve_dict}`` (preferred).
        bins: ``{class_key: bins_list}`` (fallback).

    Returns:
        Calibrated probs dict on the 0-1 scale (sums to 1), or ``None`` if
        any class lacks calibration data.
    """
    if not curves and not bins:
        return None

    cal: Dict[str, float] = {}
    for k in raw_probs:
        curve = curves.get(k) if curves else None
        b = bins.get(k) if bins else None
        if not curve and not b:
            return None
        cal[k] = apply_calibration(raw_probs[k], bins=b, curve=curve)

    s = sum(cal.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in cal.items()}
