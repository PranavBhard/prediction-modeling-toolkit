"""Calibration curve fitters.

Each fitter maps (raw model probability -> empirical outcome rate) on a
held-out set and returns the fitted curve sampled at ``n_samples`` evenly
spaced points. All fitters share one return shape (``x``/``y`` sample
points plus metadata) so storage and prediction-time application are
method-agnostic: persist the dict anywhere, then apply it with
:func:`pmt.calibration.apply_curve`.

Methods:

- **bins** -- empirical rate per fixed-width probability bin (no smoothing).
- **isotonic** -- monotonic step function via isotonic regression.
- **sigmoid** -- Platt scaling: logistic regression on the logit.
- **temperature** -- single scalar T dividing the logit.
- **beta** -- 3-parameter beta calibration (Kull et al., 2017).
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = [
    "CALIBRATION_METHODS",
    "fit_calibration_bins",
    "fit_isotonic_curve",
    "fit_sigmoid_curve",
    "fit_temperature_curve",
    "fit_beta_curve",
    "fit_calibration_curve",
]

CALIBRATION_METHODS = ("isotonic", "sigmoid", "beta", "temperature")

_MIN_SAMPLES = 20


def fit_calibration_bins(
    predicted: np.ndarray,
    actual: np.ndarray,
    bin_width: int = 5,
    min_count: int = 10,
) -> List[Dict[str, float]]:
    """Fit empirical calibration bins from predictions and outcomes.

    Args:
        predicted: Array of model probabilities (0.0-1.0).
        actual: Array of binary outcomes (0 or 1).
        bin_width: Bin width in percentage points (default 5).
        min_count: Minimum samples per bin to include.

    Returns:
        List of dicts with ``{pred_mid, actual_rate, count}`` for each valid
        bin, sorted by ``pred_mid`` ascending.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    bins = []
    for lo in range(0, 100, bin_width):
        hi = lo + bin_width
        mask = (predicted >= lo / 100) & (predicted < hi / 100)
        count = int(mask.sum())
        if count < min_count:
            continue
        bins.append({
            "pred_mid": round(float(predicted[mask].mean()), 4),
            "actual_rate": round(float(actual[mask].mean()), 4),
            "count": count,
        })
    return bins


def fit_isotonic_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_samples: int = 101,
) -> Optional[Dict]:
    """Fit an isotonic regression calibration curve.

    Provides a smooth, monotonic mapping from raw model probability to
    calibrated probability. Unlike binned calibration, isotonic regression:

    - Guarantees monotonicity (``calibrated(p1) <= calibrated(p2)`` if ``p1 <= p2``)
    - Adaptively merges bins where data is sparse
    - Has no fixed bin width

    Args:
        predicted: Array of model probabilities (0.0-1.0).
        actual: Array of binary outcomes (0 or 1).
        n_samples: Number of curve sample points to store for rendering and
            serialization (default 101 = every 1%).

    Returns:
        Dict with:
            - ``x``: list of input probabilities (sample points 0.0..1.0)
            - ``y``: list of calibrated probabilities at those points
            - ``n_steps``: number of distinct calibrated values (effective bins)
            - ``min_pred``, ``max_pred``: range of input data
            - ``n_samples``: total samples used to fit
            - ``method``: ``"isotonic"``
        Or ``None`` if fitting failed (insufficient data).
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predicted) < _MIN_SAMPLES or len(actual) < _MIN_SAMPLES:
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    try:
        iso.fit(predicted, actual)
    except Exception:
        return None

    xs = np.linspace(0.0, 1.0, n_samples)
    ys = iso.predict(xs)

    # Count distinct steps (effective bins after isotonic merging)
    n_steps = len(np.unique(np.round(ys, 6)))

    return {
        "x": [round(float(x), 4) for x in xs],
        "y": [round(float(y), 4) for y in ys],
        "n_steps": int(n_steps),
        "min_pred": round(float(predicted.min()), 4),
        "max_pred": round(float(predicted.max()), 4),
        "n_samples": int(len(predicted)),
        "method": "isotonic",
    }


def fit_sigmoid_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_samples: int = 101,
) -> Optional[Dict]:
    """Fit a Platt-scaling (sigmoid) calibration curve.

    Fits a logistic regression on the model logits -> outcome, then samples
    the resulting sigmoid at ``n_samples`` evenly spaced points so the curve
    can be stored, served, and applied identically to an isotonic curve.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predicted) < _MIN_SAMPLES:
        return None

    eps = 1e-6
    p = np.clip(predicted, eps, 1 - eps)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(logits, actual.astype(int))
    except Exception:
        return None

    xs = np.linspace(0.0, 1.0, n_samples)
    xs_c = np.clip(xs, eps, 1 - eps)
    x_logits = np.log(xs_c / (1 - xs_c)).reshape(-1, 1)
    ys = lr.predict_proba(x_logits)[:, 1]

    return {
        "x": [round(float(x), 4) for x in xs],
        "y": [round(float(y), 4) for y in ys],
        "n_steps": int(n_samples),
        "min_pred": round(float(predicted.min()), 4),
        "max_pred": round(float(predicted.max()), 4),
        "n_samples": int(len(predicted)),
        "method": "sigmoid",
    }


def fit_temperature_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_samples: int = 101,
) -> Optional[Dict]:
    """Fit a temperature-scaling calibration curve (single scalar T).

    Divides the logit by T fitted to minimize cross-entropy: T > 1 softens
    overconfident predictions, T < 1 sharpens underconfident ones. The
    fitted ``temperature`` is included in the returned dict.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predicted) < _MIN_SAMPLES:
        return None

    eps = 1e-6
    p = np.clip(predicted, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    a = actual.astype(float)

    def loss(T: float) -> float:
        if T <= 0:
            return 1e10
        scaled = logits / T
        # Numerically stable cross-entropy
        log_p1 = -np.logaddexp(0.0, -scaled)
        log_p0 = -np.logaddexp(0.0, scaled)
        return -float(np.mean(a * log_p1 + (1.0 - a) * log_p0))

    try:
        res = minimize_scalar(loss, bounds=(0.05, 10.0), method="bounded")
        T = float(res.x)
    except Exception:
        return None

    xs = np.linspace(0.0, 1.0, n_samples)
    xs_c = np.clip(xs, eps, 1 - eps)
    x_logits = np.log(xs_c / (1 - xs_c)) / T
    ys = 1.0 / (1.0 + np.exp(-x_logits))

    return {
        "x": [round(float(x), 4) for x in xs],
        "y": [round(float(y), 4) for y in ys],
        "n_steps": int(n_samples),
        "min_pred": round(float(predicted.min()), 4),
        "max_pred": round(float(predicted.max()), 4),
        "n_samples": int(len(predicted)),
        "method": "temperature",
        "temperature": round(T, 4),
    }


def fit_beta_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_samples: int = 101,
) -> Optional[Dict]:
    """Fit a 3-parameter beta calibration curve (Kull et al. 2017).

    ``g(p) = a*log(p) - b*log(1-p) + c``, then sigmoid. Fit by minimizing
    log-loss with Nelder-Mead. The logit+log basis handles asymmetric
    S-curves and probability tails better than isotonic or Platt scaling.
    Fitted ``a``, ``b``, ``c`` are included in the returned dict.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predicted) < _MIN_SAMPLES:
        return None

    eps = 1e-6
    p = np.clip(predicted, eps, 1 - eps)
    log_p = np.log(p)
    log_1mp = np.log(1 - p)
    a_arr = actual.astype(float)

    def loss(params: np.ndarray) -> float:
        a, b, c = params
        z = a * log_p - b * log_1mp + c
        z = np.clip(z, -30.0, 30.0)
        log_p1 = -np.logaddexp(0.0, -z)
        log_p0 = -np.logaddexp(0.0, z)
        return -float(np.mean(a_arr * log_p1 + (1.0 - a_arr) * log_p0))

    try:
        res = minimize(loss, x0=np.array([1.0, 1.0, 0.0]), method="Nelder-Mead")
        a_hat, b_hat, c_hat = (float(v) for v in res.x)
    except Exception:
        return None

    xs = np.linspace(0.0, 1.0, n_samples)
    xs_c = np.clip(xs, eps, 1 - eps)
    z = a_hat * np.log(xs_c) - b_hat * np.log(1 - xs_c) + c_hat
    z = np.clip(z, -30.0, 30.0)
    ys = 1.0 / (1.0 + np.exp(-z))

    return {
        "x": [round(float(x), 4) for x in xs],
        "y": [round(float(y), 4) for y in ys],
        "n_steps": int(n_samples),
        "min_pred": round(float(predicted.min()), 4),
        "max_pred": round(float(predicted.max()), 4),
        "n_samples": int(len(predicted)),
        "method": "beta",
        "a": round(a_hat, 4),
        "b": round(b_hat, 4),
        "c": round(c_hat, 4),
    }


def fit_calibration_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    method: str = "isotonic",
    n_samples: int = 101,
) -> Optional[Dict]:
    """Dispatch to the curve fitter for the requested calibration method.

    All fitters return the same dict shape (``x``/``y`` sample points plus
    metadata) so storage and prediction-time application are method-agnostic.

    Raises:
        ValueError: if ``method`` is not one of :data:`CALIBRATION_METHODS`.
    """
    if method == "isotonic":
        return fit_isotonic_curve(predicted, actual, n_samples=n_samples)
    if method == "sigmoid":
        return fit_sigmoid_curve(predicted, actual, n_samples=n_samples)
    if method == "beta":
        return fit_beta_curve(predicted, actual, n_samples=n_samples)
    if method == "temperature":
        return fit_temperature_curve(predicted, actual, n_samples=n_samples)
    raise ValueError(f"Unknown calibration method: {method!r}")
