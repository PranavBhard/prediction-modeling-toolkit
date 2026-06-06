"""Scoring metrics for probabilistic classifiers.

All functions consume plain arrays: ``y_true`` as integer class labels and
``y_proba`` as either a 1-d array of positive-class probabilities (binary)
or an ``(n, k)`` array of per-class probabilities.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "evaluate_predictions",
]


def _as_proba_matrix(y_proba: np.ndarray) -> np.ndarray:
    """Normalize probability input to an (n, k) matrix."""
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim == 1:
        return np.column_stack([1.0 - y_proba, y_proba])
    return y_proba


def _classify(y_proba: np.ndarray) -> np.ndarray:
    """Predict class labels: threshold for binary, argmax for 3+ classes."""
    if y_proba.shape[1] == 2:
        return (y_proba[:, 1] >= 0.5).astype(int)
    return np.argmax(y_proba, axis=1)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score: standard single-column for binary, multi-class for 3+.

    Binary: ``mean((p_positive - y)^2)`` -- matches sklearn ``brier_score_loss``.
    Multi-class (3+ columns): ``mean(sum_k (p_k - y_k)^2)`` per sample.
    """
    y_true = np.asarray(y_true)
    y_proba = _as_proba_matrix(y_proba)
    if y_proba.shape[1] == 2:
        return float(np.mean((y_proba[:, 1] - y_true.astype(float)) ** 2))
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over fixed-width probability bins (binary).

    ECE = sum over bins of (bin weight) * |mean confidence - mean outcome|.
    Lower is better; a perfectly calibrated model scores ~0.

    Args:
        y_true: Binary outcomes (0 or 1).
        probs: Positive-class probabilities (1-d).
        n_bins: Number of fixed-width bins (default 10).
    """
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - y_true[mask].mean())
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Score a set of probabilistic predictions against outcomes.

    Args:
        y_true: Integer class labels.
        y_proba: 1-d positive-class probabilities (binary) or (n, k) matrix.

    Returns:
        Dict with ``n_samples``, ``accuracy``, ``log_loss``, ``brier``; for
        binary problems also ``roc_auc`` (``None`` if only one class is
        present) and ``ece``.
    """
    y_true = np.asarray(y_true)
    y_proba = _as_proba_matrix(y_proba)
    preds = _classify(y_proba)

    out: Dict = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "log_loss": float(
            log_loss(y_true, y_proba, labels=list(range(y_proba.shape[1])))
        ),
        "brier": brier_score(y_true, y_proba),
    }

    if y_proba.shape[1] == 2:
        roc_auc: Optional[float]
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            if np.isnan(roc_auc):  # single-class y_true (newer sklearn)
                roc_auc = None
        except ValueError:  # single-class y_true (older sklearn)
            roc_auc = None
        out["roc_auc"] = roc_auc
        out["ece"] = expected_calibration_error(y_true, y_proba[:, 1])

    return out
