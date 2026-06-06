"""Temporal cross-validation and calibrated evaluation.

Both entry points assume rows are in **chronological order** -- splits are
always past-train / future-test, never shuffled. Leakage from the future
into training is the cardinal sin of evaluating time-ordered prediction
systems; these helpers make the safe pattern the only pattern.

Models are supplied as a zero-argument factory callable returning a fresh
sklearn-style estimator (``fit`` / ``predict_proba``), so the toolkit never
needs to know how callers construct or configure their models.
"""

from typing import Callable, Dict

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from pmt.calibration import apply_curve, fit_calibration_curve
from pmt.evaluation.metrics import evaluate_predictions

__all__ = [
    "evaluate_model_cv",
    "evaluate_with_calibration",
]


def evaluate_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    n_splits: int = 5,
) -> Dict:
    """Evaluate a model with time-series cross-validation.

    Uses :class:`sklearn.model_selection.TimeSeriesSplit`: each fold trains
    on a prefix of the data and validates on the window that follows it.

    Args:
        X: Feature matrix, rows in chronological order.
        y: Target vector.
        model_factory: Zero-arg callable returning a fresh unfitted estimator.
        n_splits: Number of CV splits.

    Returns:
        Dict with ``folds`` (per-fold metric dicts from
        :func:`evaluate_predictions`) and mean/std summaries for accuracy,
        log-loss, and Brier score.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    for train_idx, val_idx in tscv.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_proba = model.predict_proba(X[val_idx])
        folds.append(evaluate_predictions(y[val_idx], y_proba))

    summary: Dict = {"n_folds": n_splits, "folds": folds}
    for metric in ("accuracy", "log_loss", "brier"):
        values = [f[metric] for f in folds]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    return summary


def evaluate_with_calibration(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    method: str = "isotonic",
) -> Dict:
    """Evaluate a binary model with a temporal train / calibrate / evaluate split.

    The data is split chronologically into three contiguous windows:

    1. **Train** (first ``train_frac``): fit the model.
    2. **Calibrate** (next ``cal_frac``): fit the calibration curve on the
       model's out-of-sample probabilities.
    3. **Evaluate** (remainder): score raw vs. calibrated probabilities.

    The calibrator never sees training rows and the evaluation window never
    leaks into either earlier stage.

    Args:
        X: Feature matrix, rows in chronological order.
        y: Binary target vector (0 or 1).
        model_factory: Zero-arg callable returning a fresh unfitted estimator.
        train_frac: Fraction of rows used to train the model.
        cal_frac: Fraction of rows used to fit the calibrator.
        method: Calibration method (see ``pmt.calibration.CALIBRATION_METHODS``).

    Returns:
        Dict with ``raw`` and ``calibrated`` metric dicts (calibrated is
        ``None`` if curve fitting failed), the fitted ``curve``, and the
        three split sizes.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    i_train = int(n * train_frac)
    i_cal = int(n * (train_frac + cal_frac))
    if not (0 < i_train < i_cal < n):
        raise ValueError(
            f"Degenerate split for n={n}: "
            f"train={i_train}, cal={i_cal - i_train}, eval={n - i_cal}"
        )

    model = model_factory()
    model.fit(X[:i_train], y[:i_train])

    p_cal = model.predict_proba(X[i_train:i_cal])[:, 1]
    p_eval = model.predict_proba(X[i_cal:])[:, 1]
    y_eval = y[i_cal:]

    result: Dict = {
        "raw": evaluate_predictions(y_eval, p_eval),
        "calibrated": None,
        "curve": None,
        "n_train": i_train,
        "n_cal": i_cal - i_train,
        "n_eval": n - i_cal,
    }

    curve = fit_calibration_curve(p_cal, y[i_train:i_cal], method=method)
    if curve is not None:
        result["curve"] = curve
        result["calibrated"] = evaluate_predictions(y_eval, apply_curve(p_eval, curve))
    return result
