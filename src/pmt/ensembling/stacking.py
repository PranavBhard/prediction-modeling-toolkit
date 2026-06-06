"""Temporal stacking ensemble framework (binary classification).

Stacking under time-ordered data has one cardinal rule: **every layer must
be trained on data that is strictly in the past relative to the layer
above it.** A base model that has seen the meta-model's training rows
produces optimistically-biased base predictions there, the meta-model
learns to over-trust it, and the ensemble's offline metrics inflate
silently -- the failure mode is invisible until live performance misses.

This module enforces that rule structurally with a four-window split over
chronologically ordered rows::

    [0 .. i_base)        base-model training
    [i_base .. i_meta)   meta-model training (bases are out-of-sample here)
    [i_meta .. i_cal)    meta-calibrator fitting
    [i_cal .. n)         final evaluation (untouched by all training)

Base models can be fit by the trainer itself (always safe) or supplied
pre-fitted with a reported training-row count, which the **anti-leakage
guard** checks against the window boundary before anything else runs.

Meta-features default to the base models' probabilities. Production
systems typically add derived features (the specifics are where the edge
lives); supply them via ``meta_feature_fn`` without modifying the
framework.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from pmt.calibration import apply_curve, fit_calibration_curve
from pmt.evaluation import evaluate_predictions

__all__ = [
    "StackingResult",
    "TemporalStackingTrainer",
    "assert_no_training_leakage",
]


def assert_no_training_leakage(
    name: str,
    reported_train_rows: int,
    i_base: int,
) -> None:
    """Raise if a pre-fitted base model saw rows beyond its training window.

    Args:
        name: Base model name (for the error message).
        reported_train_rows: Number of rows the model reports being fit on.
            Rows are assumed chronological from index 0.
        i_base: Exclusive end of the base-training window.

    Raises:
        ValueError: when ``reported_train_rows > i_base`` -- the model was
            trained into the meta/calibration/evaluation windows, which
            silently inflates ensemble metrics.
    """
    if reported_train_rows > i_base:
        raise ValueError(
            f"ANTI-LEAKAGE FAILURE: base model '{name}' reports "
            f"{reported_train_rows} training rows, which exceeds the "
            f"base-training window of {i_base} rows. The model was trained "
            f"on meta-training/calibration/evaluation data; its out-of-sample "
            f"predictions there are optimistic and the ensemble's metrics "
            f"would be silently inflated. Re-train it on rows [0, {i_base}) only."
        )


@dataclass
class StackingResult:
    """Output of :meth:`TemporalStackingTrainer.train`.

    Attributes:
        base_metrics: Per-base-model metric dicts on the evaluation window.
        ensemble_raw: Ensemble metrics before meta-calibration.
        ensemble_calibrated: Ensemble metrics after meta-calibration
            (``None`` if calibration fitting failed, e.g. tiny window).
        curve: The fitted meta-calibrator curve dict (serializable).
        splits: Window sizes ``{"base", "meta", "cal", "eval"}``.
    """

    base_metrics: Dict[str, dict] = field(default_factory=dict)
    ensemble_raw: Optional[dict] = None
    ensemble_calibrated: Optional[dict] = None
    curve: Optional[dict] = None
    splits: Dict[str, int] = field(default_factory=dict)


class TemporalStackingTrainer:
    """Train and apply a stacking ensemble under temporal discipline.

    Args:
        base_model_factories: ``{name: zero-arg factory}`` returning fresh
            unfitted sklearn-style estimators (``fit``/``predict_proba``).
        meta_model_factory: Factory for the meta-model (default: logistic
            regression -- simple meta-models resist overfitting the small
            meta window).
        meta_feature_fn: Optional ``f(base_probs) -> extra_features`` where
            ``base_probs`` is ``(n, k)`` (one column per base model's
            positive-class probability) and the return is ``(n, m)``.
            Extra features are appended to the base probabilities.
        calibration_method: Meta-calibrator method (see
            ``pmt.calibration.CALIBRATION_METHODS``).

    Usage::

        trainer = TemporalStackingTrainer({
            "lr": lambda: LogisticRegression(),
            "gbm": lambda: GradientBoostingClassifier(),
        })
        result = trainer.train(X, y)            # rows chronological
        probs = trainer.predict_proba(X_new)    # calibrated ensemble probs
    """

    def __init__(
        self,
        base_model_factories: Dict[str, Callable],
        meta_model_factory: Optional[Callable] = None,
        meta_feature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        calibration_method: str = "isotonic",
    ):
        if not base_model_factories:
            raise ValueError("At least one base model factory is required")
        self.base_model_factories = dict(base_model_factories)
        self.meta_model_factory = meta_model_factory or (lambda: LogisticRegression())
        self.meta_feature_fn = meta_feature_fn
        self.calibration_method = calibration_method

        self._base_models: Dict[str, Any] = {}
        self._meta_model = None
        self._curve: Optional[dict] = None

    # --- internals -----------------------------------------------------------

    @staticmethod
    def _window_indices(n: int, base_frac: float, meta_frac: float, cal_frac: float):
        i_base = int(n * base_frac)
        i_meta = int(n * (base_frac + meta_frac))
        i_cal = int(n * (base_frac + meta_frac + cal_frac))
        if not (0 < i_base < i_meta < i_cal < n):
            raise ValueError(
                f"Degenerate split for n={n}: base={i_base}, "
                f"meta={i_meta - i_base}, cal={i_cal - i_meta}, eval={n - i_cal}"
            )
        return i_base, i_meta, i_cal

    def _base_probs(self, X: np.ndarray) -> np.ndarray:
        """(n, k) matrix of each base model's positive-class probability."""
        return np.column_stack(
            [self._base_models[name].predict_proba(X)[:, 1]
             for name in self.base_model_factories]
        )

    def _meta_features(self, base_probs: np.ndarray) -> np.ndarray:
        if self.meta_feature_fn is None:
            return base_probs
        extra = np.asarray(self.meta_feature_fn(base_probs), dtype=float)
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        return np.column_stack([base_probs, extra])

    # --- training -------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        base_frac: float = 0.5,
        meta_frac: float = 0.25,
        cal_frac: float = 0.125,
        prefitted_bases: Optional[Dict[str, Tuple[Any, int]]] = None,
    ) -> StackingResult:
        """Train the full stack under the four-window temporal split.

        Args:
            X: Feature matrix, rows in chronological order.
            y: Binary target vector (0 or 1).
            base_frac: Fraction of rows for base-model training.
            meta_frac: Fraction for meta-model training.
            cal_frac: Fraction for meta-calibrator fitting (the remainder
                is the evaluation window).
            prefitted_bases: Optional ``{name: (fitted_model, n_train_rows)}``.
                Each entry replaces fitting that base from its factory; the
                anti-leakage guard validates ``n_train_rows`` against the
                base window FIRST and raises on violation.

        Returns:
            :class:`StackingResult` with per-base and ensemble metrics on
            the untouched evaluation window.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        i_base, i_meta, i_cal = self._window_indices(n, base_frac, meta_frac, cal_frac)

        # Guard BEFORE any training happens
        prefitted_bases = prefitted_bases or {}
        for name, (_, n_train_rows) in prefitted_bases.items():
            assert_no_training_leakage(name, n_train_rows, i_base)

        # 1. Base models: fit on the base window only (or accept guarded prefits)
        self._base_models = {}
        for name, factory in self.base_model_factories.items():
            if name in prefitted_bases:
                self._base_models[name] = prefitted_bases[name][0]
            else:
                model = factory()
                model.fit(X[:i_base], y[:i_base])
                self._base_models[name] = model

        # 2. Meta-model: fit on base predictions over the meta window
        #    (out-of-sample for every base model)
        meta_X = self._meta_features(self._base_probs(X[i_base:i_meta]))
        self._meta_model = self.meta_model_factory()
        self._meta_model.fit(meta_X, y[i_base:i_meta])

        # 3. Meta-calibrator: fit on the calibration window
        cal_probs = self._meta_model.predict_proba(
            self._meta_features(self._base_probs(X[i_meta:i_cal]))
        )[:, 1]
        self._curve = fit_calibration_curve(
            cal_probs, y[i_meta:i_cal], method=self.calibration_method
        )

        # 4. Evaluate everything on the untouched final window
        X_eval, y_eval = X[i_cal:], y[i_cal:]
        base_probs_eval = self._base_probs(X_eval)

        result = StackingResult(
            splits={"base": i_base, "meta": i_meta - i_base,
                    "cal": i_cal - i_meta, "eval": n - i_cal},
        )
        for j, name in enumerate(self.base_model_factories):
            result.base_metrics[name] = evaluate_predictions(y_eval, base_probs_eval[:, j])

        raw = self._meta_model.predict_proba(self._meta_features(base_probs_eval))[:, 1]
        result.ensemble_raw = evaluate_predictions(y_eval, raw)
        if self._curve is not None:
            result.curve = self._curve
            result.ensemble_calibrated = evaluate_predictions(
                y_eval, apply_curve(raw, self._curve)
            )
        return result

    # --- inference --------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated ensemble probabilities, shape ``(n, 2)``."""
        if self._meta_model is None:
            raise RuntimeError("Trainer has not been trained -- call train() first")
        raw = self._meta_model.predict_proba(
            self._meta_features(self._base_probs(np.asarray(X)))
        )[:, 1]
        if self._curve is not None:
            raw = apply_curve(raw, self._curve)
        return np.column_stack([1.0 - raw, raw])
