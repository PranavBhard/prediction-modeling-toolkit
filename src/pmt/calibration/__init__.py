"""Probability calibration: curve fitting and prediction-time application.

Fit a calibration mapping (raw model probability -> empirical outcome rate)
on a held-out set, serialize the fitted curve as a plain dict, and apply it
at prediction time::

    from pmt.calibration import fit_calibration_curve, apply_curve

    curve = fit_calibration_curve(val_probs, val_outcomes, method="isotonic")
    calibrated = apply_curve(test_probs, curve)
"""

from pmt.calibration.apply import (
    apply_calibration,
    apply_curve,
    apply_multiclass_calibration,
)
from pmt.calibration.curves import (
    CALIBRATION_METHODS,
    fit_beta_curve,
    fit_calibration_bins,
    fit_calibration_curve,
    fit_isotonic_curve,
    fit_sigmoid_curve,
    fit_temperature_curve,
)

__all__ = [
    "CALIBRATION_METHODS",
    "fit_calibration_bins",
    "fit_isotonic_curve",
    "fit_sigmoid_curve",
    "fit_temperature_curve",
    "fit_beta_curve",
    "fit_calibration_curve",
    "apply_curve",
    "apply_calibration",
    "apply_multiclass_calibration",
]
