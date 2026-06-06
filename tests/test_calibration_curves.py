import numpy as np
import pytest

from pmt.calibration import (
    CALIBRATION_METHODS,
    apply_curve,
    fit_calibration_bins,
    fit_calibration_curve,
    fit_isotonic_curve,
)
from pmt.evaluation import brier_score


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_curve_shape(method, overconfident_data):
    pred, y = overconfident_data
    curve = fit_calibration_curve(pred, y, method=method)
    assert curve is not None
    assert curve["method"] == method
    assert len(curve["x"]) == 101
    assert len(curve["y"]) == 101
    assert curve["n_samples"] == len(pred)
    ys = np.asarray(curve["y"])
    assert ys.min() >= 0.0 and ys.max() <= 1.0


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_improves_brier(method, overconfident_data):
    """Every method should improve Brier score on overconfident predictions."""
    pred, y = overconfident_data
    curve = fit_calibration_curve(pred, y, method=method)
    raw = brier_score(y, pred)
    calibrated = brier_score(y, apply_curve(pred, curve))
    assert calibrated < raw


def test_isotonic_is_monotonic(overconfident_data):
    pred, y = overconfident_data
    curve = fit_isotonic_curve(pred, y)
    assert np.all(np.diff(curve["y"]) >= -1e-9)


def test_temperature_detects_overconfidence(overconfident_data):
    """Logits were stretched by 1.7, so fitted T should be > 1 (softening)."""
    pred, y = overconfident_data
    curve = fit_calibration_curve(pred, y, method="temperature")
    assert curve["temperature"] > 1.2


def test_insufficient_data_returns_none():
    pred = np.array([0.4, 0.6, 0.7])
    y = np.array([0, 1, 1])
    for method in CALIBRATION_METHODS:
        assert fit_calibration_curve(pred, y, method=method) is None


def test_unknown_method_raises(overconfident_data):
    pred, y = overconfident_data
    with pytest.raises(ValueError, match="Unknown calibration method"):
        fit_calibration_curve(pred, y, method="platt")


def test_fit_calibration_bins(overconfident_data):
    pred, y = overconfident_data
    bins = fit_calibration_bins(pred, y, bin_width=5, min_count=10)
    assert len(bins) > 0
    mids = [b["pred_mid"] for b in bins]
    assert mids == sorted(mids)
    assert sum(b["count"] for b in bins) <= len(pred)
    for b in bins:
        assert 0.0 <= b["pred_mid"] <= 1.0
        assert 0.0 <= b["actual_rate"] <= 1.0
        assert b["count"] >= 10


def test_bins_respect_min_count():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.45, 0.55, 100)  # all mass in two bins
    y = (rng.random(100) < 0.5).astype(int)
    bins = fit_calibration_bins(pred, y, bin_width=5, min_count=10)
    assert all(b["count"] >= 10 for b in bins)
    assert len(bins) <= 2
