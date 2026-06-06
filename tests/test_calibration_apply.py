import numpy as np
import pytest

from pmt.calibration import (
    apply_calibration,
    apply_curve,
    apply_multiclass_calibration,
)


def identity_curve(n=101):
    xs = list(np.linspace(0.0, 1.0, n))
    return {"x": xs, "y": xs, "method": "isotonic"}


def test_apply_curve_identity():
    probs = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
    out = apply_curve(probs, identity_curve())
    np.testing.assert_allclose(out, probs, atol=1e-9)


def test_apply_curve_clamps():
    out = apply_curve(np.array([0.0, 1.0]), identity_curve())
    assert out[0] == pytest.approx(0.01)
    assert out[1] == pytest.approx(0.99)


def test_apply_curve_interpolates():
    curve = {"x": [0.0, 0.5, 1.0], "y": [0.2, 0.5, 0.8]}
    out = apply_curve(np.array([0.25, 0.75]), curve)
    np.testing.assert_allclose(out, [0.35, 0.65], atol=1e-9)


def test_apply_calibration_prefers_curve_over_bins():
    bins = [
        {"pred_mid": 0.4, "actual_rate": 0.1, "count": 50},
        {"pred_mid": 0.6, "actual_rate": 0.9, "count": 50},
    ]
    assert apply_calibration(0.5, bins=bins, curve=identity_curve()) == pytest.approx(0.5)


def test_apply_calibration_bin_interpolation():
    bins = [
        {"pred_mid": 0.4, "actual_rate": 0.3, "count": 50},
        {"pred_mid": 0.6, "actual_rate": 0.7, "count": 50},
    ]
    assert apply_calibration(0.5, bins=bins) == pytest.approx(0.5)
    # Outside the covered range clamps to the nearest bin's rate
    assert apply_calibration(0.1, bins=bins) == pytest.approx(0.3)
    assert apply_calibration(0.9, bins=bins) == pytest.approx(0.7)


def test_apply_calibration_no_data_passthrough():
    assert apply_calibration(0.42) == pytest.approx(0.42)


def test_multiclass_renormalizes():
    raw = {"home": 0.5, "draw": 0.3, "away": 0.2}
    out = apply_multiclass_calibration(raw, curves={k: identity_curve() for k in raw})
    assert out is not None
    assert sum(out.values()) == pytest.approx(1.0)
    # Identity calibration preserves the ordering
    assert out["home"] > out["draw"] > out["away"]


def test_multiclass_missing_class_returns_none():
    raw = {"home": 0.5, "draw": 0.3, "away": 0.2}
    curves = {"home": identity_curve(), "draw": identity_curve()}  # away missing
    assert apply_multiclass_calibration(raw, curves=curves) is None


def test_multiclass_no_calibration_data_returns_none():
    assert apply_multiclass_calibration({"home": 0.6, "away": 0.4}) is None
