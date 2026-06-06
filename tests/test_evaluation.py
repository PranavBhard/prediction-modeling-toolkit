import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pmt.evaluation import (
    brier_score,
    evaluate_model_cv,
    evaluate_predictions,
    evaluate_with_calibration,
    expected_calibration_error,
)
from conftest import make_overconfident


def test_brier_perfect_predictions():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, np.array([0.0, 1.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_brier_1d_matches_2d():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
    assert brier_score(y, p) == pytest.approx(brier_score(y, np.column_stack([1 - p, p])))


def test_brier_multiclass():
    y = np.array([0, 1, 2])
    proba = np.eye(3)
    assert brier_score(y, proba) == pytest.approx(0.0)
    uniform = np.full((3, 3), 1 / 3)
    assert brier_score(y, uniform) == pytest.approx(2 / 3, abs=1e-9)


def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(11)
    n = 20000
    true_p = rng.beta(2, 2, n)
    y = (rng.random(n) < true_p).astype(int)
    ece_good = expected_calibration_error(y, true_p)
    logit = np.log(true_p / (1 - true_p))
    over = 1 / (1 + np.exp(-logit * 2.0))
    ece_bad = expected_calibration_error(y, over)
    assert ece_good < 0.02
    assert ece_bad > ece_good * 2


def test_evaluate_predictions_keys():
    pred, y = make_overconfident(n=500)
    out = evaluate_predictions(y, pred)
    assert out["n_samples"] == 500
    for key in ("accuracy", "log_loss", "brier", "roc_auc", "ece"):
        assert key in out
    assert 0.0 <= out["accuracy"] <= 1.0


def test_evaluate_predictions_single_class_auc_none():
    y = np.ones(50, dtype=int)
    out = evaluate_predictions(y, np.full(50, 0.7))
    assert out["roc_auc"] is None


def _linear_data(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    z = X @ np.array([1.2, -0.8, 0.5, 0.0]) + 0.2
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


def test_evaluate_model_cv():
    X, y = _linear_data()
    out = evaluate_model_cv(X, y, lambda: LogisticRegression(), n_splits=4)
    assert out["n_folds"] == 4
    assert len(out["folds"]) == 4
    assert out["accuracy_mean"] > 0.6  # learnable signal
    assert out["brier_mean"] < 0.25  # better than coin-flip baseline


class _OverconfidentModel:
    """Estimator whose probabilities are the first feature, overconfident."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)) * 2.0
        p_over = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p_over, p_over])


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "beta", "temperature"])
def test_evaluate_with_calibration_improves(method):
    rng = np.random.default_rng(5)
    n = 6000
    true_p = rng.beta(2, 2, n)
    X = true_p.reshape(-1, 1)
    y = (rng.random(n) < true_p).astype(int)
    out = evaluate_with_calibration(X, y, _OverconfidentModel, method=method)
    assert out["curve"] is not None
    assert out["n_train"] + out["n_cal"] + out["n_eval"] == n
    assert out["calibrated"]["brier"] < out["raw"]["brier"]
    assert out["calibrated"]["ece"] < out["raw"]["ece"]


def test_evaluate_with_calibration_degenerate_split_raises():
    X = np.zeros((5, 1))
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="Degenerate split"):
        evaluate_with_calibration(X, y, _OverconfidentModel, train_frac=0.9, cal_frac=0.2)
