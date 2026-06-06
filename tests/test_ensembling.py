import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pmt.ensembling import (
    TemporalStackingTrainer,
    assert_no_training_leakage,
)

FACTORIES = {
    "lr": lambda: LogisticRegression(),
    "rf": lambda: RandomForestClassifier(n_estimators=30, random_state=0),
    "gbm": lambda: GradientBoostingClassifier(n_estimators=30, random_state=0),
}


def make_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    z = X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3] + 0.3 * np.sin(3 * X[:, 4])
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def trained():
    X, y = make_data()
    trainer = TemporalStackingTrainer(FACTORIES)
    result = trainer.train(X, y)
    return trainer, result, (X, y)


# --- guard ------------------------------------------------------------------------


def test_leakage_guard_raises():
    with pytest.raises(ValueError, match="ANTI-LEAKAGE"):
        assert_no_training_leakage("leaky", reported_train_rows=2000, i_base=1500)


def test_leakage_guard_passes_at_boundary():
    assert_no_training_leakage("ok", reported_train_rows=1500, i_base=1500)


def test_prefitted_leaky_base_rejected_before_training():
    X, y = make_data(n=1000)
    leaky = LogisticRegression().fit(X, y)  # fit on EVERYTHING
    trainer = TemporalStackingTrainer({"lr": lambda: LogisticRegression()})
    with pytest.raises(ValueError, match="ANTI-LEAKAGE"):
        trainer.train(X, y, prefitted_bases={"lr": (leaky, len(X))})


def test_prefitted_valid_base_accepted():
    X, y = make_data(n=1000)
    i_base = 500  # base_frac=0.5
    safe = LogisticRegression().fit(X[:i_base], y[:i_base])
    trainer = TemporalStackingTrainer({"lr": lambda: LogisticRegression()})
    result = trainer.train(X, y, prefitted_bases={"lr": (safe, i_base)})
    assert result.ensemble_raw is not None


# --- training ----------------------------------------------------------------------


def test_splits_partition_rows(trained):
    _, result, (X, _) = trained
    assert sum(result.splits.values()) == len(X)
    assert result.splits["base"] == len(X) // 2


def test_degenerate_split_raises():
    X, y = make_data(n=10)
    with pytest.raises(ValueError, match="Degenerate split"):
        TemporalStackingTrainer(FACTORIES).train(X, y, base_frac=0.95, meta_frac=0.04)


def test_no_factories_raises():
    with pytest.raises(ValueError, match="At least one"):
        TemporalStackingTrainer({})


def test_per_base_metrics_present(trained):
    _, result, _ = trained
    assert set(result.base_metrics) == set(FACTORIES)
    for metrics in result.base_metrics.values():
        assert "brier" in metrics


def test_ensemble_beats_average_base(trained):
    _, result, _ = trained
    mean_base_brier = np.mean([m["brier"] for m in result.base_metrics.values()])
    assert result.ensemble_raw["brier"] < mean_base_brier


def test_meta_calibration_metrics(trained):
    _, result, _ = trained
    assert result.curve is not None
    assert result.ensemble_calibrated is not None
    # Calibration may not improve Brier on an already-decent meta-model,
    # but it must not catastrophically hurt it.
    assert result.ensemble_calibrated["brier"] < result.ensemble_raw["brier"] * 1.1


# --- inference ----------------------------------------------------------------------


def test_predict_proba_shape_and_normalization(trained):
    trainer, _, (X, _) = trained
    probs = trainer.predict_proba(X[:50])
    assert probs.shape == (50, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_before_train_raises():
    trainer = TemporalStackingTrainer(FACTORIES)
    with pytest.raises(RuntimeError, match="train"):
        trainer.predict_proba(np.zeros((3, 6)))


# --- meta-feature hook -----------------------------------------------------------------


def test_meta_feature_hook_used():
    X, y = make_data(n=1500)
    calls = []

    def spread(base_probs):
        calls.append(base_probs.shape)
        return np.std(base_probs, axis=1)  # 1-d: framework reshapes

    trainer = TemporalStackingTrainer(
        {"lr": lambda: LogisticRegression(),
         "gbm": lambda: GradientBoostingClassifier(n_estimators=20, random_state=0)},
        meta_feature_fn=spread,
    )
    result = trainer.train(X, y)
    assert result.ensemble_raw is not None
    assert calls  # hook was invoked
    # meta-model saw base probs + 1 extra column
    assert trainer._meta_model.coef_.shape[1] == 3
