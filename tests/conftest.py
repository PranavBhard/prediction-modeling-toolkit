import numpy as np
import pytest


def make_overconfident(n: int = 4000, scale: float = 1.7, seed: int = 7):
    """Synthetic miscalibrated predictions.

    True win probabilities drawn from Beta(2, 2); outcomes are Bernoulli
    draws; the "model" reports probabilities with logits stretched by
    ``scale`` (> 1 = overconfident).
    """
    rng = np.random.default_rng(seed)
    true_p = rng.beta(2, 2, n)
    y = (rng.random(n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    pred = 1.0 / (1.0 + np.exp(-logit * scale))
    return pred, y


@pytest.fixture
def overconfident_data():
    return make_overconfident()
