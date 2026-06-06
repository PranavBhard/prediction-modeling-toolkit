"""Stacking ensembles with structural temporal-leakage protection.

::

    from pmt.ensembling import TemporalStackingTrainer

    trainer = TemporalStackingTrainer({
        "lr": lambda: LogisticRegression(),
        "gbm": lambda: GradientBoostingClassifier(),
    })
    result = trainer.train(X, y)   # chronological rows; four-window split
    probs = trainer.predict_proba(X_new)
"""

from pmt.ensembling.stacking import (
    StackingResult,
    TemporalStackingTrainer,
    assert_no_training_leakage,
)

__all__ = [
    "TemporalStackingTrainer",
    "StackingResult",
    "assert_no_training_leakage",
]
