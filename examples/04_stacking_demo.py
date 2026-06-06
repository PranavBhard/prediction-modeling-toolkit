"""Temporal stacking demo: three base models, a meta-learner, and the
anti-leakage guard catching a deliberately leaked base model.

Run:
    python examples/04_stacking_demo.py

Requires matplotlib for the chart (pip install ".[examples]"); the tables
print either way.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pmt.ensembling import TemporalStackingTrainer

rng = np.random.default_rng(42)

# --- Synthetic data: linear + interaction + periodic signal -------------------
N = 6000
X = rng.normal(size=(N, 6))
z = X[:, 0] - 0.8 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] + 0.4 * np.sin(3 * X[:, 4])
y = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

FACTORIES = {
    "logistic": lambda: LogisticRegression(),
    "random_forest": lambda: RandomForestClassifier(n_estimators=120, random_state=0),
    "gbm": lambda: GradientBoostingClassifier(n_estimators=120, random_state=0),
}

trainer = TemporalStackingTrainer(FACTORIES)
result = trainer.train(X, y)

print("windows:", result.splits)
print(f"\n{'model':<16}{'brier':>9}{'logloss':>9}{'auc':>8}")
print("-" * 42)
for name, m in result.base_metrics.items():
    print(f"{name:<16}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['roc_auc']:>8.4f}")
m = result.ensemble_raw
print(f"{'ENSEMBLE (raw)':<16}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['roc_auc']:>8.4f}")
m = result.ensemble_calibrated
print(f"{'ENSEMBLE (cal)':<16}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['roc_auc']:>8.4f}")

# --- The anti-leakage guard at work -------------------------------------------
print("\n--- anti-leakage guard demo ---")
leaky_model = LogisticRegression().fit(X, y)  # trained on ALL rows, incl. eval
try:
    trainer_bad = TemporalStackingTrainer(FACTORIES)
    trainer_bad.train(X, y, prefitted_bases={"logistic": (leaky_model, N)})
except ValueError as e:
    print(f"caught: {e}")

# --- Chart ---------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed; skipping chart")
    raise SystemExit(0)

names = list(result.base_metrics) + ["ensemble\n(raw)", "ensemble\n(calibrated)"]
briers = [result.base_metrics[n]["brier"] for n in result.base_metrics]
briers += [result.ensemble_raw["brier"], result.ensemble_calibrated["brier"]]
colors = ["tab:gray"] * len(result.base_metrics) + ["tab:blue", "tab:green"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, briers, color=colors, width=0.6)
ax.bar_label(bars, fmt="%.4f", padding=2)
ax.set_ylabel("Brier score (evaluation window, lower = better)")
ax.set_title(
    f"Temporal stacking: {result.splits['base']}/{result.splits['meta']}"
    f"/{result.splits['cal']}/{result.splits['eval']} base/meta/cal/eval split"
)
ax.set_ylim(min(briers) * 0.96, max(briers) * 1.03)
fig.tight_layout()
out = "docs/gallery/ensembling_stacking_vs_base_models.png"
fig.savefig(out, dpi=150)
print(f"\nchart written to {out}")
