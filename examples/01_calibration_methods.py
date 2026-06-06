"""Compare calibration methods on a synthetic overconfident classifier.

Generates predictions whose logits are stretched 1.7x (a common failure
mode of boosted-tree and deep models), fits all four calibration methods
on a held-out window, and scores raw vs. calibrated probabilities on a
final evaluation window.

Run:
    python examples/01_calibration_methods.py

Requires matplotlib for the reliability diagram (pip install ".[examples]");
the metrics table prints either way.
"""

import numpy as np

from pmt.calibration import CALIBRATION_METHODS, apply_curve, fit_calibration_curve
from pmt.evaluation import brier_score, expected_calibration_error

rng = np.random.default_rng(42)

# --- Synthetic data: true probs, Bernoulli outcomes, overconfident model ---
N = 20_000
true_p = rng.beta(2.0, 2.0, N)
outcomes = (rng.random(N) < true_p).astype(int)
logits = np.log(true_p / (1.0 - true_p))
model_probs = 1.0 / (1.0 + np.exp(-logits * 1.7))  # stretched = overconfident

# Temporal-style split: fit calibrators on the first half, evaluate on the second
p_fit, y_fit = model_probs[: N // 2], outcomes[: N // 2]
p_eval, y_eval = model_probs[N // 2 :], outcomes[N // 2 :]

# --- Fit every method and score on the held-out window ---
rows = [("raw (uncalibrated)", brier_score(y_eval, p_eval),
         expected_calibration_error(y_eval, p_eval))]
curves = {}
for method in CALIBRATION_METHODS:
    curve = fit_calibration_curve(p_fit, y_fit, method=method)
    curves[method] = curve
    calibrated = apply_curve(p_eval, curve)
    rows.append((method, brier_score(y_eval, calibrated),
                 expected_calibration_error(y_eval, calibrated)))

print(f"{'method':<22}{'brier':>10}{'ece':>10}")
print("-" * 42)
for name, brier, ece in rows:
    print(f"{name:<22}{brier:>10.5f}{ece:>10.5f}")
T = curves["temperature"]["temperature"]
print(f"\nfitted temperature T = {T:.3f}  (T > 1 confirms overconfidence)")

# --- Reliability diagram ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed; skipping reliability diagram")
    raise SystemExit(0)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

bin_edges = np.linspace(0, 1, 21)
mids = (bin_edges[:-1] + bin_edges[1:]) / 2


def reliability(probs, y):
    xs, ys = [], []
    for lo, hi, mid in zip(bin_edges[:-1], bin_edges[1:], mids):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 50:
            xs.append(probs[mask].mean())
            ys.append(y[mask].mean())
    return xs, ys

ax.plot(*reliability(p_eval, y_eval), "o-", label="raw (overconfident)")
for method in CALIBRATION_METHODS:
    calibrated = apply_curve(p_eval, curves[method])
    ax.plot(*reliability(calibrated, y_eval), ".-", alpha=0.8, label=method)

ax.set_xlabel("mean predicted probability")
ax.set_ylabel("empirical outcome rate")
ax.set_title("Reliability diagram: raw vs. calibrated")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()

out_path = "docs/gallery/calibration_reliability_diagram.png"
fig.savefig(out_path, dpi=150)
print(f"\nreliability diagram written to {out_path}")
