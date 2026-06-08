"""Generate a win-probability surface CSV for rendering in ParaView.

Fits a 2-feature logistic model (with an interaction term, so the surface
has curvature) on synthetic data, calibrates it with pmt.calibration, then
evaluates the calibrated win probability over a dense grid. Output is a flat
``x, y, z, prob`` CSV that ParaView ingests via
Table To Points -> Delaunay 2D -> Warp By Scalar.

The ``z`` column is all zeros: ParaView's Table To Points needs three
coordinate columns, and Warp By Scalar then raises the flat sheet by ``prob``
with an adjustable scale factor (vertical exaggeration lives in the GUI, not
the data).

Run from the repo root:
    python examples/paraview/make_probability_surface_csv.py
"""

import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from pmt.calibration import apply_curve, fit_calibration_curve

rng = np.random.default_rng(42)


def features(a):
    """Two raw features plus their interaction (gives the surface a saddle)."""
    return np.column_stack([a[:, 0], a[:, 1], a[:, 0] * a[:, 1]])


# --- Synthetic training data --------------------------------------------------
N = 5000
X = rng.normal(size=(N, 2))
z = 1.3 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 0] * X[:, 1] + 0.4
y = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

split = N // 2
model = LogisticRegression().fit(features(X[:split]), y[:split])

# Calibrate on the held-out half so the surface is the *calibrated* probability
raw_cal = model.predict_proba(features(X[split:]))[:, 1]
curve = fit_calibration_curve(raw_cal, y[split:], method="isotonic")

# --- Evaluate over a grid -----------------------------------------------------
g = np.linspace(-3.0, 3.0, 90)
xx, yy = np.meshgrid(g, g)
grid = np.column_stack([xx.ravel(), yy.ravel()])
prob = apply_curve(model.predict_proba(features(grid))[:, 1], curve)

out = "docs/gallery/data/probability_surface.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "z", "prob"])
    for (xi, yi), p in zip(grid, prob):
        w.writerow([f"{xi:.4f}", f"{yi:.4f}", "0", f"{p:.5f}"])

print(f"wrote {len(grid)} rows ({len(g)}x{len(g)} grid) to {out}")
print(f"prob range: {prob.min():.3f} .. {prob.max():.3f}")
