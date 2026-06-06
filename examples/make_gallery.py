"""Regenerate the toolkit-generated images in docs/gallery/.

All charts are produced from synthetic data generated in this script --
nothing here touches any real model or dataset. Run from the repo root:

    python examples/make_gallery.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmt.calibration import (
    CALIBRATION_METHODS,
    apply_curve,
    fit_calibration_bins,
    fit_calibration_curve,
)
from pmt.evaluation import evaluate_with_calibration

OUT_DIR = "docs/gallery"
rng = np.random.default_rng(42)

# --- Shared synthetic data: overconfident model, Bernoulli outcomes ---------
N = 20_000
true_p = rng.beta(2.0, 2.0, N)
outcomes = (rng.random(N) < true_p).astype(int)
logits = np.log(true_p / (1.0 - true_p))
model_probs = 1.0 / (1.0 + np.exp(-logits * 1.7))  # stretched = overconfident

p_fit, y_fit = model_probs[: N // 2], outcomes[: N // 2]
curves = {m: fit_calibration_curve(p_fit, y_fit, method=m) for m in CALIBRATION_METHODS}


def save(fig, name):
    path = f"{OUT_DIR}/{name}"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --- 1. Fitted curve shapes: step-function isotonic vs smooth parametric ----
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="identity (no adjustment)")
for method in CALIBRATION_METHODS:
    c = curves[method]
    label = method
    if method == "temperature":
        label = f"temperature (T={c['temperature']:.2f})"
    ax.plot(c["x"], c["y"], lw=1.8, alpha=0.85, label=label)
ax.set_xlabel("raw model probability")
ax.set_ylabel("calibrated probability")
ax.set_title("Fitted calibration mappings on an overconfident model")
ax.legend()
ax.set_aspect("equal")
save(fig, "calibration_fitted_curve_shapes.png")

# --- 2. Empirical bins vs fitted isotonic curve ------------------------------
bins = fit_calibration_bins(p_fit, y_fit, bin_width=5, min_count=10)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
iso = curves["isotonic"]
ax.plot(iso["x"], iso["y"], color="tab:green", lw=2, label="fitted isotonic curve")
counts = np.array([b["count"] for b in bins], dtype=float)
sizes = 400 * counts / counts.max()
ax.scatter(
    [b["pred_mid"] for b in bins],
    [b["actual_rate"] for b in bins],
    s=sizes,
    alpha=0.55,
    color="tab:blue",
    edgecolors="navy",
    label="empirical 5%-wide bins (area ∝ samples)",
    zorder=3,
)
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("empirical outcome rate (bin)")
ax.set_title("Empirical calibration bins and the isotonic fit through them")
ax.legend(loc="upper left")
ax.set_aspect("equal")
save(fig, "calibration_empirical_bins_vs_isotonic.png")

# --- 3. Temperature scaling family -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
xs = np.linspace(0.001, 0.999, 400)
xl = np.log(xs / (1 - xs))
fitted_T = curves["temperature"]["temperature"]
for T in (0.5, 0.75, 1.0, fitted_T, 3.0):
    ys = 1 / (1 + np.exp(-xl / T))
    style = "-" if T == fitted_T else "--" if T == 1.0 else "-"
    lw = 2.5 if T == fitted_T else 1.2
    label = f"T={T:.2f}" + ("  (fitted: softens overconfidence)" if T == fitted_T else "")
    ax.plot(xs, ys, style, lw=lw, label=label)
ax.set_xlabel("raw model probability")
ax.set_ylabel("calibrated probability")
ax.set_title("Temperature scaling family: T>1 softens, T<1 sharpens")
ax.legend()
ax.set_aspect("equal")
save(fig, "calibration_temperature_scaling_family.png")

# --- 4. Temporal train/calibrate/evaluate: raw vs calibrated metrics ---------
class OverconfidentModel:
    """Toy estimator whose probabilities are the first feature, overconfident."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)) * 1.7
        po = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - po, po])


X = true_p.reshape(-1, 1)
result = evaluate_with_calibration(X, outcomes, OverconfidentModel, method="isotonic")

metrics = ["brier", "log_loss", "ece"]
raw_vals = [result["raw"][m] for m in metrics]
cal_vals = [result["calibrated"][m] for m in metrics]

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, metric, rv, cv in zip(axes, metrics, raw_vals, cal_vals):
    bars = ax.bar(["raw", "calibrated"], [rv, cv], color=["tab:red", "tab:green"], width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=2)
    ax.set_title(metric)
    ax.set_ylim(0, rv * 1.25)
fig.suptitle(
    f"Temporal train/calibrate/evaluate split "
    f"({result['n_train']}/{result['n_cal']}/{result['n_eval']} rows): raw vs calibrated",
    fontsize=11,
)
save(fig, "evaluation_temporal_split_raw_vs_calibrated.png")

print("done")
