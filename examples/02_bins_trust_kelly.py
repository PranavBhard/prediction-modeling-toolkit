"""End-to-end probability -> action: P&L bins -> trust weights -> Kelly stakes.

Simulates a season of settled positions with a realistic flaw -- the model's
edge is real for favorites but illusory for longshots -- then shows how bin
trust weights learn that pattern from realized ROI and damp longshot stakes
automatically.

Run:
    python examples/02_bins_trust_kelly.py

Requires matplotlib for the chart (pip install ".[examples]"); the tables
print either way.
"""

import numpy as np

from pmt.staking import StakeRecommendation, calculate_stake, compute_bin_trust_weights

rng = np.random.default_rng(42)

# --- 1. Simulate settled positions: edge real for favorites, not longshots ---
BIN_WIDTH = 10
N_POSITIONS = 2_000

p_market = rng.uniform(0.05, 0.95, N_POSITIONS)
# True probability: above market for favorites (real edge), below for longshots
true_p = np.clip(p_market + np.where(p_market > 0.4, 0.04, -0.05), 0.01, 0.99)
won = rng.random(N_POSITIONS) < true_p
# Unit stake at fair market odds: win pays (1/p_market - 1), loss pays -1
pnl = np.where(won, 1.0 / p_market - 1.0, -1.0)

bins = []
for lo in range(0, 100, BIN_WIDTH):
    mask = (p_market * 100 >= lo) & (p_market * 100 < lo + BIN_WIDTH)
    count = int(mask.sum())
    roi = float(pnl[mask].mean() * 100) if count else 0.0
    bins.append({"low": lo, "high": lo + BIN_WIDTH, "count": count, "roi": roi})

# --- 2. ROI bins -> trust weights ---------------------------------------------
trust = compute_bin_trust_weights(bins, min_samples=10)

print(f"{'bin':<10}{'count':>7}{'roi%':>9}{'raw':>7}{'shrunk':>8}{'trust':>7}")
print("-" * 48)
for t in trust:
    print(f"{t['prob_low']:>3.0f}-{t['prob_high']:<5.0f}{t['count']:>7}"
          f"{t['roi']:>9.1f}{t['raw_trust']:>7.2f}{t['shrunk']:>8.2f}{t['trust']:>7.2f}")

# --- 3. Size a slate of hypothetical edges with and without trust -------------
MODEL_BRIER, MARKET_BRIER = 0.205, 0.215  # model slightly beats the market
BANKROLL = 10_000

slate = [
    ("evt-01", "LONGSHOT A", 0.18, 0.12),
    ("evt-02", "LONGSHOT B", 0.30, 0.24),
    ("evt-03", "COINFLIP C", 0.55, 0.50),
    ("evt-04", "FAVORITE D", 0.68, 0.60),
    ("evt-05", "FAVORITE E", 0.85, 0.80),
]

print(f"\n{'selection':<12}{'p_mod':>7}{'p_mkt':>7}{'edge':>8}"
      f"{'trust':>7}{'stake':>9}{'(no trust)':>11}")
print("-" * 61)
recs = []
for event_id, sel, p_model, p_mkt in slate:
    kwargs = dict(model_brier=MODEL_BRIER, bankroll=BANKROLL,
                  market_brier=MARKET_BRIER)
    sized = calculate_stake(p_model, p_mkt, trust_weights=trust, **kwargs)
    naive = calculate_stake(p_model, p_mkt,
                            trust_weights=[{"prob_low": 0, "prob_high": 100,
                                            "trust": 1.0}], **kwargs)
    rec = StakeRecommendation.from_sizing(event_id, sel, p_model, p_mkt, sized)
    recs.append((rec, naive["stake"]))
    print(f"{sel:<12}{p_model:>7.2f}{p_mkt:>7.2f}{sized['edge']:>8.3f}"
          f"{sized['trust_weight']:>7.2f}{sized['stake']:>9.2f}{naive['stake']:>11.2f}")

print("\nNote how the trust layer cuts the longshot stakes (where the simulated"
      "\nedge was illusory) while leaving favorite stakes nearly untouched.")

# --- 4. Chart ------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed; skipping chart")
    raise SystemExit(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mids = [(t["prob_low"] + t["prob_high"]) / 2 for t in trust]
ax1.axhline(1.0, color="gray", lw=1, ls="--", label="neutral trust")
ax1.plot(mids, [t["raw_trust"] for t in trust], "o--", alpha=0.5, label="raw (1 + roi/100)")
ax1.plot(mids, [t["shrunk"] for t in trust], "s--", alpha=0.5, label="after shrinkage")
ax1.plot(mids, [t["trust"] for t in trust], "o-", lw=2, label="final (smoothed)")
ax1.set_xlabel("market probability bin midpoint (%)")
ax1.set_ylabel("trust weight")
ax1.set_title("Bin trust pipeline: raw ROI → shrunk → smoothed")
ax1.legend()

labels = [r.selection for r, _ in recs]
x = np.arange(len(labels))
ax2.bar(x - 0.2, [n for _, n in recs], width=0.4, color="tab:gray",
        label="uniform trust")
ax2.bar(x + 0.2, [r.stake for r, _ in recs], width=0.4, color="tab:green",
        label="bin trust applied")
ax2.set_xticks(x, labels, rotation=20, ha="right")
ax2.set_ylabel(f"stake ($, bankroll {BANKROLL:,})")
ax2.set_title("Sized slate: trust damps longshots, keeps favorites")
ax2.legend()

fig.tight_layout()
out = "docs/gallery/staking_bin_trust_kelly_pipeline.png"
fig.savefig(out, dpi=150)
print(f"\nchart written to {out}")
