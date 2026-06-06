# Stake Sizing Formula

`pmt.staking` sizes positions with a modified Kelly Criterion. The full
calculation lives in `pmt/staking/kelly.py::calculate_stake()`; the
empirical trust layer lives in `pmt/staking/bin_trust.py`.

## Inputs

| Parameter | Description |
|-----------|-------------|
| `p_model` | Model's predicted probability (0–1) |
| `p_market` | Market price / implied probability (0–1) |
| `model_brier` | Model's Brier score (lower = better) |
| `model_log_loss` | Model's log-loss (optional) |
| `market_brier` | Market's Brier score baseline over the same window |
| `market_log_loss` | Market's log-loss baseline |
| `bankroll` | Total bankroll |
| `trust_weights` | Per-bin trust weights (from `compute_bin_trust_weights`) |

If `market_brier` / `market_log_loss` are not provided, they default to the
model's own scores, which means skill = 0 (no edge assumed).

## Step 1: Skill Score

Measures how much better the model is than the market.

```
skill_bs = 1 - (model_brier / market_brier)
skill_ll = 1 - (model_log_loss / market_log_loss)
skill    = 0.6 * skill_ll + 0.4 * skill_bs
```

- `skill = 0` → model is no better than market
- `skill > 0` → model outperforms market
- `skill < 0` → model is worse than market

Log-loss is weighted higher (0.6) because it penalizes confident wrong
predictions more harshly.

## Step 2: Probability Shrinkage

Blends the model probability toward the market price, controlled by skill.

```
shrinkage_w = clamp(0.50 + 2.0 * skill, 0.10, 0.50)
p_adj       = shrinkage_w * p_model + (1 - shrinkage_w) * p_market
```

| Skill | shrinkage_w | Interpretation |
|-------|-------------|----------------|
| -0.20 | 0.10 | 90% market, 10% model (bad model) |
| 0.00 | 0.50 | 50/50 blend |
| +0.20 | 0.50 | still 50/50 — capped |

The cap at 0.50 is deliberately conservative: a model that measures better
than the market is treated as the market's *equal* for sizing purposes,
never its better. Markets are hard to beat, and measured skill is partly
luck; the cap prevents a good evaluation window from inflating every stake.
The bounds are configurable via `shrinkage_bounds` if you want the
aggressive variant (e.g. `(0.10, 0.90)`).

## Step 3: Kelly Edge

Classic Kelly Criterion using the adjusted probability.

```
market_odds = 1 / p_market
edge_kelly  = (p_adj * market_odds - 1) / (market_odds - 1)
```

This is the theoretical optimal fraction of bankroll to stake.
Positive = +EV position.

## Step 4: Edge Gate

Ramps out tiny edges that are likely noise.

```
edge      = p_adj - p_market
edge_gate = clamp((|edge| - 0.01) / 0.04, 0, 1)
```

| Edge | Gate | Effect |
|------|------|--------|
| ≤ 1% | 0.0 | No stake — edge too small to be real |
| 3% | 0.5 | Half-sized stake |
| ≥ 5% | 1.0 | Full stake |

Thresholds configurable via `edge_gate_start` / `edge_gate_full`.

## Step 5: Bin Trust Weight

Scales the stake by the model's historical realized P&L in this
probability range. See [Bin Trust Weights](#bin-trust-weights) below.

- With `trust_weights`: looks up the trust for the bin containing `p_adj`
- Fallback (no trust data): `min(1.0, p_adj / 0.30)` — ramps linearly
  from 0 at 0% to 1.0 at 30%+, a crude longshot damper

## Step 6: Final Stake

```
stake_fraction = edge_kelly × 0.25 × edge_gate × trust_weight
stake          = bankroll × stake_fraction
```

The `0.25` is quarter-Kelly — the standard variance dampener vs. full
Kelly. Each multiplier gates the position:

- **edge_kelly**: how much Kelly says to stake
- **0.25**: variance dampener (quarter-Kelly; `kelly_fraction` kwarg)
- **edge_gate**: zeros out noise-level edges
- **trust_weight**: scales down stakes in probability ranges where the
  edge historically failed to materialize

---

## Bin Trust Weights

Computed by `pmt/staking/bin_trust.py::compute_bin_trust_weights()`.

### What it answers

"In each probability bin (e.g. 20–25%, 45–50%), does our model's edge
actually materialize into realized P&L?"

### Input

A list of bins, each with `low`/`high` (probability bounds, 0–100 scale),
`count` (number of settled positions), and `roi` (realized ROI as a
percentage, e.g. `15.0` means +15%).

### Pipeline: 3 steps

#### Step 1: Raw Trust

Direct mapping from historical ROI to a multiplier:

```
raw_trust = 1.0 + roi / 100
```

| Historical ROI | Raw Trust | Meaning |
|----------------|-----------|---------|
| +50% | 1.50 | Boost stakes 50% |
| 0% | 1.00 | Break even, neutral |
| -30% | 0.70 | Cut stakes by 30% |
| -100% | 0.00 | Clamped to min_trust (0.1) |

Clamped to `[min_trust=0.1, max_trust=1.5]`.
Empty bins (count = 0) get `default_trust = 1.0`.

#### Step 2: Sample-Size Shrinkage

Bins with few positions have noisy ROI. Shrinkage pulls them toward
neutral (1.0):

```
weight       = count / (count + min_samples)
shrunk_trust = weight * raw_trust + (1 - weight) * default_trust
```

`min_samples` defaults to 10.

| Count | Weight | Effect |
|-------|--------|--------|
| 0 | 0.00 | 100% default (1.0) |
| 5 | 0.33 | 33% raw, 67% default |
| 10 | 0.50 | 50/50 |
| 30 | 0.75 | 75% raw, 25% default |
| 100 | 0.91 | Mostly raw data |

This is Bayesian shrinkage — with a few positions, we don't trust the ROI
signal; with many, we do.

#### Step 3: Neighbor Smoothing

Weighted average with adjacent bins to reduce noise from bin boundary
effects:

```
smoothed[i] = (1.0 * shrunk[i] + 0.5 * shrunk[i-1] + 0.5 * shrunk[i+1]) / total_weight
```

- Center bin has weight 1.0; each neighbor at distance `d` has weight `0.5^d`
- `smooth_neighbors` defaults to 1 (±1 bin)
- Final result clamped to `[min_trust, max_trust]`

This prevents a single bin from being wildly different from its neighbors
due to small sample sizes.

### Lookup at sizing time

`lookup_trust(trust_weights, p_adj)` converts `p_adj` to a percentage and
finds the matching bin:

```python
prob_pct = p_adj * 100
# Find bin where prob_low <= prob_pct < prob_high (top bin inclusive)
# Return that bin's trust value (default 1.0 if no match)
```

### End-to-end example

See `examples/02_bins_trust_kelly.py` for the full pipeline: synthetic
P&L bins → trust weights → sized stakes across a slate of edges.
