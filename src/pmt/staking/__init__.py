"""Stake sizing: modified Kelly Criterion + empirical bin trust weights.

The probability -> action layer. Given a model probability, a market
price, and the model's measured skill, produce a defensible stake::

    from pmt.staking import calculate_stake, compute_bin_trust_weights

    trust = compute_bin_trust_weights(pnl_bins)
    sizing = calculate_stake(
        p_model=0.62, p_market=0.55,
        model_brier=0.21, bankroll=10_000,
        market_brier=0.22, trust_weights=trust,
    )

See ``docs/stake_sizing.md`` for the full formula walkthrough.
"""

from pmt.staking.bin_trust import compute_bin_trust_weights, lookup_trust
from pmt.staking.kelly import (
    StakeRecommendation,
    calculate_stake,
    prob_to_american_odds,
)

__all__ = [
    "StakeRecommendation",
    "calculate_stake",
    "prob_to_american_odds",
    "compute_bin_trust_weights",
    "lookup_trust",
]
