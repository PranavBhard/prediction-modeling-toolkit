"""Stake sizing: modified Kelly Criterion with skill-aware shrinkage.

Sizes a position from a model probability and a market price, defending
against the three classic ways a raw Kelly stake blows up:

1. **Probability shrinkage** -- blend the model's probability toward the
   market price, weighted by the model's measured skill *relative to the
   market*. An overconfident model gets pulled toward the price.
2. **Edge gating** -- ramp stakes from zero on tiny edges that are more
   likely noise than signal.
3. **Trust weighting** -- scale by the empirical trust weight for this
   probability range (see :mod:`pmt.staking.bin_trust`), so ranges where
   edges historically failed to materialize get smaller stakes.

The full derivation with worked examples is in ``docs/stake_sizing.md``.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from pmt.staking.bin_trust import lookup_trust

__all__ = [
    "StakeRecommendation",
    "prob_to_american_odds",
    "calculate_stake",
]


def prob_to_american_odds(prob: float) -> int:
    """Convert a probability (0.0-1.0) to American odds.

    Examples:
        0.60 -> -150 (60% favorite)
        0.40 -> +150 (40% underdog)
        0.50 -> -100 (even)
    """
    if prob <= 0:
        return 0
    if prob >= 1:
        return -10000
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    return int(100 * (1 - prob) / prob)


@dataclass
class StakeRecommendation:
    """A sized position on one selection of one event."""

    event_id: str
    selection: str  # which outcome the stake is on
    model_prob: float  # 0.0-1.0
    market_prob: float  # 0.0-1.0 (implied by price)
    model_odds: int  # American odds for model_prob
    market_odds: int  # American odds for market_prob
    edge: float  # p_adj - p_market
    edge_kelly: float  # Kelly-optimal bankroll fraction
    trust_weight: float  # empirical bin trust applied
    stake_fraction: float  # final fraction of bankroll
    stake: float  # final stake amount
    p_adj: float = 0.0  # shrunk probability actually used
    skill: float = 0.0  # market-relative skill score
    shrinkage_w: float = 0.5  # weight given to the model probability
    edge_gate: float = 1.0  # edge gate multiplier (0-1)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_sizing(
        cls,
        event_id: str,
        selection: str,
        p_model: float,
        p_market: float,
        sizing: Dict[str, float],
    ) -> "StakeRecommendation":
        """Build a recommendation from :func:`calculate_stake` output."""
        return cls(
            event_id=event_id,
            selection=selection,
            model_prob=p_model,
            market_prob=p_market,
            model_odds=prob_to_american_odds(p_model),
            market_odds=prob_to_american_odds(p_market),
            edge=sizing["edge"],
            edge_kelly=sizing["edge_kelly"],
            trust_weight=sizing["trust_weight"],
            stake_fraction=sizing["stake_fraction"],
            stake=sizing["stake"],
            p_adj=sizing["p_adj"],
            skill=sizing["skill"],
            shrinkage_w=sizing["shrinkage_w"],
            edge_gate=sizing["edge_gate"],
        )


def calculate_stake(
    p_model: float,
    p_market: float,
    model_brier: float,
    bankroll: float,
    *,
    model_log_loss: Optional[float] = None,
    market_brier: Optional[float] = None,
    market_log_loss: Optional[float] = None,
    trust_weights: Optional[List[dict]] = None,
    kelly_fraction: float = 0.25,
    shrinkage_bounds: Tuple[float, float] = (0.10, 0.50),
    edge_gate_start: float = 0.01,
    edge_gate_full: float = 0.05,
    log_loss_weight: float = 0.6,
    fallback_trust_ramp: float = 0.30,
) -> Dict[str, float]:
    """Size a stake with Kelly + market-relative skill, shrinkage, and gating.

    Args:
        p_model: Model probability for the selection (0.0-1.0).
        p_market: Market implied probability (0.0-1.0).
        model_brier: Model's Brier score on a relevant evaluation window
            (lower is better).
        bankroll: Total bankroll amount.
        model_log_loss: Model's log-loss (optional; Brier-only skill if omitted).
        market_brier: Market baseline Brier score over the same window.
            Defaults to ``model_brier`` (assumes zero skill -- model no
            better than market).
        market_log_loss: Market baseline log-loss. Defaults to
            ``model_log_loss`` (zero skill).
        trust_weights: Output of
            :func:`pmt.staking.compute_bin_trust_weights` (optional).
        kelly_fraction: Fractional-Kelly multiplier (default 0.25 =
            quarter-Kelly, the standard variance dampener).
        shrinkage_bounds: ``(min, max)`` clamp on the model weight in the
            shrinkage blend. The conservative default caps at 0.50: a model
            that beats the market is treated as the market's *equal*, never
            its better, for sizing purposes.
        edge_gate_start: Edges at or below this are gated to zero.
        edge_gate_full: Edges at or above this pass at full size.
        log_loss_weight: Blend weight on log-loss skill vs Brier skill
            (log-loss is weighted higher by default because it punishes
            confident misses more harshly).
        fallback_trust_ramp: Without ``trust_weights``, trust ramps
            linearly from 0 at p=0 to 1.0 at this probability -- a crude
            longshot damper.

    Returns:
        Dict with ``stake``, ``stake_fraction``, and the diagnostic chain
        (``skill``, ``shrinkage_w``, ``p_adj``, ``edge``, ``edge_kelly``,
        ``edge_gate``, ``trust_weight``, ``kelly_fraction``).
    """
    # Default: assume market = model (skill = 0) when no baseline data
    if market_brier is None:
        market_brier = model_brier
    if market_log_loss is None:
        market_log_loss = model_log_loss

    # --- 1. Market-relative skill (blended Brier + log-loss) ---
    skill_bs = 1.0 - (model_brier / market_brier) if market_brier > 0 else 0.0

    if model_log_loss is not None and market_log_loss > 0:
        skill_ll = 1.0 - (model_log_loss / market_log_loss)
        skill = log_loss_weight * skill_ll + (1.0 - log_loss_weight) * skill_bs
    else:
        skill = skill_bs

    # --- 2. Shrink probability toward market ---
    w_min, w_max = shrinkage_bounds
    shrinkage_w = max(w_min, min(w_max, 0.50 + 2.0 * skill))
    p_adj = shrinkage_w * p_model + (1.0 - shrinkage_w) * p_market

    # --- 3. Kelly edge using the adjusted probability ---
    market_odds_decimal = 1.0 / p_market if p_market > 0 else 100.0
    if market_odds_decimal <= 1:
        edge_kelly = 0.0
    else:
        edge_kelly = (p_adj * market_odds_decimal - 1.0) / (market_odds_decimal - 1.0)

    # --- 4. Edge gating: ramp from 0 at the start threshold to full size ---
    edge = p_adj - p_market
    gate_span = edge_gate_full - edge_gate_start
    edge_gate = max(0.0, min(1.0, (abs(edge) - edge_gate_start) / gate_span))

    # --- 5. Empirical trust weight for this probability range ---
    if trust_weights:
        trust_weight = lookup_trust(trust_weights, p_adj)
    else:
        trust_weight = min(1.0, p_adj / fallback_trust_ramp)

    # --- 6. Final stake ---
    stake_fraction = max(0.0, edge_kelly * kelly_fraction * edge_gate * trust_weight)
    stake = bankroll * stake_fraction

    return {
        "stake": stake,
        "stake_fraction": stake_fraction,
        "kelly_fraction": kelly_fraction,
        "edge": edge,
        "edge_kelly": edge_kelly,
        "edge_gate": edge_gate,
        "trust_weight": trust_weight,
        "p_adj": p_adj,
        "skill": skill,
        "shrinkage_w": shrinkage_w,
    }
