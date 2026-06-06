import pytest

from pmt.staking import (
    StakeRecommendation,
    calculate_stake,
    compute_bin_trust_weights,
    lookup_trust,
    prob_to_american_odds,
)

# --- prob_to_american_odds ---------------------------------------------------


@pytest.mark.parametrize(
    "prob,odds",
    [(0.60, -150), (0.40, 150), (0.50, -100), (0.75, -300), (0.25, 300)],
)
def test_american_odds_conversion(prob, odds):
    assert prob_to_american_odds(prob) == odds


def test_american_odds_degenerate():
    assert prob_to_american_odds(0.0) == 0
    assert prob_to_american_odds(1.0) == -10000


# --- calculate_stake ----------------------------------------------------------


def test_no_edge_no_stake():
    """p_model == p_market -> zero edge -> gated to zero stake."""
    out = calculate_stake(0.55, 0.55, model_brier=0.21, bankroll=1000)
    assert out["edge"] == pytest.approx(0.0)
    assert out["edge_gate"] == 0.0
    assert out["stake"] == 0.0


def test_positive_edge_positive_stake():
    out = calculate_stake(
        0.65, 0.50, model_brier=0.20, bankroll=1000, market_brier=0.22
    )
    assert out["edge"] > 0
    assert out["stake"] > 0
    assert out["stake_fraction"] == pytest.approx(
        out["edge_kelly"] * 0.25 * out["edge_gate"] * out["trust_weight"]
    )


def test_zero_skill_when_no_market_baseline():
    """Without market baselines, skill defaults to 0 and the blend is 50/50."""
    out = calculate_stake(0.70, 0.50, model_brier=0.21, bankroll=1000)
    assert out["skill"] == pytest.approx(0.0)
    assert out["shrinkage_w"] == pytest.approx(0.50)
    assert out["p_adj"] == pytest.approx(0.60)


def test_shrinkage_cap_treats_better_model_as_equal():
    """Positive skill cannot push the model weight above the 0.50 cap."""
    out = calculate_stake(
        0.70, 0.50, model_brier=0.15, bankroll=1000, market_brier=0.25
    )
    assert out["skill"] > 0
    assert out["shrinkage_w"] == pytest.approx(0.50)


def test_bad_model_shrinks_toward_market():
    out = calculate_stake(
        0.70, 0.50, model_brier=0.28, bankroll=1000, market_brier=0.22
    )
    assert out["skill"] < 0
    assert out["shrinkage_w"] < 0.50
    assert out["shrinkage_w"] >= 0.10  # floor
    assert out["p_adj"] < 0.60  # pulled toward the market price


def test_log_loss_blends_into_skill():
    brier_only = calculate_stake(0.6, 0.5, model_brier=0.20, bankroll=1000,
                                 market_brier=0.22)
    blended = calculate_stake(0.6, 0.5, model_brier=0.20, bankroll=1000,
                              market_brier=0.22,
                              model_log_loss=0.55, market_log_loss=0.65)
    assert blended["skill"] != pytest.approx(brier_only["skill"])


def test_edge_gate_ramp():
    # Tiny edge (1% raw -> 0.5% after 50/50 shrinkage) -> fully gated
    tiny = calculate_stake(0.51, 0.50, model_brier=0.21, bankroll=1000)
    assert tiny["edge_gate"] == 0.0
    # Huge edge (20% raw -> 10% after shrinkage) -> full size
    big = calculate_stake(0.70, 0.50, model_brier=0.21, bankroll=1000)
    assert big["edge_gate"] == 1.0


def test_trust_weight_scales_stake():
    trust_half = [{"prob_low": 0, "prob_high": 100, "trust": 0.5}]
    trust_full = [{"prob_low": 0, "prob_high": 100, "trust": 1.0}]
    base = dict(model_brier=0.21, bankroll=1000)
    half = calculate_stake(0.65, 0.50, trust_weights=trust_half, **base)
    full = calculate_stake(0.65, 0.50, trust_weights=trust_full, **base)
    assert half["stake"] == pytest.approx(full["stake"] * 0.5)


def test_fallback_trust_ramps_longshots():
    """Without trust weights, low-probability positions are damped."""
    longshot = calculate_stake(0.15, 0.08, model_brier=0.21, bankroll=1000)
    assert longshot["trust_weight"] < 0.5
    favorite = calculate_stake(0.65, 0.50, model_brier=0.21, bankroll=1000)
    assert favorite["trust_weight"] == 1.0


def test_kelly_fraction_scales_linearly():
    base = dict(model_brier=0.21, bankroll=1000)
    quarter = calculate_stake(0.65, 0.50, **base)
    half = calculate_stake(0.65, 0.50, kelly_fraction=0.5, **base)
    assert half["stake"] == pytest.approx(quarter["stake"] * 2)


# --- bin trust ----------------------------------------------------------------


def _bins(rois_counts):
    return [
        {"low": i * 10, "high": (i + 1) * 10, "roi": roi, "count": count}
        for i, (roi, count) in enumerate(rois_counts)
    ]


def test_raw_trust_roi_mapping():
    out = compute_bin_trust_weights(
        _bins([(50.0, 1000)]), smooth_neighbors=0, min_samples=0
    )
    assert out[0]["raw_trust"] == pytest.approx(1.5)
    assert out[0]["trust"] == pytest.approx(1.5)


def test_raw_trust_clamped():
    out = compute_bin_trust_weights(
        _bins([(-100.0, 1000), (200.0, 1000)]), smooth_neighbors=0, min_samples=0
    )
    assert out[0]["raw_trust"] == pytest.approx(0.1)  # floor
    assert out[1]["raw_trust"] == pytest.approx(1.5)  # ceiling


def test_shrinkage_blends_toward_default():
    # count == min_samples -> 50/50 blend between raw (1.4) and default (1.0)
    out = compute_bin_trust_weights(
        _bins([(40.0, 10)]), min_samples=10, smooth_neighbors=0
    )
    assert out[0]["shrunk"] == pytest.approx(1.2)


def test_empty_bin_gets_default_trust():
    out = compute_bin_trust_weights(_bins([(0.0, 0)]), smooth_neighbors=0)
    assert out[0]["trust"] == pytest.approx(1.0)


def test_neighbor_smoothing_damps_outlier():
    # Middle bin is a -80% ROI outlier between two healthy bins
    no_smooth = compute_bin_trust_weights(
        _bins([(10.0, 200), (-80.0, 200), (10.0, 200)]), smooth_neighbors=0
    )
    smoothed = compute_bin_trust_weights(
        _bins([(10.0, 200), (-80.0, 200), (10.0, 200)]), smooth_neighbors=1
    )
    assert smoothed[1]["trust"] > no_smooth[1]["trust"]


def test_empty_input():
    assert compute_bin_trust_weights([]) == []


def test_lookup_trust():
    weights = [
        {"prob_low": 0, "prob_high": 50, "trust": 0.8},
        {"prob_low": 50, "prob_high": 100, "trust": 1.2},
    ]
    assert lookup_trust(weights, 0.25) == pytest.approx(0.8)
    assert lookup_trust(weights, 0.75) == pytest.approx(1.2)
    assert lookup_trust(weights, 1.0) == pytest.approx(1.2)  # top bin inclusive
    assert lookup_trust([], 0.5) == 1.0


# --- StakeRecommendation -------------------------------------------------------


def test_stake_recommendation_roundtrip():
    sizing = calculate_stake(0.65, 0.50, model_brier=0.20, bankroll=1000,
                             market_brier=0.22)
    rec = StakeRecommendation.from_sizing("evt-001", "HOME", 0.65, 0.50, sizing)
    d = rec.to_dict()
    assert d["event_id"] == "evt-001"
    assert d["model_odds"] == prob_to_american_odds(0.65)
    assert d["stake"] == pytest.approx(sizing["stake"])
