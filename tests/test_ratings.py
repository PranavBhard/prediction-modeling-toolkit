import numpy as np
import pytest

from pmt.ratings import Elo


def game(home, away, home_won, date, season="S1", **kw):
    return {"home": home, "away": away, "home_won": home_won,
            "date": date, "season": season, **kw}


# --- components ----------------------------------------------------------------


def test_expected_prob_symmetric():
    assert Elo.expected_home_prob(1500, 1500, 0) == pytest.approx(0.5)


def test_expected_prob_400_points():
    """+400 Elo should mean ~10:1, i.e. ~0.909."""
    assert Elo.expected_home_prob(1900, 1500, 0) == pytest.approx(10 / 11, abs=1e-9)


def test_expected_prob_home_advantage_shifts():
    assert Elo.expected_home_prob(1500, 1500, 100) > 0.5


def test_k_schedule_tiers():
    elo = Elo(k_schedule=[{"max_games": 10, "k": 32},
                          {"max_games": 30, "k": 24},
                          {"default": 16}])
    assert elo.k_for_games_played(0) == 32
    assert elo.k_for_games_played(9) == 32
    assert elo.k_for_games_played(10) == 24
    assert elo.k_for_games_played(29) == 24
    assert elo.k_for_games_played(30) == 16


def test_k_schedule_no_default_raises():
    elo = Elo(k_schedule=[{"max_games": 10, "k": 32}])
    with pytest.raises(ValueError, match="no matching tier"):
        elo.k_for_games_played(50)


def test_k_schedule_malformed_raises():
    elo = Elo(k_schedule=[{"max_games": 10}])
    with pytest.raises(ValueError, match="Malformed"):
        elo.k_for_games_played(0)


def test_no_schedule_uses_static_k():
    assert Elo(k_factor=24).k_for_games_played(100) == 24


def test_mov_multiplier_discount_favorites():
    elo = Elo()
    assert elo.mov_multiplier(0, 0) == 1.0
    # Same margin: favorite's blowout (positive elo_diff) counts for less
    # than the underdog's upset blowout (negative elo_diff)
    assert elo.mov_multiplier(20, 400) < elo.mov_multiplier(20, -400)


def test_mov_multiplier_capped():
    elo = Elo(mov_cap=3.0)
    assert elo.mov_multiplier(10_000, -400) == 3.0


def test_unknown_mov_method_raises():
    with pytest.raises(ValueError, match="Unknown mov_method"):
        Elo(mov_method="run_diff_squared")


# --- compute loop ----------------------------------------------------------------


def test_winner_gains_loser_drops():
    result = Elo(home_advantage=0).compute([game("A", "B", True, "2024-01-01")])
    assert result.current["A"] > 1500 > result.current["B"]


def test_zero_sum():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    games = [
        game(teams[rng.integers(8)], teams[(rng.integers(7) + 1) % 8],
             bool(rng.random() < 0.5), f"2024-01-{d+1:02d}")
        for d in range(28)
    ]
    games = [g for g in games if g["home"] != g["away"]]
    result = Elo().compute(games)
    total = sum(result.current.values())
    assert total == pytest.approx(1500 * len(result.current))


def test_home_win_worth_less_than_road_win():
    """With home advantage, the expected-prob asymmetry makes a home win
    smaller than a road win for the same matchup."""
    home_win = Elo(home_advantage=100).compute([game("A", "B", True, "2024-01-01")])
    road_win = Elo(home_advantage=100).compute([game("B", "A", False, "2024-01-01")])
    gain_home = home_win.current["A"] - 1500
    gain_road = road_win.current["A"] - 1500
    assert gain_road > gain_home


def test_history_stores_pregame_ratings():
    result = Elo(home_advantage=0).compute([
        game("A", "B", True, "2024-01-01"),
        game("A", "B", True, "2024-01-02"),
    ])
    # First appearance: the untouched starting rating
    assert result.history[("A", "2024-01-01", "S1")] == 1500
    # Second appearance: above 1500 (won game 1) but below the final rating
    # (game 2's win hasn't been applied yet at pre-game time)
    pre_game2 = result.history[("A", "2024-01-02", "S1")]
    assert 1500 < pre_game2 < result.current["A"]


def test_carryover_regression():
    """Pre-game rating of the first S2 game = mean + alpha * (end-of-S1 - mean)."""
    s1_games = [game("A", "B", True, f"2024-01-{d+1:02d}") for d in range(10)]

    # Reference run with no boundary: A's rating after the 10 S1 games
    end_s1 = Elo(home_advantage=0, k_factor=40).compute(s1_games).current["A"]

    result = Elo(home_advantage=0, k_factor=40, carryover_alpha=0.5).compute(
        s1_games + [game("A", "B", True, "2024-10-01", season="S2")]
    )
    a_pre_s2 = result.history[("A", "2024-10-01", "S2")]
    assert a_pre_s2 == pytest.approx(1500 + 0.5 * (end_s1 - 1500))
    assert a_pre_s2 < end_s1  # excursion halved


def test_no_carryover_keeps_ratings():
    s1 = [game("A", "B", True, "2024-01-01")]
    end_s1 = Elo(home_advantage=0).compute(s1).current["A"]
    result = Elo(home_advantage=0, carryover_alpha=None).compute(
        s1 + [game("A", "B", True, "2024-10-01", season="S2")]
    )
    assert result.history[("A", "2024-10-01", "S2")] == pytest.approx(end_s1)


def test_neutral_site_no_home_edge():
    neutral = Elo(home_advantage=100).compute(
        [game("A", "B", True, "2024-01-01", neutral_site=True)]
    )
    home = Elo(home_advantage=100).compute([game("A", "B", True, "2024-01-01")])
    # Neutral-site win: expectation was 0.5, so the gain is larger than at home
    assert neutral.current["A"] - 1500 > home.current["A"] - 1500


def test_game_type_ha_override():
    override = Elo(home_advantage=100, home_advantage_overrides={"postseason": 0})
    post = override.compute([game("A", "B", True, "2024-06-01", game_type="postseason")])
    regular = override.compute([game("A", "B", True, "2024-06-01")])
    assert post.current["A"] > regular.current["A"]


def test_mov_scales_update():
    base = dict(home_advantage=0)
    plain = Elo(**base).compute(
        [game("A", "B", True, "2024-01-01", home_score=10, away_score=0)]
    )
    mov = Elo(mov_method="elo_mov_capped", **base).compute(
        [game("A", "B", True, "2024-01-01", home_score=10, away_score=0)]
    )
    assert mov.current["A"] > plain.current["A"]


def test_invalid_games_skipped():
    result = Elo().compute([
        game("A", "B", True, "2024-01-01"),
        {"home": "A", "away": "B"},  # missing date/season/home_won
    ])
    assert result.n_games == 1
    assert result.n_skipped == 1


# --- rating deviation -------------------------------------------------------------


def test_rd_decays_with_games():
    elo = Elo(track_rd=True, rd_start=350, rd_decay_per_game=15, rd_min=30)
    games = [game("A", "B", bool(i % 2), f"2024-01-{i+1:02d}") for i in range(25)]
    result = elo.compute(games)
    assert result.current_rd["A"] == 30  # decayed to the floor
    assert result.rd_history[("A", "2024-01-01", "S1")] == 350


def test_rd_inflates_at_season_boundary():
    elo = Elo(track_rd=True, carryover_alpha=0.75, rd_decay_per_game=15,
              rd_season_inflation=50, home_advantage=0)
    games = [game("A", "B", True, f"2024-01-{d+1:02d}") for d in range(10)]
    games.append(game("A", "B", True, "2024-10-01", season="S2"))
    result = elo.compute(games)
    rd_end_s1 = 350 - 10 * 15  # 200
    assert result.rd_history[("A", "2024-10-01", "S2")] == pytest.approx(rd_end_s1 + 50)


def test_high_rd_opponent_shrinks_update():
    """Beating an uncertain opponent teaches us less -> smaller K."""
    certain = Elo(home_advantage=0).compute([game("A", "B", True, "2024-01-01")])
    uncertain = Elo(track_rd=True, home_advantage=0, rd_start=350).compute(
        [game("A", "B", True, "2024-01-01")]
    )
    assert uncertain.current["A"] - 1500 < certain.current["A"] - 1500


# --- per-team home advantage --------------------------------------------------------


def test_per_team_home_advantage_learned():
    """A team that keeps winning at home as a rating-equal should earn an
    empirical HA different from the league default."""
    elo = Elo(home_advantage=50, per_team_home_advantage=True,
              per_team_ha_min_games=5, per_team_ha_blend=0.5)
    games = []
    # A wins every home game vs rotating opponents
    for d in range(12):
        games.append(game("A", f"OPP{d}", True, f"2024-01-{d+1:02d}"))
    result = elo.compute(games)
    assert result.home_advantages is not None
    assert "A" in result.home_advantages
    assert result.home_advantages["A"] > 50  # overperformance -> above default


# --- convergence ---------------------------------------------------------------------


def test_ratings_recover_latent_strength():
    """Round-robin convergence: ratings should correlate strongly with the
    latent strengths that generated the results, and a dynamic K schedule
    (fast early learning, stable late) should converge better than static K."""
    rng = np.random.default_rng(7)
    n_teams = 12
    strength = rng.normal(1500, 120, n_teams)
    teams = [f"T{i:02d}" for i in range(n_teams)]

    games = []
    day = 0
    for _ in range(4):  # 4 full round-robins
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                day += 1
                p_home = 1 / (1 + 10 ** ((strength[j] - strength[i]) / 400))
                games.append(game(teams[i], teams[j],
                                  bool(rng.random() < p_home),
                                  f"2024-{day // 28 + 1:02d}-{day % 28 + 1:02d}"))

    def corr(result):
        final = np.array([result.current[t] for t in teams])
        return np.corrcoef(final, strength)[0, 1]

    static = corr(Elo(home_advantage=0, k_factor=24).compute(games))
    dynamic = corr(Elo(home_advantage=0, k_schedule=[
        {"max_games": 20, "k": 40},
        {"max_games": 50, "k": 24},
        {"default": 12},
    ]).compute(games))

    assert static > 0.85
    assert dynamic > static  # the schedule earns its keep
