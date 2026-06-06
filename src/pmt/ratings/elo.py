"""Configurable Elo rating engine.

A pure-computation Elo implementation supporting the refinements that
matter in practice for team-sport rating systems:

- **Dynamic K schedules** -- larger updates early in a season when ratings
  are uncertain, smaller once established.
- **Season carryover regression** -- ratings regress toward a mean at
  season boundaries (roster turnover erodes last season's information).
- **Margin-of-victory adjustment** -- FiveThirtyEight-style capped
  multiplier that discounts blowouts by favorites to prevent
  autocorrelation and rating inflation.
- **Home advantage** -- league default, per-game-type overrides, neutral
  sites, and optionally per-team empirical home advantage learned from
  each team's home over/underperformance.
- **Glicko-style rating deviation (RD)** -- per-team uncertainty that
  shrinks with games played, inflates at season boundaries, and scales
  K down when the opponent's rating is itself uncertain.

No storage, no I/O: feed it chronological game dicts, get histories back.
Caching is the caller's problem.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

__all__ = ["Elo", "EloResult"]

TeamDateSeason = Tuple[str, str, str]


@dataclass
class EloResult:
    """Output of :meth:`Elo.compute`.

    Attributes:
        history: ``(team, date, season) -> pre-game Elo`` for every game
            appearance. Pre-game means the rating *going into* that game --
            the value a prediction made before the game would have used.
        current: ``team -> rating`` after the final game.
        rd_history: ``(team, date, season) -> pre-game RD`` (only when RD
            tracking is enabled).
        current_rd: ``team -> RD`` after the final game (only when RD
            tracking is enabled).
        home_advantages: ``team -> blended empirical home advantage`` (only
            when per-team home advantage is enabled; teams reaching the
            minimum home-game threshold only).
        n_games: number of valid games processed.
        n_skipped: games dropped for missing required fields.
    """

    history: Dict[TeamDateSeason, float] = field(default_factory=dict)
    current: Dict[str, float] = field(default_factory=dict)
    rd_history: Optional[Dict[TeamDateSeason, float]] = None
    current_rd: Optional[Dict[str, float]] = None
    home_advantages: Optional[Dict[str, float]] = None
    n_games: int = 0
    n_skipped: int = 0


class Elo:
    """Elo rating engine over chronological game records.

    Each game is a flat dict with keys:

    - ``home``, ``away``: team identifiers
    - ``date``: sortable date string (e.g. ISO ``"2024-01-15"``)
    - ``season``: season identifier (e.g. ``"2023-2024"``)
    - ``home_won``: bool
    - ``home_score``, ``away_score``: optional ints (needed for
      margin-of-victory adjustment)
    - ``neutral_site``: optional bool
    - ``game_type``: optional str (matched against
      ``home_advantage_overrides``)

    Args:
        starting_elo: Initial rating for unseen teams.
        k_factor: Static K factor (used when no ``k_schedule`` matches).
        home_advantage: Home advantage in Elo points.
        k_schedule: Optional dynamic K schedule -- a list like
            ``[{"max_games": 10, "k": 32}, {"max_games": 30, "k": 24},
            {"default": 16}]``. ``max_games`` is an exclusive upper bound
            on the team's *pre-game* count of games played this season.
            Per game, the two teams' schedule-Ks are averaged so updates
            stay zero-sum.
        carryover_alpha: When set, season boundaries regress every rating:
            ``new = mean + alpha * (old - mean)``. ``None`` disables
            carryover (each team keeps its rating across seasons).
        carryover_mean: Regression target (defaults to ``starting_elo``).
        neutral_site_home_advantage: Home advantage on neutral sites
            (usually 0).
        home_advantage_overrides: ``{game_type: ha}`` overrides (e.g.
            reduced advantage in postseason bubbles).
        mov_method: ``None`` (no margin adjustment), ``"elo_mov_capped"``
            (FiveThirtyEight-style capped multiplier), or
            ``"log_score_diff"`` (simple ``ln(1 + diff)``).
        mov_cap: Cap on the capped MOV multiplier.
        per_team_home_advantage: Learn each team's empirical home
            advantage from its home over/underperformance, blended with
            the league default.
        per_team_ha_blend: Weight on the empirical estimate in the blend.
        per_team_ha_min_games: Home games required before the per-team
            estimate is used.
        track_rd: Enable Glicko-style rating deviation tracking.
        rd_start: Initial RD for unseen teams (high = very uncertain).
        rd_min: RD floor -- even established teams keep some uncertainty.
        rd_decay_per_game: RD reduction per game played.
        rd_season_inflation: RD added back at season boundaries
            (roster turnover = renewed uncertainty), capped at ``rd_start``.
    """

    def __init__(
        self,
        *,
        starting_elo: float = 1500.0,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        k_schedule: Optional[List[dict]] = None,
        carryover_alpha: Optional[float] = None,
        carryover_mean: Optional[float] = None,
        neutral_site_home_advantage: float = 0.0,
        home_advantage_overrides: Optional[Dict[str, float]] = None,
        mov_method: Optional[str] = None,
        mov_cap: float = 3.0,
        per_team_home_advantage: bool = False,
        per_team_ha_blend: float = 0.5,
        per_team_ha_min_games: int = 10,
        track_rd: bool = False,
        rd_start: float = 350.0,
        rd_min: float = 30.0,
        rd_decay_per_game: float = 15.0,
        rd_season_inflation: float = 50.0,
    ):
        if mov_method not in (None, "elo_mov_capped", "log_score_diff"):
            raise ValueError(f"Unknown mov_method: {mov_method!r}")
        self.starting_elo = starting_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.k_schedule = k_schedule or []
        self.carryover_alpha = carryover_alpha
        self.carryover_mean = carryover_mean if carryover_mean is not None else starting_elo
        self.neutral_site_ha = neutral_site_home_advantage
        self.ha_overrides = home_advantage_overrides or {}
        self.mov_method = mov_method
        self.mov_cap = mov_cap
        self.per_team_ha_enabled = per_team_home_advantage
        self.per_team_ha_blend = per_team_ha_blend
        self.per_team_ha_min_games = per_team_ha_min_games
        self.track_rd = track_rd
        self.rd_start = rd_start
        self.rd_min = rd_min
        self.rd_decay = rd_decay_per_game
        self.rd_season_inflation = rd_season_inflation

    # --- Components (each individually testable) -----------------------------

    @staticmethod
    def expected_home_prob(home_elo: float, away_elo: float, home_advantage: float = 0.0) -> float:
        """Logistic expectation: P(home win) given ratings and home advantage."""
        return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400.0))

    def k_for_games_played(self, games_played: int) -> float:
        """Look up K from the schedule based on the pre-game count.

        ``max_games`` is an exclusive upper bound: ``max_games=10`` covers
        pre-game counts 0..9 (the team's first 10 games).
        """
        if not self.k_schedule:
            return self.k_factor
        for entry in self.k_schedule:
            if "max_games" in entry:
                if "k" not in entry:
                    raise ValueError(f"Malformed k_schedule entry (missing 'k'): {entry}")
                if games_played < entry["max_games"]:
                    return float(entry["k"])
        for entry in self.k_schedule:
            if "default" in entry:
                return float(entry["default"])
        raise ValueError(
            f"k_schedule has no matching tier for games_played={games_played} "
            f"and no default entry: {self.k_schedule}"
        )

    def mov_multiplier(self, score_diff: int, elo_diff: float) -> float:
        """FiveThirtyEight-style capped margin-of-victory multiplier.

        ``ln(1 + |score_diff|) * (2.2 / (elo_diff * 0.001 + 2.2))``, capped
        at ``mov_cap``. ``elo_diff`` is winner-minus-loser, so blowouts *by
        the favorite* are discounted (preventing autocorrelation and rating
        inflation) while upsets by margin count nearly in full.
        """
        if score_diff <= 0:
            return 1.0
        raw = math.log(1 + score_diff) * (2.2 / (elo_diff * 0.001 + 2.2))
        return min(raw, self.mov_cap)

    @staticmethod
    def _g(rd_value: float) -> float:
        """Glicko g-function: discounts result information vs opponent RD."""
        q = math.log(10) / 400.0
        return 1.0 / math.sqrt(1 + 3 * q**2 * rd_value**2 / math.pi**2)

    @staticmethod
    def _valid(game: dict) -> bool:
        return all(
            [
                game.get("season"),
                game.get("home"),
                game.get("away"),
                game.get("date"),
                "home_won" in game,
            ]
        )

    # --- Main loop -------------------------------------------------------------

    def compute(
        self,
        games: List[dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EloResult:
        """Run the rating loop over games (sorted by season, then date).

        Pure computation: the loop holds all state internally and the same
        input always produces the same output.
        """
        elo: Dict[str, float] = defaultdict(lambda: self.starting_elo)
        rd: Optional[Dict[str, float]] = (
            defaultdict(lambda: self.rd_start) if self.track_rd else None
        )
        games_played: Dict[str, int] = defaultdict(int)
        current_season: Optional[str] = None

        # Per-team home advantage tracking: residual = actual - neutral expectation
        team_home_margins = defaultdict(list) if self.per_team_ha_enabled else None
        team_home_adv: Dict[str, float] = {}

        result = EloResult(
            rd_history={} if self.track_rd else None,
            current_rd={} if self.track_rd else None,
            home_advantages={} if self.per_team_ha_enabled else None,
        )

        valid_games = [g for g in games if self._valid(g)]
        result.n_skipped = len(games) - len(valid_games)
        sorted_games = sorted(valid_games, key=lambda g: (g["season"], g["date"]))
        total = len(sorted_games)
        result.n_games = total

        for idx, game in enumerate(sorted_games):
            season = game["season"]

            # Season boundary: carryover regression, RD inflation, counters reset
            if season != current_season:
                if current_season is not None and self.carryover_alpha is not None:
                    for team in list(elo.keys()):
                        elo[team] = self.carryover_mean + self.carryover_alpha * (
                            elo[team] - self.carryover_mean
                        )
                    if rd is not None:
                        for team in list(rd.keys()):
                            rd[team] = min(rd[team] + self.rd_season_inflation, self.rd_start)
                if team_home_margins is not None:
                    team_home_margins.clear()
                    team_home_adv.clear()
                games_played.clear()
                current_season = season

            home, away, game_date = game["home"], game["away"], game["date"]

            # Record pre-game state
            result.history[(home, game_date, season)] = elo[home]
            result.history[(away, game_date, season)] = elo[away]
            if rd is not None:
                result.rd_history[(home, game_date, season)] = rd[home]
                result.rd_history[(away, game_date, season)] = rd[away]

            # Resolve home advantage: neutral > game_type override > per-team > default
            is_neutral = game.get("neutral_site", False)
            game_type = game.get("game_type", "")
            if is_neutral:
                ha = self.neutral_site_ha
            elif game_type in self.ha_overrides:
                ha = self.ha_overrides[game_type]
            elif self.per_team_ha_enabled and home in team_home_adv:
                ha = team_home_adv[home]
            else:
                ha = self.home_advantage

            expected_home = self.expected_home_prob(elo[home], elo[away], ha)

            # Dynamic K: average both teams' schedule-K (preserves zero-sum)
            k = (
                self.k_for_games_played(games_played[home])
                + self.k_for_games_played(games_played[away])
            ) / 2

            # RD scaling: a high-RD opponent means the result taught us less
            if rd is not None:
                k *= (self._g(rd[away]) + self._g(rd[home])) / 2

            actual_home = 1.0 if game["home_won"] else 0.0
            elo_change = k * (actual_home - expected_home)

            # Margin-of-victory adjustment
            if self.mov_method is not None:
                score_diff = abs(
                    int(game.get("home_score", 0)) - int(game.get("away_score", 0))
                )
                if self.mov_method == "elo_mov_capped":
                    winner = elo[home] if game["home_won"] else elo[away]
                    loser = elo[away] if game["home_won"] else elo[home]
                    elo_change *= self.mov_multiplier(score_diff, winner - loser)
                elif self.mov_method == "log_score_diff" and score_diff > 0:
                    elo_change *= math.log(1 + score_diff)

            elo[home] += elo_change
            elo[away] -= elo_change

            if rd is not None:
                rd[home] = max(rd[home] - self.rd_decay, self.rd_min)
                rd[away] = max(rd[away] - self.rd_decay, self.rd_min)

            # Per-team home advantage: how far did the home team beat the
            # *neutral* expectation, on average?
            if team_home_margins is not None and not is_neutral:
                neutral_expected = self.expected_home_prob(elo[home], elo[away], 0.0)
                team_home_margins[home].append(actual_home - neutral_expected)
                n_home = len(team_home_margins[home])
                if n_home >= self.per_team_ha_min_games:
                    avg_margin = sum(team_home_margins[home]) / n_home
                    empirical_ha = avg_margin * 400.0  # margin residual -> Elo points
                    team_home_adv[home] = (
                        self.per_team_ha_blend * empirical_ha
                        + (1 - self.per_team_ha_blend) * self.home_advantage
                    )

            # Increment AFTER the update -- counts are pre-game
            games_played[home] += 1
            games_played[away] += 1

            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total)

        if progress_callback:
            progress_callback(total, total)

        result.current = dict(elo)
        if rd is not None:
            result.current_rd = dict(rd)
        if team_home_adv:
            result.home_advantages = dict(team_home_adv)
        return result
