"""Team rating systems.

A configurable Elo engine with dynamic K schedules, season carryover
regression, capped margin-of-victory adjustment, per-team home advantage,
and Glicko-style rating deviation::

    from pmt.ratings import Elo

    elo = Elo(k_factor=24, carryover_alpha=0.75, mov_method="elo_mov_capped")
    result = elo.compute(games)          # chronological flat game dicts
    result.current["Team A"]             # rating after the last game
    result.history[("Team A", "2024-01-15", "2023-2024")]  # pre-game rating
"""

from pmt.ratings.elo import Elo, EloResult

__all__ = ["Elo", "EloResult"]
