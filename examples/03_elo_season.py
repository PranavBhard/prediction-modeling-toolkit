"""Elo convergence and season carryover on a simulated league.

Simulates two seasons of a 12-team league with latent team strengths
(shuffled slightly between seasons), runs the Elo engine with a dynamic K
schedule and carryover regression, and shows:

1. Rating trajectories converging toward latent strength, with the
   carryover regression visibly pulling everyone toward the mean at the
   season boundary.
2. Final ratings vs. latent strengths (the engine recovers the truth).

Run:
    python examples/03_elo_season.py
"""

import numpy as np

from pmt.ratings import Elo

rng = np.random.default_rng(11)

N_TEAMS = 12
ROUNDS_PER_SEASON = 3  # 3 x double round-robin = 66 games/team/season
teams = [f"T{i:02d}" for i in range(N_TEAMS)]


def simulate_season(strength, season, start_day=0):
    games, day = [], start_day
    for _ in range(ROUNDS_PER_SEASON):
        order = rng.permutation([(i, j) for i in range(N_TEAMS)
                                 for j in range(N_TEAMS) if i != j])
        for i, j in order:
            day += 1
            p_home = 1 / (1 + 10 ** ((strength[j] - strength[i] - 60) / 400))
            games.append({
                "home": teams[i], "away": teams[j],
                "home_won": bool(rng.random() < p_home),
                "date": f"day-{day:05d}", "season": season,
            })
    return games, day


strength_s1 = rng.normal(1500, 130, N_TEAMS)
# Season 2: strengths drift (roster turnover)
strength_s2 = 1500 + 0.7 * (strength_s1 - 1500) + rng.normal(0, 60, N_TEAMS)

games_s1, day = simulate_season(strength_s1, "S1")
games_s2, _ = simulate_season(strength_s2, "S2", start_day=day)
games = games_s1 + games_s2

elo = Elo(
    home_advantage=60,
    k_schedule=[{"max_games": 20, "k": 40}, {"max_games": 50, "k": 24}, {"default": 12}],
    carryover_alpha=0.7,
)
result = elo.compute(games)

final = np.array([result.current[t] for t in teams])
corr = np.corrcoef(final, strength_s2)[0, 1]
print(f"{result.n_games} games over 2 seasons, {N_TEAMS} teams")
print(f"correlation(final Elo, latent S2 strength) = {corr:.3f}")
print(f"\n{'team':<6}{'latent S2':>11}{'final elo':>11}")
print("-" * 28)
for idx in np.argsort(-strength_s2):
    print(f"{teams[idx]:<6}{strength_s2[idx]:>11.0f}{final[idx]:>11.0f}")

# --- Chart -----------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed; skipping chart")
    raise SystemExit(0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: trajectories for the strongest, weakest, and median teams
order = np.argsort(strength_s1)
picks = [order[-1], order[len(order) // 2], order[0]]
colors = ["tab:green", "tab:gray", "tab:red"]
labels = ["strongest (S1)", "median (S1)", "weakest (S1)"]

# Build per-team trajectories in game order
season_break = None
for pick, color, label in zip(picks, colors, labels):
    team = teams[pick]
    points = sorted(
        ((d, v) for (t, d, s), v in result.history.items() if t == team),
        key=lambda x: x[0],
    )
    ys = [v for _, v in points]
    ax1.plot(range(len(ys)), ys, color=color, lw=1.6, label=label)
    if season_break is None:
        s1_count = sum(1 for (t, d, s) in result.history if t == team and s == "S1")
        season_break = s1_count

ax1.axvline(season_break - 0.5, color="black", lw=1, ls=":")
ax1.annotate("season boundary\n(carryover regression α=0.7)",
             xy=(season_break - 0.5, ax1.get_ylim()[1]),
             xytext=(8, -10), textcoords="offset points", fontsize=9, va="top")
ax1.axhline(1500, color="gray", lw=0.8, ls="--")
ax1.set_xlabel("game number")
ax1.set_ylabel("pre-game Elo")
ax1.set_title("Rating trajectories with dynamic K + season carryover")
ax1.legend(loc="lower left")

# Panel 2: final rating vs latent strength
ax2.scatter(strength_s2, final, s=60, alpha=0.8, edgecolors="navy")
lims = [min(strength_s2.min(), final.min()) - 30, max(strength_s2.max(), final.max()) + 30]
ax2.plot(lims, lims, "k--", lw=1, label="perfect recovery")
for idx in range(N_TEAMS):
    ax2.annotate(teams[idx], (strength_s2[idx], final[idx]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
ax2.set_xlabel("latent strength (season 2 truth)")
ax2.set_ylabel("final Elo rating")
ax2.set_title(f"Engine recovers latent strength (r = {corr:.3f})")
ax2.legend()

fig.tight_layout()
out = "docs/gallery/ratings_elo_convergence_carryover.png"
fig.savefig(out, dpi=150)
print(f"\nchart written to {out}")
