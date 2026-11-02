"""Classic Elo (FiveThirtyEight-style) — the simple baseline every other model must beat.

Ratings move after each game by K * margin-of-victory multiplier * surprise.
Between seasons every team is pulled 1/3 of the way back to the average (1505).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .config import ELO_K, ELO_HOME_ADV, ELO_SEASON_REGRESSION

MEAN = 1505.0


def win_prob(elo_diff: float | np.ndarray) -> float | np.ndarray:
    """P(team A wins) given A's Elo minus B's Elo (home edge already included)."""
    return 1.0 / (1.0 + 10 ** (-np.asarray(elo_diff) / 400.0))


def run_elo(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return pregame Elo for both teams in every game (future games get current ratings).

    Only results from earlier rows are used, so these are safe to use as features.
    """
    ratings: dict[str, float] = {}
    season_of_rating: dict[str, int] = {}
    out = []
    for g in schedule.itertuples(index=False):
        for t in (g.home_team, g.away_team):
            r = ratings.get(t, MEAN)
            if t in season_of_rating and season_of_rating[t] != g.season:
                r = MEAN + (1 - ELO_SEASON_REGRESSION) * (r - MEAN)   # new season: regress
            ratings[t], season_of_rating[t] = r, g.season
        h, a = ratings[g.home_team], ratings[g.away_team]
        out.append((g.game_id, h, a))

        if pd.isna(g.home_score):  # not played yet -> no update
            continue
        hfa = 0 if g.location == "Neutral" else ELO_HOME_ADV
        diff = h + hfa - a
        margin = g.home_score - g.away_score
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = win_prob(diff)
        # Margin-of-victory multiplier (dampened for favorites to curb autocorrelation)
        winner_diff = diff if margin > 0 else -diff
        mov = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2) if margin != 0 else 1.0
        shift = ELO_K * mov * (actual - expected)
        ratings[g.home_team] += shift
        ratings[g.away_team] -= shift
    return pd.DataFrame(out, columns=["game_id", "home_elo", "away_elo"])
