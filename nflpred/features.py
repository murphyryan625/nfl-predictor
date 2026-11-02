"""Pregame features for every game, built only from information available before kickoff.

Leakage rules this file follows:
  * Team ratings for a game use only games played on EARLIER dates.
  * QB ratings use only games before the game's date.
  * Elo ratings are pregame values.
  * Weather comes from the schedule (game-time conditions). Live predictions use a forecast instead.

Every feature is oriented home-minus-away, and the target is home margin (home pts - away pts).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .config import RECENCY_HALF_LIFE_GAMES, PRIOR_GAMES_WEIGHT, PRIOR_REGRESSION
from .elo import run_elo
from .qb import QBRatings

# Per-game stats we turn into ratings. "_def" values are what the defense ALLOWED (lower = better).
STATS = ["epa_off", "epa_def", "pass_epa_off", "pass_epa_def", "rush_epa_off", "rush_epa_def",
         "success_off", "success_def", "pass_success_off", "pass_success_def",
         "rush_success_off", "rush_success_def", "qb_played"]

# Home stadium coordinates (lat, lon). Relocated franchises handled in `team_location`.
STADIUMS = {
    "ARI": (33.528, -112.263), "ATL": (33.755, -84.401), "BAL": (39.278, -76.623), "BUF": (42.774, -78.787),
    "CAR": (35.226, -80.853), "CHI": (41.862, -87.617), "CIN": (39.095, -84.516), "CLE": (41.506, -81.700),
    "DAL": (32.748, -97.093), "DEN": (39.744, -105.020), "DET": (42.340, -83.046), "GB": (44.501, -88.062),
    "HOU": (29.685, -95.411), "IND": (39.760, -86.164), "JAX": (30.324, -81.637), "KC": (39.049, -94.484),
    "LA": (33.953, -118.339), "LAC": (33.953, -118.339), "LV": (36.091, -115.184), "MIA": (25.958, -80.239),
    "MIN": (44.974, -93.258), "NE": (42.091, -71.264), "NO": (29.951, -90.081), "NYG": (40.814, -74.074),
    "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016), "SEA": (47.595, -122.332),
    "SF": (37.403, -121.970), "TB": (27.976, -82.503), "TEN": (36.166, -86.771), "WAS": (38.908, -76.865),
}
OLD_HOMES = {("LA", 2016): (38.633, -90.188),    # St. Louis before 2016
             ("LAC", 2017): (32.783, -117.120),  # San Diego before 2017
             ("LV", 2020): (37.752, -122.201)}   # Oakland before 2020


def team_location(team: str, season: int) -> tuple[float, float]:
    for (t, moved_in), coords in OLD_HOMES.items():
        if team == t and season < moved_in:
            return coords
    return STADIUMS.get(team, (39.8, -98.6))


def miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(h))


# --------------------------------------------------------------------------
# Team ratings: recency-weighted in-season average blended with last season
# --------------------------------------------------------------------------
def team_ratings(schedule: pd.DataFrame, team_games: pd.DataFrame, qbr: QBRatings) -> pd.DataFrame:
    """Pregame rating of every team in every scheduled game (past and future).

    rating = (w_prior * prior + sum(w_i * game_i)) / (w_prior + sum(w_i))
      w_i     = 0.5 ** (games_ago / HALF_LIFE)       -> recent games count more
      w_prior = PRIOR_GAMES_WEIGHT, fading the same way as the season goes on
      prior   = league_mean + PRIOR_REGRESSION * (last season's final rating - league_mean)
    """
    tg = team_games.copy()
    # Add the rating of the QB who started each game as one more "stat" (used for QB-change feature)
    starters = pd.concat([
        schedule[["game_id", "home_team", "home_qb_id", "gameday"]].set_axis(["game_id", "team", "qb_id", "gameday"], axis=1),
        schedule[["game_id", "away_team", "away_qb_id", "gameday"]].set_axis(["game_id", "team", "qb_id", "gameday"], axis=1)])
    starters["qb_played"] = [qbr.rating(q, d) for q, d in zip(starters.qb_id, starters.gameday)]
    tg = tg.merge(starters[["game_id", "team", "qb_played"]], on=["game_id", "team"], how="left")
    stats_by_game = tg.set_index(["game_id", "team"])[STATS]
    league_mean = tg.groupby("season")[STATS].mean()

    decay = 0.5 ** (1 / RECENCY_HALF_LIFE_GAMES)
    rows = []
    prev_final: dict[str, np.ndarray] = {}
    for season, sched in schedule.groupby("season", sort=True):
        mean_prev = league_mean.loc[season - 1].values if (season - 1) in league_mean.index else league_mean.iloc[0].values
        teams = pd.unique(sched[["home_team", "away_team"]].values.ravel())
        for team in teams:
            prior = mean_prev + PRIOR_REGRESSION * (prev_final.get(team, mean_prev) - mean_prev)
            num, den, w_prior = np.zeros(len(STATS)), 0.0, float(PRIOR_GAMES_WEIGHT)
            games = sched[(sched.home_team == team) | (sched.away_team == team)].sort_values("gameday")
            for gid in games.game_id:
                rating = (w_prior * prior + num) / (w_prior + den)
                rows.append((gid, team, *rating))
                if (gid, team) in stats_by_game.index:          # game was played -> learn from it
                    x = stats_by_game.loc[(gid, team)].values.astype(float)
                    x = np.where(np.isnan(x), rating, x)        # e.g. a game with zero runs
                    num, den, w_prior = num * decay + x, den * decay + 1, w_prior * decay
            prev_final[team] = (w_prior * prior + num) / (w_prior + den)
    return pd.DataFrame(rows, columns=["game_id", "team", *STATS])


# --------------------------------------------------------------------------
# Assemble the modeling table
# --------------------------------------------------------------------------
FEATURES = [
    "elo_diff",
    "d_pass_off", "d_pass_def", "d_rush_off", "d_rush_def", "d_success_off", "d_success_def",
    "d_qb_starter", "d_qb_change",
    "home", "rest_diff", "bye_diff", "travel_diff",
    "div_x_elo",
]
# Computed and tested in the backtest, but they did not improve out-of-sample accuracy,
# so the production model leaves them out (and we don't need weather forecasts live).
EXPERIMENTAL = ["wind_x_pass", "cold_x_dome_away"]


def build_features(schedule: pd.DataFrame, team_games: pd.DataFrame, qbr: QBRatings,
                   starter_override: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per game with FEATURES + target columns.

    starter_override: {team: {"qb_id": ...}} for upcoming games (projected starters).
    """
    s = schedule.copy()
    if starter_override:
        upcoming = s.home_score.isna()
        for side in ("home", "away"):
            ids = s.loc[upcoming, f"{side}_team"].map(lambda t: (starter_override.get(t) or {}).get("qb_id"))
            names = s.loc[upcoming, f"{side}_team"].map(lambda t: (starter_override.get(t) or {}).get("qb_name"))
            s.loc[upcoming, f"{side}_qb_id"] = ids.where(ids.notna(), s.loc[upcoming, f"{side}_qb_id"])
            s.loc[upcoming, f"{side}_qb_name"] = names.where(names.notna(), s.loc[upcoming, f"{side}_qb_name"])

    s = s.merge(run_elo(s), on="game_id")
    tr = team_ratings(s, team_games, qbr)
    h = tr.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home_team"})
    a = tr.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away_team"})
    s = s.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])

    s["h_qb_rating"] = [qbr.rating(q, d) for q, d in zip(s.home_qb_id, s.gameday)]
    s["a_qb_rating"] = [qbr.rating(q, d) for q, d in zip(s.away_qb_id, s.gameday)]

    neutral = s.location.eq("Neutral")
    f = pd.DataFrame({"game_id": s.game_id})
    f["elo_diff"] = s.home_elo - s.away_elo
    # Matchup-style differences (EPA/play). For defense, lower allowed is better, so these are away - home.
    f["d_pass_off"] = s.h_pass_epa_off - s.a_pass_epa_off
    f["d_pass_def"] = s.a_pass_epa_def - s.h_pass_epa_def
    f["d_rush_off"] = s.h_rush_epa_off - s.a_rush_epa_off
    f["d_rush_def"] = s.a_rush_epa_def - s.h_rush_epa_def
    f["d_success_off"] = s.h_success_off - s.a_success_off
    f["d_success_def"] = s.a_success_def - s.h_success_def
    # QB: how good is today's starter, and how does he compare to the QBs behind the team's recent stats
    f["d_qb_starter"] = s.h_qb_rating - s.a_qb_rating
    f["d_qb_change"] = (s.h_qb_rating - s.h_qb_played) - (s.a_qb_rating - s.a_qb_played)
    f["home"] = (~neutral).astype(float)
    f["rest_diff"] = (s.home_rest - s.away_rest).clip(-7, 7)
    f["bye_diff"] = (s.home_rest >= 13).astype(float) - (s.away_rest >= 13).astype(float)
    away_trip = [miles(team_location(at, se), team_location(ht, se)) / 1000
                 for at, ht, se in zip(s.away_team, s.home_team, s.season)]
    f["travel_diff"] = np.where(neutral, 0.0, away_trip)  # home team travels 0
    f["div_x_elo"] = s.div_game * f.elo_diff
    outdoor = s.roof.isin(["outdoors", "open"])
    wind = pd.to_numeric(s.wind, errors="coerce").fillna(0)
    temp = pd.to_numeric(s.temp, errors="coerce").fillna(60)
    f["wind_x_pass"] = np.where(outdoor, wind.clip(0, 30) / 10, 0) * (f.d_pass_off + f.d_pass_def)
    away_dome = s.away_team.isin(["ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "MIN", "NO", "LA", "LAC"])
    f["cold_x_dome_away"] = (outdoor & (temp < 40) & away_dome).astype(float)

    meta = s[["game_id", "season", "week", "game_type", "gameday", "gametime", "home_team", "away_team",
              "home_score", "away_score", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
              "location", "roof", "home_elo", "away_elo", "h_qb_rating", "a_qb_rating"]]
    out = meta.merge(f, on="game_id")
    out["margin"] = out.home_score - out.away_score
    out["home_win"] = np.where(out.margin > 0, 1.0, np.where(out.margin < 0, 0.0, 0.5))
    return out
