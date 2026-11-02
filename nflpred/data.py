"""Loading nflverse data with a local cache.

History never changes, so for each finished season we boil the huge play-by-play
table (~50k rows x 370 columns) down to two small tables and save them as parquet:

  data/team_games/<season>.parquet  one row per team per game (EPA, success rate...)
  data/qb_games/<season>.parquet    one row per passer per game (dropbacks, EPA)

Finished seasons are downloaded once. The current season is re-downloaded each
run because new games get added (it takes a few seconds).
"""
from __future__ import annotations

import pandas as pd
import nflreadpy as nfl

from .config import DATA_DIR, FIRST_SEASON

TEAM_DIR = DATA_DIR / "team_games"
QB_DIR = DATA_DIR / "qb_games"

# nflverse uses current abbreviations in pbp but old ones in some tables; normalize.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def current_season() -> int:
    return nfl.get_current_season()


def load_schedules() -> pd.DataFrame:
    """All games 1999-present, including future ones (scores are NaN until played)."""
    s = nfl.load_schedules().to_pandas()
    s = s[s.season >= FIRST_SEASON].copy()
    for c in ("home_team", "away_team"):
        s[c] = s[c].replace(TEAM_FIX)
    # Betting columns are dropped on purpose: the model must never see them.
    betting = ["away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
               "home_spread_odds", "total_line", "under_odds", "over_odds"]
    s = s.drop(columns=[c for c in betting if c in s.columns])
    s["gameday"] = pd.to_datetime(s["gameday"])
    return s.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


def _summarize_pbp(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce one season of play-by-play to team-game and QB-game summaries."""
    plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)]
    plays = plays[plays["epa"].notna() & plays["posteam"].notna()].copy()
    plays["posteam"] = plays["posteam"].replace(TEAM_FIX)
    plays["defteam"] = plays["defteam"].replace(TEAM_FIX)
    plays["is_pass"] = plays["pass"] == 1

    def agg(g: pd.DataFrame) -> pd.Series:
        p, r = g[g.is_pass], g[~g.is_pass]
        return pd.Series({
            "plays": len(g),
            "epa": g.epa.mean(), "success": g.success.mean(),
            "pass_epa": p.epa.mean(), "pass_success": p.success.mean(),
            "rush_epa": r.epa.mean(), "rush_success": r.success.mean(),
        })

    keys = ["season", "week", "game_id"]
    off = plays.groupby(keys + ["posteam", "defteam"]).apply(agg, include_groups=False).reset_index()
    off = off.rename(columns={"posteam": "team", "defteam": "opp"})
    # A team's defense on a given game = the opponent's offense in that game.
    dfn = off.rename(columns={"team": "opp", "opp": "team"})
    stat_cols = ["plays", "epa", "success", "pass_epa", "pass_success", "rush_epa", "rush_success"]
    team_games = off.merge(
        dfn[keys + ["team", "opp"] + stat_cols], on=keys + ["team", "opp"], suffixes=("_off", "_def"))

    # QB-level: every dropback credited to the passer (includes sacks and scrambles).
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["qb_epa"].notna()].copy()
    db["passer"] = db["passer_player_id"].fillna(db["rusher_player_id"])
    db["passer_name"] = db["passer_player_name"].fillna(db["rusher_player_name"])
    db["posteam"] = db["posteam"].replace(TEAM_FIX)
    qb_games = (db.dropna(subset=["passer"])
                  .groupby(keys + ["posteam", "passer"])
                  .agg(name=("passer_name", "first"), dropbacks=("qb_epa", "size"), epa_sum=("qb_epa", "sum"))
                  .reset_index().rename(columns={"posteam": "team", "passer": "qb_id"}))
    return team_games, qb_games


def update_cache(seasons: list[int], refresh_current: bool = True) -> None:
    """Download and summarize any seasons that aren't cached yet."""
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    QB_DIR.mkdir(parents=True, exist_ok=True)
    cur = current_season()
    for season in seasons:
        tpath, qpath = TEAM_DIR / f"{season}.parquet", QB_DIR / f"{season}.parquet"
        if tpath.exists() and qpath.exists() and not (refresh_current and season == cur):
            continue
        print(f"  downloading play-by-play {season} ...", flush=True)
        try:
            pbp = nfl.load_pbp([season]).to_pandas()
        except Exception as e:  # e.g. current season before week 1
            print(f"  (no pbp for {season}: {e})")
            continue
        team_games, qb_games = _summarize_pbp(pbp)
        team_games.to_parquet(tpath, index=False)
        qb_games.to_parquet(qpath, index=False)


def load_team_games() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(TEAM_DIR.glob("*.parquet"))], ignore_index=True)


def load_qb_games() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(QB_DIR.glob("*.parquet"))], ignore_index=True)
