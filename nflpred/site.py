"""Writes the JSON files the static dashboard reads:

  site/data/current.json   this week's games and predictions
  site/data/history.json   every scored prediction + weekly/season summaries
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import SITE_DATA
from .data import TEAM_FIX
from .predict import upcoming_week, CENTRAL
from .track import load_ledger


def _teams() -> dict:
    try:
        import nflreadpy as nfl
        t = nfl.load_teams().to_pandas()
        t["team_abbr"] = t["team_abbr"].replace(TEAM_FIX)
        t = t.drop_duplicates("team_abbr", keep="last")
        return {r.team_abbr: {"name": r.team_name, "nick": r.team_nick, "color": r.team_color}
                for r in t.itertuples()}
    except Exception as e:
        print(f"  warning: team info unavailable ({e})")
        return {}


def _clean(v):
    """Make values JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def build_site(schedule: pd.DataFrame) -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    ledger = load_ledger()
    now = dt.datetime.now(CENTRAL).isoformat(timespec="minutes")
    season = int(schedule.season.max())
    week, _ = upcoming_week(schedule, season)

    # ---------------- current week ----------------
    if week < 0 and not ledger.empty:              # season over: show the last predicted week
        week = int(ledger[ledger.season == season].week.max())
    cur = ledger[(ledger.season == season) & (ledger.week == week)] if not ledger.empty else ledger
    current = {
        "generated_at": now, "season": season, "week": week,
        "teams": _teams(),
        "games": _records(cur.sort_values(["kickoff_ct", "game_id"])) if not cur.empty else [],
    }
    (SITE_DATA / "current.json").write_text(json.dumps(current, indent=1), encoding="utf-8")

    # ---------------- history ----------------
    hist = ledger[ledger.home_score.notna()].copy() if not ledger.empty and "home_score" in ledger else pd.DataFrame()
    summary, weekly = {}, []
    if not hist.empty:
        hist["margin"] = hist.home_score - hist.away_score
        hist["y"] = np.where(hist.margin > 0, 1.0, np.where(hist.margin < 0, 0.0, 0.5))
        hist["pick"] = np.where(hist.home_win_prob >= 0.5, hist.home_team, hist.away_team)
        hist["pick_prob"] = np.maximum(hist.home_win_prob, 1 - hist.home_win_prob)
        hist["correct"] = np.where(hist.y == 0.5, np.nan, (hist.home_win_prob >= 0.5) == (hist.y == 1))
        hist["spread_error"] = (hist.pred_margin - hist.margin).abs()
        hist["brier"] = (hist.home_win_prob - hist.y) ** 2
        hist = hist.sort_values(["season", "week", "kickoff_ct"])

        def summarize(g):
            decided = g.correct.dropna()
            return {"games": int(len(g)), "wins": int(decided.sum()), "losses": int(len(decided) - decided.sum()),
                    "brier": round(float(g.brier.mean()), 4), "spread_mae": round(float(g.spread_error.mean()), 2)}

        for (s, w), g in hist.groupby(["season", "week"], sort=True):
            upto = hist[(hist.season == s) & (hist.week <= w)]
            weekly.append({"season": int(s), "week": int(w), **summarize(g),
                           "running_brier": round(float(upto.brier.mean()), 4),
                           "live": bool((g.run != "backfill").any())})
        summary = {"all": summarize(hist),
                   "live_only": summarize(hist[hist.run != "backfill"]) if (hist.run != "backfill").any() else None}
        hist = hist.drop(columns=["y"])

    history = {"generated_at": now, "summary": summary, "weekly": weekly,
               "games": _records(hist.sort_values(["season", "week", "kickoff_ct"], ascending=[False, False, True]))
               if not hist.empty else []}
    (SITE_DATA / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"Site data written: week {week}, {len(current['games'])} games; {len(history['games'])} scored")
