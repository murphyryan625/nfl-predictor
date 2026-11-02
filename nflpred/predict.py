"""Predict the upcoming week's games."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import MIN_TRAIN_SEASON, TIMEZONE, QB_PRIOR_DROPBACKS
from .features import build_features, FEATURES
from .models import MarginModel
from .qb import QBRatings, project_starters

EASTERN, CENTRAL = ZoneInfo("America/New_York"), ZoneInfo(TIMEZONE)


def kickoff_central(gameday, gametime) -> dt.datetime:
    """nflverse lists kickoff in US Eastern time; convert to Central."""
    hh, mm = (str(gametime) if isinstance(gametime, str) else "13:00").split(":")[:2]
    et = dt.datetime.combine(pd.Timestamp(gameday).date(), dt.time(int(hh), int(mm)), tzinfo=EASTERN)
    return et.astimezone(CENTRAL)


def upcoming_week(schedule: pd.DataFrame, season: int) -> tuple[int, pd.DataFrame]:
    """The earliest week of `season` that still has unplayed games."""
    left = schedule[(schedule.season == season) & schedule.home_score.isna()]
    if left.empty:
        return -1, left
    week = int(left.week.min())
    return week, schedule[(schedule.season == season) & (schedule.week == week)]


def predict_upcoming(season: int, schedule: pd.DataFrame, team_games: pd.DataFrame, qbr: QBRatings,
                     run_label: str = "") -> pd.DataFrame:
    week, games = upcoming_week(schedule, season)
    if week < 0:
        print("No upcoming games this season.")
        return pd.DataFrame()
    pending = games[games.home_score.isna()]
    teams = sorted(set(pending.home_team) | set(pending.away_team))
    print(f"Predicting {season} week {week}: {len(pending)} games")

    starters = project_starters(season, week, teams, schedule)
    table = build_features(schedule, team_games, qbr, starter_override=starters)

    train = table[table.margin.notna() & (table.season >= MIN_TRAIN_SEASON)]
    model = MarginModel(FEATURES).fit(train)
    target = table[table.game_id.isin(pending.game_id)].copy()
    target["pred_margin"] = model.predict_margin(target)
    target["home_win_prob"] = model.predict_proba(target)

    now = dt.datetime.now(CENTRAL)
    rows = []
    for g in target.itertuples(index=False):
        notes = []
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            info = starters.get(team, {})
            notes += [f"{team}: {f}" for f in info.get("flags", [])]
            qb_id = getattr(g, f"{side}_qb_id")
            if qbr.dropbacks_known(qb_id) is False:
                notes.append(f"{team}: starter has no NFL dropbacks — rated as replacement level")
            elif _career_dropbacks(qbr, qb_id) < QB_PRIOR_DROPBACKS:
                notes.append(f"{team}: starter has limited NFL history")
        if g.week <= 3 and g.game_type == "REG":
            notes.append("Early season: ratings lean heavily on last season")
        if abs(g.home_win_prob - 0.5) < 0.05:
            notes.append("Toss-up")
        if g.location == "Neutral":
            notes.append("Neutral site (no home field)")
        kick = kickoff_central(g.gameday, g.gametime)
        rows.append({
            "game_id": g.game_id, "season": int(g.season), "week": int(g.week), "game_type": g.game_type,
            "kickoff_ct": kick.isoformat(), "home_team": g.home_team, "away_team": g.away_team,
            "home_qb": g.home_qb_name, "away_qb": g.away_qb_name,
            "home_qb_flags": starters.get(g.home_team, {}).get("flags", []),
            "away_qb_flags": starters.get(g.away_team, {}).get("flags", []),
            "pred_margin": round(float(g.pred_margin), 2),          # + means home favored
            "home_win_prob": round(float(g.home_win_prob), 4),
            "notes": notes, "predicted_at": now.isoformat(timespec="minutes"), "run": run_label,
        })
    return pd.DataFrame(rows)


def _career_dropbacks(qbr: QBRatings, qb_id) -> float:
    # Number of games is a cheap proxy (~35 dropbacks per start)
    return len(qbr.history.get(qb_id, ([], []))[0]) * 35
