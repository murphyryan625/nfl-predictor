"""Writes this week's picks as a small, readable CSV for other tools:

  site/data/picks.csv                  published with the site (GitHub Pages)
  <DASHBOARD_DIR>/predictions.csv      "My Model Picks" table in the personal Streamlit dashboard
                                       (only written when that folder exists, i.e. on the laptop)

This is a display copy of the current week. The full record stays in data/predictions.csv (the ledger).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import DASHBOARD_DIR, SITE_DATA

COLUMNS = ["Week", "Kickoff (CT)", "Matchup", "Pick", "Win %", "Spread", "QBs", "Notes", "Result"]


def _kickoff(iso: str) -> str:
    try:
        d = dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return ""
    return f"{d:%a} {d.month}/{d.day} {d.hour % 12 or 12}:{d.minute:02d} {'AM' if d.hour < 12 else 'PM'}"


def _spread(pick: str, margin: float) -> str:
    pts = round(abs(margin) * 2) / 2            # nearest half point
    return "Pick'em" if pts == 0 else f"{pick} -{pts:g}"


def _result(r, pick: str) -> str:
    if pd.isna(r.home_score) or pd.isna(r.away_score):
        return ""
    hs, as_ = int(r.home_score), int(r.away_score)
    score = f"{r.away_team} {as_}, {r.home_team} {hs}"
    if hs == as_:
        return f"{score} (tie)"
    winner = r.home_team if hs > as_ else r.away_team
    return f"{score} ({'correct' if winner == pick else 'wrong'})"


def picks_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game from ledger rows (see track.py for the ledger columns)."""
    rows = []
    for r in games.sort_values(["kickoff_ct", "game_id"]).itertuples():
        home_pick = r.home_win_prob >= 0.5
        pick = r.home_team if home_pick else r.away_team
        prob = r.home_win_prob if home_pick else 1 - r.home_win_prob
        qb = lambda v: v if isinstance(v, str) and v else "?"
        rows.append({
            "Week": int(r.week),
            "Kickoff (CT)": _kickoff(r.kickoff_ct),
            "Matchup": f"{r.away_team} @ {r.home_team}",
            "Pick": pick,
            "Win %": f"{prob:.0%}",
            "Spread": _spread(pick, r.pred_margin),
            "QBs": f"{qb(r.away_qb)} vs {qb(r.home_qb)}",
            "Notes": "; ".join(r.notes) if isinstance(r.notes, list) else "",
            "Result": _result(r, pick),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def write_picks(games: pd.DataFrame) -> None:
    table = picks_table(games) if not games.empty else pd.DataFrame(columns=COLUMNS)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    targets = [SITE_DATA / "picks.csv"]
    if DASHBOARD_DIR.is_dir():                  # absent on GitHub Actions, so nothing is written there
        targets.append(DASHBOARD_DIR / "predictions.csv")
    for path in targets:
        try:
            table.to_csv(path, index=False, encoding="utf-8")
            print(f"Picks written: {path} ({len(table)} games)")
        except OSError as e:                    # e.g. the file is open in Excel - don't fail the run over it
            print(f"  warning: couldn't write {path} ({e})")
