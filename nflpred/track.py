"""The prediction ledger: data/predictions.csv.

* Every run writes its predictions for games that have NOT kicked off yet
  (a newer run replaces an older prediction for the same game, e.g. Sunday's
  re-run after inactives). Once a game kicks off its prediction is frozen.
* `score_results` fills in final scores so the dashboard can show the track record.
* `backfill` adds out-of-sample predictions for earlier weeks of the current season
  (made the backtest way: model trained only on prior seasons), labeled "backfill".
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from .config import DATA_DIR, MIN_TRAIN_SEASON
from .predict import kickoff_central, CENTRAL

LEDGER = DATA_DIR / "predictions.csv"
LIST_COLS = ["home_qb_flags", "away_qb_flags", "notes"]


def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame()
    df = pd.read_csv(LEDGER)
    for c in LIST_COLS:
        if c in df:
            df[c] = df[c].fillna("[]").map(json.loads)
    return df


def save_ledger(df: pd.DataFrame) -> None:
    out = df.copy()
    for c in LIST_COLS:
        if c in out:
            out[c] = out[c].map(lambda v: json.dumps(v if isinstance(v, list) else []))
    out.sort_values(["season", "week", "kickoff_ct", "game_id"]).to_csv(LEDGER, index=False)


def record_predictions(preds: pd.DataFrame) -> None:
    if preds.empty:
        return
    now = dt.datetime.now(CENTRAL)
    fresh = preds[pd.to_datetime(preds.kickoff_ct, utc=True) > now]   # never touch games already underway
    ledger = load_ledger()
    if not ledger.empty:
        ledger = ledger[~ledger.game_id.isin(fresh.game_id)]
    save_ledger(pd.concat([ledger, fresh], ignore_index=True))
    print(f"Recorded {len(fresh)} predictions ({len(preds) - len(fresh)} games already started, left unchanged)")


def score_results(schedule: pd.DataFrame) -> None:
    ledger = load_ledger()
    if ledger.empty:
        return
    scores = schedule.set_index("game_id")[["home_score", "away_score"]]
    ledger["home_score"] = ledger.game_id.map(scores.home_score)
    ledger["away_score"] = ledger.game_id.map(scores.away_score)
    save_ledger(ledger)
    print(f"Scored ledger: {ledger.home_score.notna().sum()} of {len(ledger)} games have results")


def backfill(table: pd.DataFrame, season: int) -> None:
    """Out-of-sample predictions for this season's completed games (model trained on earlier seasons)."""
    from .features import FEATURES
    from .models import MarginModel
    train = table[table.margin.notna() & table.season.between(MIN_TRAIN_SEASON, season - 1)]
    done = table[(table.season == season) & table.margin.notna()].copy()
    model = MarginModel(FEATURES).fit(train)
    done["pred_margin"] = model.predict_margin(done).round(2)
    done["home_win_prob"] = model.predict_proba(done).round(4)
    rows = pd.DataFrame({
        "game_id": done.game_id, "season": done.season, "week": done.week, "game_type": done.game_type,
        "kickoff_ct": [kickoff_central(d, t).isoformat() for d, t in zip(done.gameday, done.gametime)],
        "home_team": done.home_team, "away_team": done.away_team,
        "home_qb": done.home_qb_name, "away_qb": done.away_qb_name,
        "home_qb_flags": [[] for _ in range(len(done))], "away_qb_flags": [[] for _ in range(len(done))],
        "pred_margin": done.pred_margin, "home_win_prob": done.home_win_prob,
        "notes": [[] for _ in range(len(done))], "predicted_at": "", "run": "backfill",
        "home_score": done.home_score, "away_score": done.away_score,
    })
    ledger = load_ledger()
    if not ledger.empty:
        rows = rows[~rows.game_id.isin(ledger.game_id)]   # never overwrite real live predictions
    save_ledger(pd.concat([ledger, rows], ignore_index=True))
    print(f"Backfilled {len(rows)} games for {season}")
