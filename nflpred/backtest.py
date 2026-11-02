"""Walk-forward backtest: for each test season N, train on seasons < N, predict season N.

The features themselves are already pregame-only (see features.py), and the model
never sees any game from season N while it is being trained, so every number in
the report is genuinely out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIRST_TEST_SEASON, MIN_TRAIN_SEASON, ROOT
from .models import MarginModel, LogitModel, EloModel


def metrics(p: np.ndarray, y: np.ndarray, pred_margin: np.ndarray | None, margin: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    decided = y != 0.5
    out = {
        "games": len(y),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p[decided] > 0.5) == (y[decided] == 1))),
    }
    if pred_margin is not None:
        out["spread_mae"] = float(np.mean(np.abs(pred_margin - margin)))
    return out


def calibration_table(p: np.ndarray, y: np.ndarray, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)) -> pd.DataFrame:
    # Fold to the favorite's perspective? No — keep home-team view so both tails are visible.
    cut = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"bin": cut, "p": p, "y": y}).groupby("bin", observed=True).agg(
        games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t


def walk_forward(table: pd.DataFrame, make_models: dict, last_season: int) -> pd.DataFrame:
    """Return out-of-sample predictions for every game in FIRST_TEST_SEASON..last_season."""
    done = table[table.margin.notna() & (table.season >= MIN_TRAIN_SEASON)]
    preds = []
    for season in range(FIRST_TEST_SEASON, last_season + 1):
        train, test = done[done.season < season], done[done.season == season]
        if test.empty:
            continue
        row = test[["game_id", "season", "week", "margin", "home_win"]].copy()
        for name, make in make_models.items():
            m = make().fit(train)
            row[f"{name}_p"] = m.predict_proba(test)
            row[f"{name}_margin"] = m.predict_margin(test) if hasattr(m, "predict_margin") else np.nan
        preds.append(row)
    return pd.concat(preds, ignore_index=True)


def summarize(preds: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        pm = preds[f"{n}_margin"].values
        m = metrics(preds[f"{n}_p"].values, preds.home_win.values,
                    None if np.isnan(pm).all() else pm, preds.margin.values)
        rows.append({"model": n, **m})
    return pd.DataFrame(rows).set_index("model")
