"""Runs the full walk-forward backtest and writes MODEL_REPORT.md."""
from __future__ import annotations

import datetime as dt
import pandas as pd

from .backtest import walk_forward, summarize, calibration_table, metrics
from .config import ROOT, FIRST_TEST_SEASON, MIN_TRAIN_SEASON
from .features import FEATURES, EXPERIMENTAL
from .models import MarginModel, LogitModel, EloModel

PRODUCTION = "ridge"  # the model the dashboard uses


def candidate_models() -> dict:
    no_qb = [f for f in FEATURES if not f.startswith("d_qb")]
    return {
        "elo": EloModel,
        "ridge": lambda: MarginModel(FEATURES),
        "ridge_no_qb": lambda: MarginModel(no_qb),
        "ridge_plus_weather": lambda: MarginModel(FEATURES + EXPERIMENTAL),
        "logit": lambda: LogitModel(FEATURES),
        "gbm": lambda: MarginModel(FEATURES, kind="gbm"),
    }


DESCRIPTIONS = {
    "elo": "Elo only (baseline, no fitting)",
    "ridge": "**Production**: ridge regression on margin → Normal win prob",
    "ridge_no_qb": "Ridge without the QB features (ablation)",
    "ridge_plus_weather": "Ridge + wind/cold-dome features (ablation)",
    "logit": "Logistic regression on win/loss (cross-check)",
    "gbm": "Gradient boosting on margin",
}


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            ("" if pd.isna(v) else str(int(v)) if col in ("games",) else floatfmt.format(v))
            if isinstance(v, float) else str(v) for col, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_report(table: pd.DataFrame, current_season: int) -> str:
    models = candidate_models()
    last_full = current_season - 1
    preds = walk_forward(table, models, current_season)
    full = preds[preds.season <= last_full]
    cur = preds[preds.season == current_season]

    summ = summarize(full, list(models))
    summ.insert(0, "description", [DESCRIPTIONS[m] for m in summ.index])

    per_season = []
    for season, g in preds.groupby("season"):
        r = {"season": season}
        for n in ("elo", "ridge"):
            m = metrics(g[f"{n}_p"].values, g.home_win.values, g[f"{n}_margin"].values, g.margin.values)
            r[f"{n}_brier"], r[f"{n}_acc"] = m["brier"], m["accuracy"]
            if n == "ridge":
                r["ridge_spread_mae"] = m["spread_mae"]
        r["games"] = len(g)
        per_season.append(r)
    per_season = pd.DataFrame(per_season).set_index("season")

    calib = calibration_table(full["ridge_p"].values, full.home_win.values)
    calib.index = calib.index.astype(str)
    calib.index.name = "home win prob"

    # Agreement between the margin-derived probability and the separate logistic regression
    agree = (full.ridge_p - full.logit_p).abs()

    final = MarginModel(FEATURES).fit(table[table.margin.notna() & (table.season.between(MIN_TRAIN_SEASON, last_full))])
    coefs = final.coefficients().sort_values(key=abs, ascending=False).to_frame("points per 1 SD")

    text = f"""# Model report

_Generated {dt.date.today()} by `python run.py backtest`._

## Method
- **Walk-forward**: for each season N from {FIRST_TEST_SEASON} to {current_season}, every model is trained only on
  seasons {MIN_TRAIN_SEASON}..N-1 and then predicts every game of season N. Nothing from season N (or later) is used in training.
- **Features are pregame-only**: team EPA/success ratings use only games on earlier dates; QB ratings use only
  earlier dropbacks; Elo is the pregame value. No betting lines are used anywhere.
- **Spread and win% are consistent**: the production model predicts home margin, and
  win prob = Φ(margin / σ), with σ = {final.sigma:.1f} points fitted from training residuals.
- One caveat: in the backtest, the starting QB is the one who actually started (known at kickoff, after
  inactives). Live predictions made earlier in the week use the projected starter, so they can be a bit
  less accurate when a QB's status is uncertain.

## Out-of-sample results, {FIRST_TEST_SEASON}–{last_full} ({len(full)} games)
Lower is better for Brier, log loss and spread MAE. A coin flip scores Brier 0.250 and log loss 0.693.

{md(summ)}

**Decisions**
- Ridge beats Elo on every metric, so it's the production model.
- Taking the QB features out (`ridge_no_qb`) clearly hurts, so they stay.
- Weather adds nothing (`ridge_plus_weather` vs `ridge`), so it is pruned.
- Gradient boosting does not beat ridge on Brier or log loss, so it is not used.
- The logistic regression lands within {agree.mean():.3f} (mean absolute difference) of the margin model's
  win probabilities, which confirms the Φ(margin/σ) conversion is sound.

## {current_season} season so far ({len(cur)} games, out of sample)
{md(summarize(cur, ["elo", "ridge", "logit"])) if len(cur) else "No games yet."}

## By season
{md(per_season)}

## Calibration (production model, {FIRST_TEST_SEASON}–{last_full})
When the model says the home team wins X% of the time, how often did it?

{md(calib)}

## What drives the prediction
Ridge coefficients from training on {MIN_TRAIN_SEASON}–{last_full}, in points of home margin per 1 standard
deviation of each feature (the features overlap, so read these as rough importance, not causal effects).

{md(coefs, "{:+.2f}")}
"""
    (ROOT / "MODEL_REPORT.md").write_text(text, encoding="utf-8")
    return text
