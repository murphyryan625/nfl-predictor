"""Command-line entry point.

    python run.py backtest   # refresh data, walk-forward backtest, write MODEL_REPORT.md
    python run.py full       # Tuesday: refresh data, predict the upcoming week, build the site
    python run.py refresh    # Thu/Sun: same as full (re-pulls injuries & QB status) — kept separate for clarity in logs
    python run.py score      # Monday: score finished games into the tracking page, rebuild the site
    python run.py export     # rewrite this week's picks CSVs from the ledger (no downloads, no new predictions)

full / refresh / score also write the picks CSVs (site/data/picks.csv, and predictions.csv in the
personal dashboard folder when it exists - see nflpred/export.py).

There is no saved model file: training takes about a second, so every run retrains on
all completed seasons. That means "retrain" happens automatically every run.
"""
from __future__ import annotations

import argparse
import sys
import time

from nflpred import data
from nflpred.config import FIRST_SEASON
from nflpred.features import build_features
from nflpred.qb import QBRatings


def load_everything(refresh: bool = True):
    season = data.current_season()
    if refresh:
        print(f"Updating data cache (seasons {FIRST_SEASON}-{season}) ...")
        data.update_cache(list(range(FIRST_SEASON, season + 1)))
    schedule = data.load_schedules()
    team_games, qb_games = data.load_team_games(), data.load_qb_games()
    return season, schedule, team_games, QBRatings(qb_games, schedule)


def cmd_backtest(args):
    from nflpred.report import run_report
    season, schedule, team_games, qbr = load_everything(not args.no_refresh)
    table = build_features(schedule, team_games, qbr)
    print(run_report(table, season))


def cmd_predict(args):
    from nflpred.predict import predict_upcoming
    from nflpred.track import record_predictions, score_results
    from nflpred.site import build_site
    season, schedule, team_games, qbr = load_everything(not args.no_refresh)
    preds = predict_upcoming(season, schedule, team_games, qbr, run_label=args.command)
    record_predictions(preds)
    score_results(schedule)
    build_site(schedule)


def cmd_score(args):
    from nflpred.track import score_results
    from nflpred.site import build_site
    season, schedule, *_ = load_everything(not args.no_refresh)
    score_results(schedule)
    build_site(schedule)


def cmd_export(args):
    """Rewrite the picks CSVs from the ledger as it stands. No downloads, no new predictions."""
    from nflpred.export import write_picks
    from nflpred.site import current_week_games
    from nflpred.track import load_ledger
    _, week, cur = current_week_games(data.load_schedules(), load_ledger())
    print(f"Exporting week {week}")
    write_picks(cur)


def cmd_backfill(args):
    """One-time: add out-of-sample predictions for this season's finished games to the tracking page."""
    from nflpred.track import backfill
    from nflpred.site import build_site
    season, schedule, team_games, qbr = load_everything(not args.no_refresh)
    backfill(build_features(schedule, team_games, qbr), season)
    build_site(schedule)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="NFL game predictions")
    p.add_argument("command", choices=["backtest", "full", "refresh", "score", "backfill", "export"])
    p.add_argument("--no-refresh", action="store_true", help="use cached data only (no downloads)")
    args = p.parse_args()
    t = time.time()
    {"backtest": cmd_backtest, "full": cmd_predict, "refresh": cmd_predict, "score": cmd_score,
     "backfill": cmd_backfill, "export": cmd_export}[args.command](args)
    print(f"Done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
