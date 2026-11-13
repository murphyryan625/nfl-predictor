"""Command line for the CLV tracker. Run from the repo root:

    python -m clv weekly            # everything, in order: snapshot -> log picks -> close -> grade -> report
    python -m clv status            # upcoming games (with their game_id) and picks still waiting on something
    python -m clv report            # rebuild the report from the picks file and print it

    python -m clv snapshot          # just record the current market
    python -m clv log [--replace]   # snapshot + log the model's picks (--replace re-logs games not yet started)
    python -m clv close             # attach closing lines to picks whose games have kicked off
    python -m clv grade             # fill in results and units

    python -m clv add --game 2026_10_BUF_NYJ --type spread --side BUF --line -8.5 --price -110 --book DraftKings
    python -m clv set-close --pick 2026_10_BUF_NYJ:spread:placed:1 --line -9.5 --price -108 --opp-price -112
    python -m clv backfill my_past_picks.csv

The same steps run automatically inside `python run.py full|refresh|score`.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import markets, tracker
from .report import write_report


def _status() -> None:
    now = tracker.now_utc()
    games, picks = tracker.model_games(), tracker.load_picks()
    up = games[games.kickoff_utc > now].sort_values("kickoff_utc")
    key = "set" if markets.api_key() else "not set (using free nflverse lines)"
    print(f"ODDS_API_KEY: {key}\n\nUpcoming games with a model prediction: {len(up)}")
    for g in up.itertuples():
        mine = picks[picks.game_id == g.game_id]
        print(f"  {g.game_id:<18} {g.away_team} @ {g.home_team}  model: {g.home_team} {-g.pred_margin:+.1f}, "
              f"{g.home_win_prob:.0%}   picks logged: {len(mine)}")
    kicked = pd.to_datetime(picks.kickoff_utc, utc=True) <= now
    for title, rows in (("Kicked off, no closing line yet", picks[kicked & (picks.close_source == "")]),
                        ("Kicked off, not graded yet", picks[kicked & (picks.result == "")])):
        print(f"\n{title}: {len(rows)}")
        for p in rows.itertuples():
            print(f"  {p.pick_id}")
    print(f"\nTotal picks on file: {len(picks)}")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m clv", description="Closing Line Value tracker")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("weekly", "status", "report", "snapshot", "close", "grade"):
        sub.add_parser(name)
    sub.add_parser("log").add_argument("--replace", action="store_true",
                                       help="re-log picks for games that haven't kicked off")
    a = sub.add_parser("add", help="log a bet by hand")
    a.add_argument("--game", required=True, help="game_id, e.g. 2026_10_BUF_NYJ (see `status`)")
    a.add_argument("--type", required=True, choices=["spread", "moneyline", "total"])
    a.add_argument("--side", required=True, help="team abbreviation, or over/under")
    a.add_argument("--line", type=float, help="handicap of your side (-3.5 = laying 3.5), or the total")
    a.add_argument("--price", type=float, required=True, help="American odds, e.g. -110")
    a.add_argument("--book", default="")
    a.add_argument("--stake", type=float, default=1.0, help="units (default 1)")
    a.add_argument("--track", default="placed", help='report group (default "placed")')
    c = sub.add_parser("set-close", help="type in a closing line (manual fallback)")
    c.add_argument("--pick", required=True, help="pick_id (see `status`)")
    c.add_argument("--line", type=float, help="closing handicap of your side, or closing total")
    c.add_argument("--price", type=float, help="closing price on your side")
    c.add_argument("--opp-price", type=float, help="closing price on the other side (needed to remove the vig)")
    sub.add_parser("backfill", help="load past picks from a CSV").add_argument("csv")
    args = p.parse_args(argv)

    try:
        if args.command == "weekly":
            tracker.run("weekly")
        elif args.command == "status":
            _status()
        elif args.command == "report":
            write_report()
        elif args.command in ("snapshot", "log"):
            games = tracker.model_games()
            snaps = tracker.take_snapshot(games)
            if args.command == "log":
                tracker.log_picks(games, snaps, replace=args.replace)
        elif args.command == "close":
            tracker.capture_close()
            write_report(quiet=True)
        elif args.command == "grade":
            tracker.grade()
            write_report(quiet=True)
        elif args.command == "add":
            pick = tracker.add_pick(markets.load_market_schedule(), args.game, args.type, args.side, args.price,
                                    line=args.line, book=args.book, stake=args.stake, track=args.track)
            picks = tracker.load_picks()
            tracker.save_picks(pd.concat([picks, pd.DataFrame([pick], columns=tracker.PICK_COLUMNS)], ignore_index=True)
                               if len(picks) else pd.DataFrame([pick], columns=tracker.PICK_COLUMNS))
            print(f"Logged {pick['pick_id']}: {pick['side']} {pick['bet_type']}"
                  + (f" {pick['line']:+g}" if pick["line"] is not None else "") + f" at {pick['price']:+g}")
        elif args.command == "set-close":
            pick = tracker.set_close(args.pick, price=args.price, opp_price=args.opp_price, line=args.line)
            print(f"{pick['pick_id']}: CLV EV {pick['clv_ev']:+.1%}"
                  + (f", {pick['clv_points']:+g} pts" if pick["clv_points"] is not None else ""))
            write_report(quiet=True)
        elif args.command == "backfill":
            tracker.backfill(args.csv)
            tracker.run("score")
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
