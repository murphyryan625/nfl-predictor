"""Turns clv_picks.csv into the numbers on the CLV page (site/data/clv.json) and a console summary.

Each "track" is reported separately: "model" = picks logged automatically from the model,
"placed" = bets entered by hand.

How to read the main numbers
  avg CLV EV    expected profit per unit staked IF the no-vig closing line is the true price.
                Betting the closing line itself scores about -2% (moneyline) to -4.5% (spread at
                -110), so anything above zero means you beat a fair price, not just the vig.
  beat close %  share of bets that got a better number (or the same number at a better price).
                A coin-flipper sits near 50%.
  expected vs actual units   sum of stake x CLV EV, against real profit. Over a long run they
                should converge; a gap over a short run is variance, in either direction.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from .config import ML_EDGE_BUCKETS, REPORT_JSON, SPREAD_EDGE_BUCKETS
from .odds_math import mean_ci
from .tracker import load_picks


def _r(v, digits=4):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)


def _stats(df: pd.DataFrame) -> dict:
    """Summary of a group of picks. CLV figures use picks with a closing line; results use graded picks."""
    closed, graded = df[df.clv_ev.notna()], df[df.result != ""]
    both = graded[graded.clv_ev.notna()]
    line_bets = closed[closed.clv_points.notna()]
    staked = float(graded.stake.sum())
    return {
        "n": int(len(df)), "n_closed": int(len(closed)), "n_graded": int(len(graded)),
        "avg_clv_ev": _r(closed.clv_ev.mean()) if len(closed) else None,
        "avg_clv_prob": _r(closed.clv_prob.mean()) if len(closed) else None,
        "avg_clv_points": _r(line_bets.clv_points.mean(), 2) if len(line_bets) else None,
        "beat_close_pct": _r(closed.beat_close.mean()) if len(closed) else None,
        "wins": int((graded.result == "W").sum()), "losses": int((graded.result == "L").sum()),
        "pushes": int((graded.result == "P").sum()),
        "units": _r(graded.profit_units.sum(), 2) if len(graded) else None,
        "roi": _r(graded.profit_units.sum() / staked) if staked else None,
        # Expected and actual profit over the SAME bets (closed and graded), so they can be compared.
        "expected_units": _r((both.stake * both.clv_ev).sum(), 2) if len(both) else None,
        "actual_units": _r(both.profit_units.sum(), 2) if len(both) else None,
    }


def _edge_bucket(row) -> str:
    """Bucket by how strongly the model disagreed with the market when the pick was logged."""
    e = row.model_edge
    if pd.isna(e) or row.bet_type == "total":
        return "no model edge recorded"
    if e < 0:
        return "against the model"
    buckets = SPREAD_EDGE_BUCKETS if row.bet_type == "spread" else ML_EDGE_BUCKETS
    label = next(lab for lab, lo, hi in buckets if lo <= e < hi)
    return f"{row.bet_type}: {label}"


# Display order for the edge table: weakest to strongest, spreads first.
EDGE_ORDER = ([f"spread: {lab}" for lab, _, _ in SPREAD_EDGE_BUCKETS] + [f"moneyline: {lab}" for lab, _, _ in ML_EDGE_BUCKETS]
              + ["against the model", "no model edge recorded"])


def _sample_size(df: pd.DataFrame) -> dict:
    """Is the average CLV distinguishable from zero yet?

    A spread pick and a moneyline pick on the same game move together, so they are not two
    independent observations. CLV is therefore averaged per game first, and the confidence interval
    is computed across games:   CI = mean +/- 1.96 * sd / sqrt(games).
    """
    closed = df[df.clv_ev.notna()]
    ci = mean_ci(closed.groupby("game_id").clv_ev.mean())
    graded = df[df.result != ""]
    # For contrast: the noise in actual results (profit per unit staked, per bet).
    result_sd = float((graded.profit_units / graded.stake).std(ddof=1)) if len(graded) > 1 else None
    out = {"games": ci["n"], "mean": _r(ci["mean"]), "sd": _r(ci["sd"]), "lo": _r(ci["lo"]), "hi": _r(ci["hi"]),
           "games_needed": ci["n_needed"], "result_sd": _r(result_sd), "bets_needed_by_results": None}
    notes = []
    if ci["n"] < 2:
        notes.append("Not enough closed picks yet for a confidence interval (need at least 2 games).")
    else:
        notes.append(f"Average CLV is {ci['mean']:+.1%} per unit over {ci['n']} games; the 95% confidence interval "
                     f"is {ci['lo']:+.1%} to {ci['hi']:+.1%}.")
        if ci["lo"] > 0:
            notes.append("The whole interval is above zero: this is unlikely to be luck.")
        elif ci["hi"] < 0:
            notes.append("The whole interval is below zero: these picks are losing to the closing line.")
        else:
            notes.append("The interval still includes zero, so this could be noise.")
        if ci["n_needed"]:
            more = max(ci["n_needed"] - ci["n"], 0)
            notes.append(f"If the average and spread stay as they are, about {ci['n_needed']} games are needed "
                         f"for it to be statistically meaningful" + (f" ({more} more)." if more else " (already there)."))
            if result_sd:
                by_results = math.ceil((1.96 * result_sd / ci["mean"]) ** 2)
                out["bets_needed_by_results"] = by_results
                notes.append(f"Judging the same edge from wins and losses alone would take about {by_results:,} bets, "
                             f"which is why CLV is the faster signal.")
        if ci["n"] < 30:
            notes.append("Under about 30 games the interval itself is rough; treat it as a guide.")
    out["notes"] = notes
    return out


def _timeline(df: pd.DataFrame) -> list[dict]:
    """Cumulative expected units (stake x CLV EV) and actual units, in kickoff order, over picks
    that are both closed and graded."""
    d = df[df.clv_ev.notna() & (df.result != "")].sort_values(["kickoff_utc", "pick_id"])
    exp, act = (d.stake * d.clv_ev).cumsum(), d.profit_units.cumsum()
    return [{"n": i + 1, "week": int(w), "label": f"W{int(w)} {s} {t}", "cum_clv": _r(e, 3), "cum_units": _r(a, 3)}
            for i, (w, s, t, e, a) in enumerate(zip(d.week, d.side, d.bet_type, exp, act))]


def _table(df: pd.DataFrame, by) -> list[dict]:
    return [{"key": k if isinstance(k, str) else int(k), **_stats(g)} for k, g in df.groupby(by, sort=True)]


def build_report(picks: pd.DataFrame) -> dict:
    tracks = {}
    for track, df in picks.groupby("track", sort=True):
        df = df.assign(edge_bucket=[_edge_bucket(r) for r in df.itertuples()],
                       season_week=df.season.astype(int) * 100 + df.week.astype(int))
        weeks = _table(df, "season_week")
        for w in weeks:
            w["season"], w["week"] = divmod(w.pop("key"), 100)
        rows = df.sort_values(["kickoff_utc", "pick_id"], ascending=[False, True])
        tracks[track] = {
            "summary": _stats(df), "sample": _sample_size(df),
            "by_type": _table(df, "bet_type"), "by_week": weeks,
            "by_edge": sorted(_table(df, "edge_bucket"), key=lambda r: EDGE_ORDER.index(r["key"])),
            "timeline": _timeline(df),
            "picks": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                      for r in rows.drop(columns=["edge_bucket", "season_week"]).to_dict("records")],
        }
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "tracks": tracks}


def _pct(v, signed=True):
    return "   n/a" if v is None else (f"{v:+.1%}" if signed else f"{v:.0%}")


def format_report(report: dict) -> str:
    out = []
    for track, t in report["tracks"].items():
        s = t["summary"]
        out += [f"\n=== CLV: {track} picks ===",
                f"{s['n']} picks, {s['n_closed']} with a closing line, {s['n_graded']} graded",
                f"Average CLV EV {_pct(s['avg_clv_ev'])}   beat the close {_pct(s['beat_close_pct'], False)}"
                + (f"   avg {s['avg_clv_points']:+.2f} pts on spreads/totals" if s["avg_clv_points"] is not None else ""),
                f"Record {s['wins']}-{s['losses']}-{s['pushes']}   {s['units'] or 0:+.2f} units"
                + (f"   ROI {s['roi']:+.1%}" if s["roi"] is not None else "")
                + (f"   (CLV expected {s['expected_units']:+.2f} units on the same bets)" if s["expected_units"] is not None else "")]
        out += ["  " + n for n in t["sample"]["notes"]]
        for title, rows, key in (("By bet type", t["by_type"], lambda r: r["key"]),
                                 ("By week", t["by_week"], lambda r: f"{r['season']} week {r['week']}"),
                                 ("By model edge", t["by_edge"], lambda r: r["key"])):
            out.append(f"\n{title:<28}{'picks':>6}{'CLV EV':>9}{'CLV pts':>9}{'beat':>7}{'record':>10}{'units':>8}")
            for r in rows:
                pts = "" if r["avg_clv_points"] is None else f"{r['avg_clv_points']:+.2f}"
                units = "" if r["units"] is None else f"{r['units']:+.2f}"
                out.append(f"{key(r):<28}{r['n']:>6}{_pct(r['avg_clv_ev']):>9}{pts:>9}{_pct(r['beat_close_pct'], False):>7}"
                           f"{str(r['wins']) + '-' + str(r['losses']) + '-' + str(r['pushes']):>10}{units:>8}")
    return "\n".join(out) if out else "No picks logged yet."


def write_report(quiet: bool = False) -> dict:
    report = build_report(load_picks())
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if not quiet:
        print(format_report(report))
        print(f"\nCLV report written: {REPORT_JSON}")
    return report
