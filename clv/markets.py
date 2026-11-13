"""Where market lines come from. Two sources, both reduced to the same "snapshot" row per game:

* The Odds API (needs the ODDS_API_KEY environment variable): many books at once, so the snapshot
  holds a consensus line, a consensus no-vig probability, and the best price available.
* nflverse (free, no key; the same schedule file the model already downloads): one line per game.
  It tracks the current line before kickoff and is the published closing line afterwards.

Snapshot fields (home-team perspective; the away side is the mirror image)
    home_line              consensus spread as the home team's handicap (-3.5 = home lays 3.5)
    home_spread_price      typical (median) price on the home side at that line; away_spread_price likewise
    home_cover_fair        no-vig probability the home team covers home_line
    home_ml / away_ml      typical (median) moneyline prices
    home_ml_fair           no-vig probability the home team wins
    pick_home_line         the spread at the books a pick is priced at (config.PICK_BOOKS, or the
                           PROXY_BOOKS median when those are not in the feed); usually = home_line
    best_*_price / _book   the price a pick on that side is logged at, and which book it is from
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import config
from .config import ODDS_API_KEY_ENV, ODDS_API_MARKETS, ODDS_API_REGIONS, ODDS_API_URL
from .odds_math import american_to_decimal, decimal_to_american, no_vig_probs

UTC = dt.timezone.utc
SNAPSHOT_COLUMNS = [
    "ts_utc", "game_id", "source", "n_books",
    "home_line", "home_spread_price", "away_spread_price", "home_cover_fair",
    "home_ml", "away_ml", "home_ml_fair",
    "pick_home_line", "best_home_spread_price", "best_home_spread_book", "best_away_spread_price", "best_away_spread_book",
    "best_home_ml", "best_home_ml_book", "best_away_ml", "best_away_ml_book",
]

# The Odds API uses full team names; the model's ledger uses nflverse abbreviations.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}   # same fix nflpred/data.py applies


# ----------------------------------------------------------------------------------------------
# nflverse
# ----------------------------------------------------------------------------------------------

def load_market_schedule() -> pd.DataFrame:
    """The nflverse schedule WITH its betting columns (nflpred/data.py drops them on purpose).
    Adds kickoff_utc. Used for final scores, closing lines, and as the keyless line source."""
    import nflreadpy as nfl
    from zoneinfo import ZoneInfo
    s = nfl.load_schedules().to_pandas()
    for c in ("home_team", "away_team"):
        s[c] = s[c].replace(TEAM_FIX)
    eastern = ZoneInfo("America/New_York")           # nflverse lists kickoff in US Eastern time

    def kickoff(day, time):
        hh, mm = (time if isinstance(time, str) and ":" in time else "13:00").split(":")[:2]
        return dt.datetime.combine(pd.Timestamp(day).date(), dt.time(int(hh), int(mm)), tzinfo=eastern).astimezone(UTC)

    s["kickoff_utc"] = [kickoff(d, t) for d, t in zip(s.gameday, s.gametime)]
    return s


def _price(v):
    """A usable American price, or None (nflverse leaves blanks, and occasionally a 0)."""
    return float(v) if pd.notna(v) and abs(float(v)) >= 100 else None


def nflverse_snapshot(row, ts: dt.datetime) -> dict | None:
    """One schedule row -> a snapshot. nflverse's spread_line is "points the home team is favoured
    by", so the home handicap is its negative (spread_line 9.5 -> home_line -9.5)."""
    if pd.isna(row.spread_line) and pd.isna(row.home_moneyline):
        return None
    hs, as_ = _price(row.home_spread_odds), _price(row.away_spread_odds)
    hm, am = _price(row.home_moneyline), _price(row.away_moneyline)
    has_line = pd.notna(row.spread_line)
    return {
        "ts_utc": ts.isoformat(timespec="seconds"), "game_id": row.game_id, "source": "nflverse", "n_books": 1,
        "home_line": -float(row.spread_line) if has_line else None,
        "home_spread_price": hs, "away_spread_price": as_,
        # No prices published -> assume the standard -110 both ways, i.e. a 50/50 line.
        "home_cover_fair": (no_vig_probs(hs, as_)[0] if hs and as_ else 0.5) if has_line else None,
        "home_ml": hm, "away_ml": am,
        "home_ml_fair": no_vig_probs(hm, am)[0] if hm and am else None,
        "pick_home_line": -float(row.spread_line) if has_line else None,
        "best_home_spread_price": hs, "best_home_spread_book": "nflverse" if hs else None,
        "best_away_spread_price": as_, "best_away_spread_book": "nflverse" if as_ else None,
        "best_home_ml": hm, "best_home_ml_book": "nflverse" if hm else None,
        "best_away_ml": am, "best_away_ml_book": "nflverse" if am else None,
    }


def nflverse_snapshots(schedule: pd.DataFrame, game_ids, ts: dt.datetime) -> list[dict]:
    rows = schedule[schedule.game_id.isin(list(game_ids))]
    return [s for s in (nflverse_snapshot(r, ts) for r in rows.itertuples()) if s]


def nflverse_total(row) -> dict | None:
    """Closing total for manually logged total bets: line plus the no-vig probability of the over."""
    if pd.isna(row.total_line):
        return None
    o, u = _price(row.over_odds), _price(row.under_odds)
    return {"total": float(row.total_line), "over_price": o, "under_price": u,
            "over_fair": no_vig_probs(o, u)[0] if o and u else 0.5}


# ----------------------------------------------------------------------------------------------
# The Odds API
# ----------------------------------------------------------------------------------------------

def api_key() -> str | None:
    return os.environ.get(ODDS_API_KEY_ENV) or None


def fetch_odds_api(key: str, timeout: int = 30) -> list[dict]:
    """One call for every upcoming NFL game. Costs 2 of the free tier's 500 monthly credits."""
    query = urllib.parse.urlencode({"apiKey": key, "regions": ODDS_API_REGIONS, "markets": ODDS_API_MARKETS,
                                    "oddsFormat": "american", "dateFormat": "iso"})
    try:
        with urllib.request.urlopen(f"{ODDS_API_URL}?{query}", timeout=timeout) as r:
            left = r.headers.get("x-requests-remaining")
            events = json.load(r)
    except urllib.error.HTTPError as e:      # never echo the URL: it contains the key
        raise RuntimeError(f"The Odds API returned HTTP {e.code} ({e.reason})") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"couldn't reach The Odds API ({e.reason})") from None
    print(f"  The Odds API: {len(events)} games" + (f", {left} credits left this month" if left else ""))
    return events


def _median_price(prices: list[float]) -> int:
    """Median taken in decimal odds (American odds jump from -100 to +100, so they don't average)."""
    return decimal_to_american(statistics.median(american_to_decimal(p) for p in prices))


def _best(offers: list[tuple[float, str]]) -> tuple[float, str]:
    """Highest-paying (price, book)."""
    return max(offers, key=lambda o: american_to_decimal(o[0]))


def _common_line(spreads: list[tuple]) -> float:
    """The handicap most books are hanging; ties go to the one nearest the median."""
    points = [s[1] for s in spreads]
    med = statistics.median(points)
    counts = {p: points.count(p) for p in set(points)}
    return max(counts, key=lambda p: (counts[p], -abs(p - med)))


def _pick_price(offers: list[tuple[float, str]], use_best: bool) -> tuple[float, str]:
    """The (price, label) a pick is logged at: the best price among my books, or the proxy median."""
    if use_best:
        return _best(offers)
    return _median_price([o[0] for o in offers]), f"median of {len(offers)} US books (proxy)"


def consensus(event: dict) -> dict | None:
    """Boil one Odds API event (many bookmakers) down to the snapshot fields.

    Market view (used for the closing line), from EVERY book:
      Spread: the consensus line is the handicap most books are hanging. Prices and the no-vig cover
      probability use only the books at that line.
      Moneyline: every book's two prices are de-vigged separately, then averaged.
    Pick price (what a pick is logged at), from config.PICK_BOOKS when the feed has them (their line,
    best price among them), otherwise the median of config.PROXY_BOOKS, labelled as a proxy.
    """
    home, away = event["home_team"], event["away_team"]
    spreads, mls = [], []        # (book, home_point, home_price, away_price) / (book, home_price, away_price)
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            o = {x["name"]: x for x in market.get("outcomes", [])}
            if home not in o or away not in o:
                continue
            try:
                if market["key"] == "spreads":
                    spreads.append((book["title"], float(o[home]["point"]), float(o[home]["price"]), float(o[away]["price"])))
                elif market["key"] == "h2h":
                    mls.append((book["title"], float(o[home]["price"]), float(o[away]["price"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not spreads and not mls:
        return None
    out = {k: None for k in SNAPSHOT_COLUMNS[4:]}
    out["n_books"] = max(len(spreads), len(mls))
    def mine(rows):
        """(rows a pick may be priced at, True if they are my own books rather than the proxy set)"""
        own = [r for r in rows if r[0] in config.PICK_BOOKS]
        return (own, True) if own else ([r for r in rows if r[0] in config.PROXY_BOOKS] or rows, False)

    if spreads:
        line = _common_line(spreads)
        at = [s for s in spreads if s[1] == line]
        rows, own = mine(spreads)
        pick_line = _common_line(rows)
        rows = [s for s in rows if s[1] == pick_line]
        bh, ba = _pick_price([(s[2], s[0]) for s in rows], own), _pick_price([(s[3], s[0]) for s in rows], own)
        out.update(home_line=line, pick_home_line=pick_line,
                   home_spread_price=_median_price([s[2] for s in at]),
                   away_spread_price=_median_price([s[3] for s in at]),
                   home_cover_fair=statistics.fmean(no_vig_probs(s[2], s[3])[0] for s in at),
                   best_home_spread_price=bh[0], best_home_spread_book=bh[1],
                   best_away_spread_price=ba[0], best_away_spread_book=ba[1])
    if mls:
        rows, own = mine(mls)
        bh, ba = _pick_price([(m[1], m[0]) for m in rows], own), _pick_price([(m[2], m[0]) for m in rows], own)
        out.update(home_ml=_median_price([m[1] for m in mls]), away_ml=_median_price([m[2] for m in mls]),
                   home_ml_fair=statistics.fmean(no_vig_probs(m[1], m[2])[0] for m in mls),
                   best_home_ml=bh[0], best_home_ml_book=bh[1], best_away_ml=ba[0], best_away_ml_book=ba[1])
    return out


def _flip(snap: dict) -> dict:
    """Swap home and away (for neutral-site games the API and nflverse can disagree on who is home)."""
    f = dict(snap)
    for a, b in (("home_spread_price", "away_spread_price"), ("home_ml", "away_ml"),
                 ("best_home_spread_price", "best_away_spread_price"), ("best_home_spread_book", "best_away_spread_book"),
                 ("best_home_ml", "best_away_ml"), ("best_home_ml_book", "best_away_ml_book")):
        f[a], f[b] = snap[b], snap[a]
    for k in ("home_cover_fair", "home_ml_fair"):
        f[k] = None if snap[k] is None else 1 - snap[k]
    for k in ("home_line", "pick_home_line"):
        f[k] = None if snap[k] is None else -snap[k]
    return f


def odds_api_snapshots(events: list[dict], games: pd.DataFrame, ts: dt.datetime) -> list[dict]:
    """Match API events to our games (same two teams, kickoff within 36 hours) and build snapshots.

    `games` needs game_id, home_team, away_team, kickoff_utc. Games already under way are skipped:
    once a game starts the API serves live in-play odds, which must never be mistaken for a close.
    """
    out = []
    for ev in events:
        try:
            start = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            home, away = TEAM_ABBR[ev["home_team"]], TEAM_ABBR[ev["away_team"]]
        except (KeyError, ValueError):
            continue
        if start <= ts:
            continue
        near = games[(games.kickoff_utc - start).abs() < pd.Timedelta(hours=36)]
        same = near[(near.home_team == home) & (near.away_team == away)]
        swapped = near[(near.home_team == away) & (near.away_team == home)]
        match = same if len(same) else swapped
        snap = consensus(ev)
        if match.empty or snap is None:
            continue
        if not len(same):
            snap = _flip(snap)
        out.append({"ts_utc": ts.isoformat(timespec="seconds"), "game_id": match.iloc[0].game_id,
                    "source": "oddsapi", **snap})
    return out
