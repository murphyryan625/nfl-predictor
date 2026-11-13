"""The weekly workflow: snapshot the market -> log picks -> capture the close -> grade.

Storage is two CSV files in data/ (see config.py):

  clv_picks.csv            one row per bet: what I took, what it closed at, CLV, result
  clv_odds_snapshots.csv   append-only log of the market each time the pipeline ran

Why CSV and not SQLite: this repo's data is committed to git by GitHub Actions. CSV diffs line by
line, so every pick and every later edit to it is visible in history and merges cleanly; a SQLite
file is a binary blob that git can't diff or merge. At a few hundred rows a season, nothing here
needs a database.

A pick is logged ONCE, the first time a game has both a model prediction and a market line, and is
then frozen like a real bet would be. Later model re-runs (Thursday/Sunday refreshes) do not move it.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import markets
from .config import CLOSE_WINDOW_MINUTES, DEFAULT_STAKE, PICKS_CSV, SNAPSHOTS_CSV
from .odds_math import (american_to_decimal, grade_bet, moneyline_clv, no_vig_probs, profit_units,
                        spread_clv, total_clv)

UTC = dt.timezone.utc

PICK_COLUMNS = [
    # --- the bet, fixed when it is logged ---
    "pick_id", "track", "source", "game_id", "season", "week", "kickoff_utc", "home_team", "away_team",
    "bet_type",            # spread | moneyline | total
    "side",                # team abbreviation, or over / under
    "line",                # spread: handicap of my side (-3.5 = laying 3.5). total: the total. moneyline: blank
    "price",               # American odds I could get
    "book", "line_ts",     # where and when (UTC) that price was available
    "stake",               # units
    "model_number",        # spread: the model's fair handicap for my side. moneyline: model win prob of my side
    "model_edge",          # spread: points the model disagrees with the market by. moneyline: model prob - no-vig market prob
    "market_fair_prob",    # no-vig market probability of my side when the pick was logged
    # --- the close ---
    "close_line", "close_price", "close_fair_prob", "close_source", "close_ts",
    "clv_points",          # spread/total only: points better than the close
    "clv_prob",            # spread/total: cover-probability gained. moneyline: fair close prob - my implied prob
    "clv_ev",              # expected profit per unit staked if the no-vig close is right (comparable across bet types)
    "beat_close",          # 1 = better number/price than the close, 0 = worse, 0.5 = identical
    # --- the result ---
    "home_score", "away_score", "result", "profit_units",
]
TEXT_COLUMNS = ["pick_id", "track", "source", "game_id", "kickoff_utc", "home_team", "away_team", "bet_type",
                "side", "book", "line_ts", "close_source", "close_ts", "result"]
NUMERIC_COLUMNS = [c for c in PICK_COLUMNS if c not in TEXT_COLUMNS]
SNAPSHOT_TEXT = ["ts_utc", "game_id", "source"] + [c for c in markets.SNAPSHOT_COLUMNS if c.endswith("_book")]


def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


# ----------------------------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------------------------

def _read(path, columns, text_columns) -> pd.DataFrame:
    if not path.exists():
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)   # everything as text first: no surprises
    for c in columns:
        if c not in df:
            df[c] = ""
        df[c] = df[c].astype(object) if c in text_columns else pd.to_numeric(df[c], errors="coerce")
    return df[columns]


def load_picks() -> pd.DataFrame:
    return _read(PICKS_CSV, PICK_COLUMNS, TEXT_COLUMNS)


def save_picks(picks: pd.DataFrame) -> None:
    PICKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = picks[PICK_COLUMNS].sort_values(["kickoff_utc", "game_id", "bet_type", "pick_id"])
    out.to_csv(PICKS_CSV, index=False, float_format="%.6g")


def load_snapshots() -> pd.DataFrame:
    return _read(SNAPSHOTS_CSV, markets.SNAPSHOT_COLUMNS, SNAPSHOT_TEXT)


def append_snapshots(rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=markets.SNAPSHOT_COLUMNS)
    if rows:
        SNAPSHOTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(SNAPSHOTS_CSV, mode="a", index=False, header=not SNAPSHOTS_CSV.exists(), float_format="%.6g")
    return new


def _num(v):
    """A float, or None for blank/NaN."""
    return None if v is None or pd.isna(v) else float(v)


# ----------------------------------------------------------------------------------------------
# Model predictions (read-only view of the predictor's ledger)
# ----------------------------------------------------------------------------------------------

def model_games() -> pd.DataFrame:
    """Live predictions from data/predictions.csv. This is the only place the tracker touches the
    model's output; rows labelled "backfill" (predicted after the fact) are left out."""
    from nflpred.track import load_ledger
    led = load_ledger()
    if led.empty:
        return pd.DataFrame(columns=["game_id", "season", "week", "kickoff_utc", "home_team", "away_team",
                                     "pred_margin", "home_win_prob"])
    led = led[led.run != "backfill"].copy()
    led["kickoff_utc"] = pd.to_datetime(led.kickoff_ct, utc=True)
    return led[["game_id", "season", "week", "kickoff_utc", "home_team", "away_team", "pred_margin", "home_win_prob"]]


# ----------------------------------------------------------------------------------------------
# 1. Snapshot the market
# ----------------------------------------------------------------------------------------------

def take_snapshot(games: pd.DataFrame, now: dt.datetime | None = None, events: list | None = None,
                  schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Record the current market for every game in `games` that hasn't kicked off.

    Uses The Odds API when ODDS_API_KEY is set; any game it doesn't cover (or every game, when
    there is no key or the call fails) falls back to the nflverse line. `events` and `schedule`
    can be passed in directly (tests, or to avoid a second download).
    """
    now = now or now_utc()
    upcoming = games[games.kickoff_utc > now]
    if upcoming.empty:
        return append_snapshots([])
    rows = []
    if events is None and markets.api_key():
        try:
            events = markets.fetch_odds_api(markets.api_key())
        except RuntimeError as e:
            print(f"  warning: {e}; using nflverse lines instead")
    if events:
        rows = markets.odds_api_snapshots(events, upcoming, now)
    missing = set(upcoming.game_id) - {r["game_id"] for r in rows}
    if missing:
        schedule = markets.load_market_schedule() if schedule is None else schedule
        rows += markets.nflverse_snapshots(schedule, missing, now)
    snaps = append_snapshots(rows)
    by_source = snaps.source.value_counts().to_dict() if len(snaps) else {}
    print(f"Market snapshot: {len(snaps)} of {len(upcoming)} upcoming games {by_source or ''}")
    return snaps


# ----------------------------------------------------------------------------------------------
# 2. Log picks
# ----------------------------------------------------------------------------------------------

def _new_pick(game, bet_type, side, line, price, book, ts, stake, track, source, pick_id,
              model_number=None, model_edge=None, market_fair_prob=None) -> dict:
    row = {c: (None if c in NUMERIC_COLUMNS else "") for c in PICK_COLUMNS}
    row.update(pick_id=pick_id, track=track, source=source, game_id=game.game_id, season=int(game.season),
               week=int(game.week), kickoff_utc=pd.Timestamp(game.kickoff_utc).isoformat(),
               home_team=game.home_team, away_team=game.away_team, bet_type=bet_type, side=side,
               line=line, price=float(price), book=book or "", line_ts=ts, stake=float(stake),
               model_number=model_number, model_edge=model_edge, market_fair_prob=market_fair_prob)
    return row


def model_picks_for_game(game, snap) -> list[dict]:
    """The model's side of the spread and of the moneyline for one game, priced at `snap`.

    The line and prices are the ones at my books (pick_home_line, best_*; see config.PICK_BOOKS).
    Spread.    The model predicts the home team wins by pred_margin; the market's number is -home_line.
               Model above the market -> take the home side, below -> the away side.
                   model_edge = |pred_margin + home_line|   (points of disagreement)
    Moneyline. Take whichever side the model rates above the no-vig market probability.
                   model_edge = model win prob - no-vig market prob   (of the side taken)
    A game is skipped for a bet type only when model and market agree exactly, or there is no price.
    """
    picks, ts = [], snap["ts_utc"]
    consensus_line = _num(snap.get("home_line"))
    home_line = _num(snap.get("pick_home_line"))
    home_line = consensus_line if home_line is None else home_line
    if home_line is not None:
        diff = float(game.pred_margin) + home_line
        if diff != 0:
            home = diff > 0
            who = "home" if home else "away"
            price, book = _num(snap.get(f"best_{who}_spread_price")), snap.get(f"best_{who}_spread_book")
            if price is None:                      # line known but no price published: assume standard juice
                price, book = -110.0, "assumed -110"
            # The no-vig cover probability describes the consensus line; blank if my book hangs a different one.
            fair_home = _num(snap.get("home_cover_fair")) if home_line == consensus_line else None
            picks.append(_new_pick(
                game, "spread", game.home_team if home else game.away_team,
                home_line if home else -home_line, price, book, ts, DEFAULT_STAKE, "model", "auto",
                f"{game.game_id}:spread:model",
                model_number=round(-float(game.pred_margin) if home else float(game.pred_margin), 2),
                model_edge=round(abs(diff), 2),
                market_fair_prob=None if fair_home is None else (fair_home if home else 1 - fair_home)))
    fair_home = _num(snap.get("home_ml_fair"))
    if fair_home is not None:
        edge_home = float(game.home_win_prob) - fair_home
        home = edge_home > 0
        price = _num(snap.get(f"best_{'home' if home else 'away'}_ml"))
        if edge_home != 0 and price is not None:
            picks.append(_new_pick(
                game, "moneyline", game.home_team if home else game.away_team, None, price,
                snap.get(f"best_{'home' if home else 'away'}_ml_book"), ts, DEFAULT_STAKE, "model", "auto",
                f"{game.game_id}:moneyline:model",
                model_number=round(float(game.home_win_prob) if home else 1 - float(game.home_win_prob), 4),
                model_edge=round(abs(edge_home), 4),
                market_fair_prob=round(fair_home if home else 1 - fair_home, 4)))
    return picks


def log_picks(games: pd.DataFrame, snaps: pd.DataFrame, now: dt.datetime | None = None, replace: bool = False) -> int:
    """Log the model's picks for every upcoming game that doesn't have them yet.

    replace=True re-logs picks for games that haven't kicked off (new model number, current line).
    Picks for games that have started are never touched.
    """
    now = now or now_utc()
    picks = load_picks()
    latest = snaps.sort_values("ts_utc").drop_duplicates("game_id", keep="last").set_index("game_id") if len(snaps) else snaps
    new = []
    for game in games[games.kickoff_utc > now].itertuples():
        if game.game_id not in latest.index:
            continue
        for p in model_picks_for_game(game, latest.loc[game.game_id].to_dict() | {"game_id": game.game_id}):
            if replace or p["pick_id"] not in set(picks.pick_id):
                new.append(p)
    if new:
        picks = picks[~picks.pick_id.isin({p["pick_id"] for p in new})]
        save_picks(pd.concat([picks, pd.DataFrame(new, columns=PICK_COLUMNS)], ignore_index=True)
                   if len(picks) else pd.DataFrame(new, columns=PICK_COLUMNS))
    print(f"Picks logged: {len(new)} new")
    return len(new)


def _schedule_game(schedule: pd.DataFrame, game_id: str):
    g = schedule[schedule.game_id == game_id]
    if g.empty:
        raise ValueError(f"unknown game_id {game_id!r} (format: 2026_10_BUF_NYJ = season_week_away_home)")
    return g.iloc[0]


def _model_view(game_id: str, bet_type: str, side_is_home: bool, line) -> tuple[float | None, float | None]:
    """(model_number, model_edge) for a hand-entered bet, when the model predicted that game.
    The spread edge is signed here: negative means I bet against the model's number."""
    try:
        g = model_games()
    except Exception:
        return None, None
    g = g[g.game_id == game_id]
    if g.empty or bet_type == "total":
        return None, None
    g = g.iloc[0]
    if bet_type == "spread":
        side_margin = float(g.pred_margin) if side_is_home else -float(g.pred_margin)
        return round(-side_margin, 2), round(side_margin + float(line), 2)
    return round(float(g.home_win_prob) if side_is_home else 1 - float(g.home_win_prob), 4), None


def add_pick(schedule: pd.DataFrame, game_id: str, bet_type: str, side: str, price: float, line: float | None = None,
             book: str = "", stake: float = DEFAULT_STAKE, line_ts: str | None = None, track: str = "placed",
             source: str = "manual", model_number=None, model_edge=None, picks: pd.DataFrame | None = None) -> dict:
    """Build one hand-entered (or backfilled) pick and validate it. Does not save."""
    game = _schedule_game(schedule, game_id)
    bet_type, side = bet_type.lower(), side.strip()
    if bet_type not in ("spread", "moneyline", "total"):
        raise ValueError(f"bet type must be spread, moneyline or total: {bet_type}")
    if bet_type == "total":
        side = side.lower()
        if side not in ("over", "under"):
            raise ValueError("a total bet's side must be 'over' or 'under'")
    else:
        side = side.upper()
        if side not in (game.home_team, game.away_team):
            raise ValueError(f"side must be {game.away_team} or {game.home_team} for {game_id}")
    if bet_type != "moneyline" and (line is None or pd.isna(line)):
        raise ValueError(f"a {bet_type} bet needs a line")
    american_to_decimal(price)                                  # raises on a bad price
    line = None if bet_type == "moneyline" else float(line)
    if model_number is None or pd.isna(model_number):
        model_number, auto_edge = _model_view(game_id, bet_type, side == game.home_team, line)
        model_edge = auto_edge if model_edge is None or pd.isna(model_edge) else model_edge
    existing = load_picks() if picks is None else picks
    n = 1 + int(existing.pick_id.str.startswith(f"{game_id}:{bet_type}:{track}:").sum())
    return _new_pick(game, bet_type, side, line, price, book, line_ts or now_utc().isoformat(timespec="seconds"),
                     stake, track, source, f"{game_id}:{bet_type}:{track}:{n}",
                     model_number=model_number, model_edge=model_edge)


# ----------------------------------------------------------------------------------------------
# 3. Capture the close and compute CLV
# ----------------------------------------------------------------------------------------------

def apply_close(pick: dict, close_line, close_price, close_fair_prob: float, source: str, ts: str) -> dict:
    """Fill in the closing fields and the CLV numbers for one pick (formulas are in odds_math.py).

    close_line / close_price / close_fair_prob are all for MY side: the closing handicap (or total),
    the closing price with vig, and the no-vig closing probability.
    """
    price, bet_type = float(pick["price"]), pick["bet_type"]
    if bet_type == "moneyline":
        ev = moneyline_clv(price, close_fair_prob)
        points, prob = None, close_fair_prob - 1 / american_to_decimal(price)
    else:
        r = (spread_clv(float(pick["line"]), price, float(close_line), close_fair_prob) if bet_type == "spread"
             else total_clv(pick["side"], float(pick["line"]), price, float(close_line), close_fair_prob))
        points, prob, ev = r["points"], r["prob"], r["ev"]
    # "Beat the close": a better number; if the number is the same (or it's a moneyline), a better price.
    if points:
        beat = 1.0 if points > 0 else 0.0
    elif close_price is not None and not pd.isna(close_price):
        mine, close = american_to_decimal(price), american_to_decimal(close_price)
        beat = 0.5 if abs(mine - close) < 1e-9 else float(mine > close)
    else:
        beat = 0.5 if abs(ev) < 1e-9 else float(ev > 0)
    pick.update(close_line=None if bet_type == "moneyline" else float(close_line),
                close_price=_num(close_price), close_fair_prob=round(close_fair_prob, 4),
                close_source=source, close_ts=ts,
                clv_points=points, clv_prob=round(prob, 4), clv_ev=round(ev, 4), beat_close=beat)
    return pick


def _close_from_snapshot(pick: dict, snap: dict) -> tuple | None:
    """(close_line, close_price, close_fair_prob) for the pick's side out of a home-perspective snapshot."""
    home = pick["side"] == pick["home_team"]
    if pick["bet_type"] == "spread":
        line, fair = _num(snap.get("home_line")), _num(snap.get("home_cover_fair"))
        if line is None:
            return None
        fair = 0.5 if fair is None else fair
        return (line if home else -line, _num(snap.get("home_spread_price" if home else "away_spread_price")),
                fair if home else 1 - fair)
    if pick["bet_type"] == "moneyline":
        fair = _num(snap.get("home_ml_fair"))
        if fair is None:
            return None
        return None, _num(snap.get("home_ml" if home else "away_ml")), fair if home else 1 - fair
    return None


def capture_close(now: dt.datetime | None = None, schedule: pd.DataFrame | None = None) -> int:
    """Attach a closing line to every pick whose game has kicked off and doesn't have one yet.

    Order of preference:
      1. the last Odds API snapshot taken within CLOSE_WINDOW_MINUTES before kickoff (a true
         multi-book consensus close, when the pipeline happened to run that late);
      2. once the game is final, the closing line published by nflverse.
    Closes entered by hand (`set-close`) are never overwritten.
    """
    now = now or now_utc()
    picks = load_picks()
    kick = pd.to_datetime(picks.kickoff_utc, utc=True)
    todo = picks[(picks.close_source == "") & (kick <= now)]
    if todo.empty:
        print("Closing lines: nothing to capture")
        return 0
    snaps = load_snapshots()
    snaps = snaps[snaps.source == "oddsapi"].assign(ts=lambda d: pd.to_datetime(d.ts_utc, utc=True))
    schedule = markets.load_market_schedule() if schedule is None else schedule
    final = {r.game_id: r for r in schedule[schedule.home_score.notna()].itertuples()}
    done = 0
    for i, pick in todo.iterrows():
        pick, k, close, source, ts = pick.to_dict(), kick[i], None, "", ""
        near = snaps[(snaps.game_id == pick["game_id"]) & (snaps.ts < k)
                     & (snaps.ts >= k - pd.Timedelta(minutes=CLOSE_WINDOW_MINUTES))]
        if len(near):
            last = near.sort_values("ts").iloc[-1].to_dict()
            close, source, ts = _close_from_snapshot(pick, last), "oddsapi", last["ts_utc"]
        if close is None and pick["game_id"] in final:
            row = final[pick["game_id"]]
            source, ts = "nflverse", pd.Timestamp(k).isoformat()
            if pick["bet_type"] == "total":
                t = markets.nflverse_total(row)
                if t:
                    over = pick["side"] == "over"
                    close = (t["total"], t["over_price"] if over else t["under_price"],
                             t["over_fair"] if over else 1 - t["over_fair"])
            else:
                snap = markets.nflverse_snapshot(row, k)
                close = _close_from_snapshot(pick, snap) if snap else None
        if close is None:
            continue
        for c, v in apply_close(pick, *close, source, ts).items():
            picks.at[i, c] = v
        done += 1
    if done:
        save_picks(picks)
    print(f"Closing lines: captured {done}, still waiting on {len(todo) - done}")
    return done


def set_close(pick_id: str, price: float | None = None, opp_price: float | None = None, line: float | None = None) -> dict:
    """Manual-entry fallback: type in the closing line for one pick.

    price / opp_price are the closing prices on my side and on the other side; both are needed to
    remove the vig. For spreads and totals they may be left out, which assumes a 50/50 close.
    """
    picks = load_picks()
    hit = picks.index[picks.pick_id == pick_id]
    if not len(hit):
        raise ValueError(f"no pick with id {pick_id!r} (see `python -m clv status`)")
    pick = picks.loc[hit[0]].to_dict()
    if pick["bet_type"] == "moneyline":
        if price is None or opp_price is None:
            raise ValueError("a moneyline close needs both --price and --opp-price")
    elif line is None:
        raise ValueError(f"a {pick['bet_type']} close needs --line")
    fair = no_vig_probs(price, opp_price)[0] if price is not None and opp_price is not None else 0.5
    pick = apply_close(pick, line, price, fair, "manual", now_utc().isoformat(timespec="seconds"))
    for c, v in pick.items():
        picks.at[hit[0], c] = v
    save_picks(picks)
    return pick


# ----------------------------------------------------------------------------------------------
# 4. Grade
# ----------------------------------------------------------------------------------------------

def grade(schedule: pd.DataFrame | None = None, now: dt.datetime | None = None) -> int:
    """Fill in final scores, W/L/P and profit in units for every finished, ungraded pick."""
    now = now or now_utc()
    picks = load_picks()
    todo = picks[(picks.result == "") & (pd.to_datetime(picks.kickoff_utc, utc=True) <= now)]
    if todo.empty:
        print("Grading: nothing to grade")
        return 0
    schedule = markets.load_market_schedule() if schedule is None else schedule
    final = schedule[schedule.home_score.notna()].set_index("game_id")
    done = 0
    for i, p in todo.iterrows():
        if p.game_id not in final.index:
            continue
        hs, as_ = float(final.loc[p.game_id].home_score), float(final.loc[p.game_id].away_score)
        result = grade_bet(p.bet_type, p.side == p.home_team, p.side, _num(p.line), hs, as_)
        picks.at[i, "home_score"], picks.at[i, "away_score"] = hs, as_
        picks.at[i, "result"] = result
        picks.at[i, "profit_units"] = round(profit_units(result, p.price, p.stake), 4)
        done += 1
    if done:
        save_picks(picks)
    print(f"Grading: {done} picks graded, {len(todo) - done} games not final yet")
    return done


# ----------------------------------------------------------------------------------------------
# Backfill
# ----------------------------------------------------------------------------------------------

BACKFILL_REQUIRED = ["game_id", "bet_type", "side", "price"]
BACKFILL_OPTIONAL = ["line", "book", "line_ts", "stake", "track", "model_number", "model_edge",
                     "close_line", "close_price", "close_opp_price"]


def backfill(csv_path, schedule: pd.DataFrame | None = None) -> int:
    """Load past picks from a CSV (see clv/backfill_template.csv for the columns).

    Rows that include a closing line get their CLV straight away; the rest pick up the nflverse
    closing line on the next `close` step. Everything is then graded as usual. Re-loading the same
    file does not create duplicates (same game, bet type, side, line and price = same bet).
    """
    rows = pd.read_csv(csv_path)
    lack = [c for c in BACKFILL_REQUIRED if c not in rows]
    if lack:
        raise ValueError(f"{csv_path} is missing column(s): {', '.join(lack)}")
    for c in BACKFILL_OPTIONAL:
        if c not in rows:
            rows[c] = None
    schedule = markets.load_market_schedule() if schedule is None else schedule
    picks, added = load_picks(), 0
    seen = {(p.game_id, p.bet_type, p.side, _num(p.line), p.price) for p in picks.itertuples()}
    for n, r in enumerate(rows.itertuples(), start=2):
        try:
            kickoff = pd.Timestamp(_schedule_game(schedule, r.game_id).kickoff_utc).isoformat()
            pick = add_pick(schedule, r.game_id, r.bet_type, str(r.side), r.price, line=_num(r.line),
                            book="" if pd.isna(r.book) else str(r.book),
                            stake=DEFAULT_STAKE if pd.isna(r.stake) else float(r.stake),
                            line_ts=kickoff if pd.isna(r.line_ts) else str(r.line_ts),   # unknown time: not after kickoff
                            track="model" if pd.isna(r.track) else str(r.track), source="backfill",
                            model_number=_num(r.model_number), model_edge=_num(r.model_edge), picks=picks)
            key = (pick["game_id"], pick["bet_type"], pick["side"], pick["line"], pick["price"])
            if key in seen:
                continue
            cl, cp, co = _num(r.close_line), _num(r.close_price), _num(r.close_opp_price)
            if (cp is not None and co is not None) if pick["bet_type"] == "moneyline" else cl is not None:
                fair = no_vig_probs(cp, co)[0] if cp is not None and co is not None else 0.5
                pick = apply_close(pick, cl, cp, fair, "backfill", pick["kickoff_utc"])
        except ValueError as e:
            raise ValueError(f"{csv_path} line {n}: {e}") from None
        seen.add(key)
        picks = pd.concat([picks, pd.DataFrame([pick], columns=PICK_COLUMNS)], ignore_index=True) if len(picks) \
            else pd.DataFrame([pick], columns=PICK_COLUMNS)
        added += 1
    if added:
        save_picks(picks)
    print(f"Backfill: {added} picks added, {len(rows) - added} already present")
    return added


# ----------------------------------------------------------------------------------------------
# Pipeline entry point (called from run.py)
# ----------------------------------------------------------------------------------------------

def run(stage: str = "weekly", replace: bool = False) -> None:
    """stage "predict" / "weekly": snapshot -> log -> close -> grade -> report.
       stage "score":              close -> grade -> report (no new picks)."""
    from .report import write_report
    now = now_utc()
    schedule = markets.load_market_schedule()
    if stage != "score":
        games = model_games()
        log_picks(games, take_snapshot(games, now, schedule=schedule), now, replace=replace)
    capture_close(now, schedule)
    grade(schedule, now)
    write_report()
