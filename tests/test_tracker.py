"""End-to-end tests of the workflow on made-up games: Odds API parsing -> picks -> close -> grade -> report.
No network and no real data files (see conftest.py)."""
import datetime as dt

import pandas as pd
import pytest

from clv import config, markets, tracker
from clv.odds_math import american_to_decimal, no_vig_probs
from clv.report import build_report, format_report

UTC = dt.timezone.utc
KICK = dt.datetime(2026, 11, 15, 18, 0, tzinfo=UTC)
approx = lambda v, tol=1e-4: pytest.approx(v, abs=tol)


@pytest.fixture(autouse=True)
def clean_files(monkeypatch):
    for f in (config.PICKS_CSV, config.SNAPSHOTS_CSV):
        f.unlink(missing_ok=True)
    # Unless a test says otherwise, all three made-up books count as "my books" (best price wins).
    monkeypatch.setattr(config, "PICK_BOOKS", ["DraftKings", "FanDuel", "BetMGM"])
    yield


def book(title, home, away, point, hp, ap, hml, aml):
    return {"key": title.lower(), "title": title, "markets": [
        {"key": "spreads", "outcomes": [{"name": home, "price": hp, "point": point},
                                        {"name": away, "price": ap, "point": -point}]},
        {"key": "h2h", "outcomes": [{"name": home, "price": hml}, {"name": away, "price": aml}]}]}


def event(start=KICK, home="New York Jets", away="Buffalo Bills"):
    return {"id": "x", "commence_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": home, "away_team": away,
            "bookmakers": [book("DraftKings", home, away, 8.5, -110, -110, 330, -425),
                           book("FanDuel", home, away, 8.5, -105, -115, 340, -440),
                           book("BetMGM", home, away, 9.0, -115, -105, 320, -420)]}


def games(pred_margin=-6.0, home_win_prob=0.32):
    return pd.DataFrame([{"game_id": "2026_10_BUF_NYJ", "season": 2026, "week": 10, "kickoff_utc": pd.Timestamp(KICK),
                          "home_team": "NYJ", "away_team": "BUF", "pred_margin": pred_margin,
                          "home_win_prob": home_win_prob}])


def schedule(home_score=None, away_score=None, spread_line=-9.5, hml=380.0, aml=-500.0):
    """A one-game stand-in for the nflverse schedule (spread_line -9.5 = home is a 9.5-point dog)."""
    return pd.DataFrame([{"game_id": "2026_10_BUF_NYJ", "season": 2026, "week": 10, "kickoff_utc": pd.Timestamp(KICK),
                          "home_team": "NYJ", "away_team": "BUF", "home_score": home_score, "away_score": away_score,
                          "spread_line": spread_line, "home_spread_odds": -108.0, "away_spread_odds": -112.0,
                          "home_moneyline": hml, "away_moneyline": aml,
                          "total_line": 44.5, "over_odds": -110.0, "under_odds": -110.0}])


# ---------------------------------------------------------------- market parsing

def test_consensus_line_best_price_and_fair_probability():
    c = markets.consensus(event())
    assert c["home_line"] == 8.5                                   # two of three books
    assert (c["best_home_spread_price"], c["best_home_spread_book"]) == (-105, "FanDuel")
    assert (c["best_away_spread_price"], c["best_away_spread_book"]) == (-110, "DraftKings")
    # fair cover prob = average over the two books AT 8.5 (BetMGM's 9.0 is left out)
    expected = (no_vig_probs(-110, -110)[0] + no_vig_probs(-105, -115)[0]) / 2
    assert c["home_cover_fair"] == approx(expected)
    assert (c["best_home_ml"], c["best_home_ml_book"]) == (340, "FanDuel")
    assert (c["best_away_ml"], c["best_away_ml_book"]) == (-420, "BetMGM")
    ml = [no_vig_probs(a, b)[0] for a, b in ((330, -425), (340, -440), (320, -420))]
    assert c["home_ml_fair"] == approx(sum(ml) / 3)
    assert c["home_ml"] == 330                                     # median price


def test_pick_is_priced_at_my_book_only_when_the_feed_has_it(monkeypatch):
    monkeypatch.setattr(config, "PICK_BOOKS", ["Fanatics"])
    ev = event()
    ev["bookmakers"].append(book("Fanatics", ev["home_team"], ev["away_team"], 9.0, -112, -108, 300, -400))
    c = markets.consensus(ev)
    assert c["home_line"] == 8.5 and c["pick_home_line"] == 9.0          # market says 8.5, my book hangs 9
    assert (c["best_home_spread_price"], c["best_home_spread_book"]) == (-112, "Fanatics")
    assert (c["best_home_ml"], c["best_away_ml"], c["best_away_ml_book"]) == (300, -400, "Fanatics")
    # ...and the pick is logged at that line and price, not at a better number elsewhere.
    snap = {"ts_utc": "t", "game_id": "2026_10_BUF_NYJ", **c}
    p = {x["bet_type"]: x for x in tracker.model_picks_for_game(next(games().itertuples()), snap)}
    assert (p["spread"]["line"], p["spread"]["price"], p["spread"]["book"]) == (9.0, -112, "Fanatics")
    assert p["spread"]["model_edge"] == approx(3.0) and p["spread"]["market_fair_prob"] is None
    assert (p["moneyline"]["price"], p["moneyline"]["book"]) == (300, "Fanatics")


def test_without_my_book_the_pick_uses_the_median_of_regulated_books(monkeypatch):
    monkeypatch.setattr(config, "PICK_BOOKS", ["Fanatics"])
    monkeypatch.setattr(config, "PROXY_BOOKS", ["DraftKings", "FanDuel"])
    ev = event()
    ev["bookmakers"].append(book("BetOnline.ag", ev["home_team"], ev["away_team"], 8.5, 105, -125, 400, -500))
    c = markets.consensus(ev)
    # median of DraftKings (-110, +330) and FanDuel (-105, +340); the offshore +105 / +400 is ignored
    assert c["best_home_spread_price"] == -107 and c["best_home_ml"] == 335
    assert c["best_home_spread_book"] == "median of 2 US books (proxy)"
    assert c["pick_home_line"] == 8.5


def test_events_are_matched_to_games_and_live_games_are_skipped():
    now = KICK - dt.timedelta(days=2)
    snaps = markets.odds_api_snapshots([event()], games(), now)
    assert [s["game_id"] for s in snaps] == ["2026_10_BUF_NYJ"] and snaps[0]["source"] == "oddsapi"
    assert markets.odds_api_snapshots([event()], games(), KICK + dt.timedelta(minutes=5)) == []   # in-play odds
    assert markets.odds_api_snapshots([event(start=KICK + dt.timedelta(days=7))], games(), now) == []


def test_home_and_away_swapped_by_the_api_are_flipped_back():
    now = KICK - dt.timedelta(days=2)
    normal = markets.odds_api_snapshots([event()], games(), now)[0]
    swapped = event(home="Buffalo Bills", away="New York Jets")    # API thinks BUF is home, at the same numbers
    flipped = markets.odds_api_snapshots([swapped], games(), now)[0]
    assert flipped["home_line"] == -normal["home_line"]
    assert flipped["home_ml_fair"] == approx(1 - normal["home_ml_fair"])
    assert flipped["best_away_ml"] == normal["best_home_ml"]


def test_nflverse_snapshot_sign_convention():
    s = markets.nflverse_snapshot(next(schedule().itertuples()), KICK)
    assert s["home_line"] == 9.5                                   # spread_line -9.5 -> home gets 9.5
    assert s["home_ml_fair"] == approx(no_vig_probs(380, -500)[0])
    assert s["home_cover_fair"] == approx(no_vig_probs(-108, -112)[0])


# ---------------------------------------------------------------- the weekly workflow

def run_week(pred_margin=-6.0, home_win_prob=0.32, close_minutes_before=30):
    """Tuesday: snapshot + log. Just before kickoff: another snapshot (the line has moved to NYJ +7.5)."""
    g = games(pred_margin, home_win_prob)
    tue = KICK - dt.timedelta(days=5)
    snaps = tracker.take_snapshot(g, tue, events=[event()])
    tracker.log_picks(g, snaps, tue)
    late = event()
    for b in late["bookmakers"]:
        b["markets"] = book(b["title"], late["home_team"], late["away_team"], 7.5, -110, -110, 280, -350)["markets"]
    tracker.take_snapshot(g, KICK - dt.timedelta(minutes=close_minutes_before), events=[late])
    return g


def test_log_picks_takes_the_models_side_at_the_best_price():
    run_week()
    p = tracker.load_picks().set_index("bet_type")
    # Model: NYJ lose by 6. Market: NYJ +8.5. Model likes NYJ by 2.5 points -> NYJ +8.5 at the best price.
    assert p.loc["spread", "side"] == "NYJ" and p.loc["spread", "line"] == 8.5
    assert p.loc["spread", "price"] == -105 and p.loc["spread", "book"] == "FanDuel"
    assert p.loc["spread", "model_edge"] == approx(2.5) and p.loc["spread", "model_number"] == approx(6.0)
    # Model: NYJ 32%. Market no-vig about 22%. -> NYJ moneyline at the best price, +340.
    assert p.loc["moneyline", "side"] == "NYJ" and p.loc["moneyline", "price"] == 340
    assert p.loc["moneyline", "model_edge"] == approx(0.32 - p.loc["moneyline", "market_fair_prob"])


def test_model_on_the_other_side():
    run_week(pred_margin=-12.0, home_win_prob=0.15)
    p = tracker.load_picks().set_index("bet_type")
    assert p.loc["spread", "side"] == "BUF" and p.loc["spread", "line"] == -8.5 and p.loc["spread", "price"] == -110
    assert p.loc["spread", "model_edge"] == approx(3.5) and p.loc["spread", "model_number"] == approx(-12.0)
    assert p.loc["moneyline", "side"] == "BUF" and p.loc["moneyline", "price"] == -420


def test_picks_are_frozen_once_logged():
    g = run_week()
    later = tracker.take_snapshot(g, KICK - dt.timedelta(days=1), events=[event()])
    assert tracker.log_picks(games(pred_margin=-20.0, home_win_prob=0.05), later, KICK - dt.timedelta(days=1)) == 0
    assert set(tracker.load_picks().side) == {"NYJ"}
    assert tracker.log_picks(g, later, KICK + dt.timedelta(hours=1), replace=True) == 0     # never after kickoff


def test_close_from_a_late_snapshot_then_grade_and_report():
    run_week()
    after = KICK + dt.timedelta(hours=4)
    assert tracker.capture_close(after, schedule()) == 2
    p = tracker.load_picks().set_index("bet_type")
    # Spread: took +8.5, closed +7.5 at -110/-110 -> +1 point, crossing no big key number.
    assert p.loc["spread", "close_source"] == "oddsapi" and p.loc["spread", "close_line"] == 7.5
    assert p.loc["spread", "clv_points"] == 1.0 and p.loc["spread", "beat_close"] == 1
    assert 0.01 < p.loc["spread", "clv_prob"] < 0.04 and p.loc["spread", "clv_ev"] > -1 / 22
    # Moneyline: took +340, closed +280/-350 -> fair = 0.26316 / (0.26316 + 0.77778) = 0.25281
    fair = no_vig_probs(280, -350)[0]
    assert p.loc["moneyline", "close_fair_prob"] == approx(fair)
    assert p.loc["moneyline", "clv_ev"] == approx(american_to_decimal(340) * fair - 1)       # +11.2%
    assert p.loc["moneyline", "beat_close"] == 1 and pd.isna(p.loc["moneyline", "clv_points"])

    # NYJ lose 20-27: +8.5 covers, the moneyline loses.
    assert tracker.grade(schedule(home_score=20, away_score=27), after) == 2
    p = tracker.load_picks().set_index("bet_type")
    assert p.loc["spread", "result"] == "W" and p.loc["spread", "profit_units"] == approx(100 / 105)
    assert p.loc["moneyline", "result"] == "L" and p.loc["moneyline", "profit_units"] == -1

    t = build_report(tracker.load_picks())["tracks"]["model"]
    assert t["summary"]["n"] == 2 and t["summary"]["wins"] == 1 and t["summary"]["losses"] == 1
    assert t["summary"]["units"] == approx(100 / 105 - 1, 0.01)
    assert t["summary"]["beat_close_pct"] == 1.0
    assert t["summary"]["expected_units"] == approx(p.clv_ev.sum(), 0.01)
    assert [r["key"] for r in t["by_type"]] == ["moneyline", "spread"]
    assert t["by_week"][0]["season"] == 2026 and t["by_week"][0]["week"] == 10
    assert {r["key"] for r in t["by_edge"]} == {"spread: 2-3.5 pts", "moneyline: 5-10%"}   # 32% model vs ~22.5% market
    assert t["sample"]["games"] == 1                              # two bets, one game: one observation
    assert len(t["timeline"]) == 2 and t["timeline"][-1]["cum_units"] == approx(100 / 105 - 1, 0.01)
    assert "model picks" in format_report(build_report(tracker.load_picks()))


def test_close_waits_for_the_final_then_falls_back_to_nflverse():
    run_week(close_minutes_before=600)                             # last snapshot was 10 hours before kickoff: too early
    after = KICK + dt.timedelta(hours=1)
    assert tracker.capture_close(after, schedule()) == 0           # game not final: keep waiting
    assert tracker.capture_close(after, schedule(home_score=20, away_score=27)) == 2
    p = tracker.load_picks().set_index("bet_type")
    assert p.loc["spread", "close_source"] == "nflverse" and p.loc["spread", "close_line"] == 9.5
    assert p.loc["spread", "clv_points"] == -1.0 and p.loc["spread", "beat_close"] == 0      # took +8.5, closed +9.5
    assert p.loc["moneyline", "close_fair_prob"] == approx(no_vig_probs(380, -500)[0])
    assert p.loc["moneyline", "clv_ev"] < 0                        # took +340, closed +380


def test_no_api_key_uses_nflverse_lines_for_the_pick():
    g = games()
    tue = KICK - dt.timedelta(days=5)
    tracker.log_picks(g, tracker.take_snapshot(g, tue, schedule=schedule()), tue)
    p = tracker.load_picks().set_index("bet_type")
    assert p.loc["spread", "line"] == 9.5 and p.loc["spread", "price"] == -108 and p.loc["spread", "book"] == "nflverse"
    assert p.loc["moneyline", "price"] == 380


def test_manual_pick_manual_close_and_backfill(tmp_path):
    sched = schedule(home_score=20, away_score=27)
    pick = tracker.add_pick(sched, "2026_10_BUF_NYJ", "spread", "buf", -110, line=-8.5, book="DraftKings")
    assert pick["pick_id"] == "2026_10_BUF_NYJ:spread:placed:1" and pick["side"] == "BUF"
    tracker.save_picks(pd.DataFrame([pick], columns=tracker.PICK_COLUMNS))
    closed = tracker.set_close(pick["pick_id"], line=-9.5, price=-110, opp_price=-110)
    assert closed["clv_points"] == 1.0 and closed["close_source"] == "manual"
    tracker.capture_close(KICK + dt.timedelta(hours=5), sched)
    assert tracker.load_picks().iloc[0].close_source == "manual"   # a manual close is never overwritten

    for bad in (dict(bet_type="spread", side="KC", price=-110, line=-3), dict(bet_type="spread", side="BUF", price=-110),
                dict(bet_type="moneyline", side="BUF", price=50), dict(bet_type="total", side="BUF", price=-110, line=44)):
        with pytest.raises(ValueError):
            tracker.add_pick(sched, "2026_10_BUF_NYJ", **bad)
    with pytest.raises(ValueError):
        tracker.add_pick(sched, "2026_10_XXX_YYY", "moneyline", "BUF", -110)

    csv = tmp_path / "past.csv"
    csv.write_text("game_id,bet_type,side,line,price,close_line,close_price,close_opp_price\n"
                   "2026_10_BUF_NYJ,moneyline,NYJ,,400,,380,-500\n"
                   "2026_10_BUF_NYJ,total,under,45.5,-110,,,\n", encoding="utf-8")
    assert tracker.backfill(csv, sched) == 2
    assert tracker.backfill(csv, sched) == 0                       # loading it twice adds nothing
    tracker.capture_close(KICK + dt.timedelta(hours=5), sched)     # the total gets the nflverse close (44.5)
    tracker.grade(sched, KICK + dt.timedelta(hours=5))
    p = tracker.load_picks().set_index("bet_type")
    assert p.loc["moneyline", "clv_ev"] == approx(5.0 * no_vig_probs(380, -500)[0] - 1)
    assert p.loc["total", "close_line"] == 44.5 and p.loc["total", "clv_points"] == 1.0
    assert p.loc["total", "result"] == "L"                         # 47 points, under 45.5
    assert p.loc["spread", "result"] == "L"                        # BUF won by 7, laying 8.5
    assert set(build_report(tracker.load_picks())["tracks"]) == {"model", "placed"}
