"""Tests for the betting math. Expected values are worked out by hand in the comments, so a failure
here means a formula changed, not that a number drifted.

    .venv\\Scripts\\python -m pytest
"""
import math

import pytest

from clv.odds_math import (american_to_decimal, decimal_to_american, grade_bet, implied_prob, mean_ci,
                           moneyline_clv, no_vig_probs, overround, prob_to_american, profit_units,
                           spread_clv, total_clv)

approx = lambda v, tol=1e-4: pytest.approx(v, abs=tol)


# ---------------------------------------------------------------- odds conversion

@pytest.mark.parametrize("american, decimal", [
    (100, 2.0), (-100, 2.0),          # even money, either spelling
    (150, 2.5),                       # 1 + 150/100
    (-150, 1 + 100 / 150),            # 1.6667
    (-110, 1 + 100 / 110),            # 1.9091
    (350, 4.5), (-1650, 1 + 100 / 1650),
])
def test_american_to_decimal(american, decimal):
    assert american_to_decimal(american) == approx(decimal)


@pytest.mark.parametrize("bad", [0, 50, -99, 99.9])
def test_american_prices_between_minus_100_and_100_are_rejected(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


@pytest.mark.parametrize("american", [-1650, -455, -150, -110, -105, 100, 102, 130, 350, 950])
def test_decimal_round_trip(american):
    assert decimal_to_american(american_to_decimal(american)) == american


def test_implied_prob():
    assert implied_prob(-110) == approx(110 / 210)      # 0.5238
    assert implied_prob(150) == approx(100 / 250)       # 0.4000
    assert implied_prob(-300) == approx(0.75)
    assert implied_prob(100) == approx(0.5)


def test_prob_to_american():
    assert prob_to_american(0.60) == -150
    assert prob_to_american(0.40) == 150
    assert prob_to_american(0.50) == 100


# ---------------------------------------------------------------- vig removal

def test_no_vig_even_market():
    # -110/-110: 0.5238 + 0.5238 = 1.0476 -> 0.5 each
    assert no_vig_probs(-110, -110) == (approx(0.5), approx(0.5))
    assert overround(-110, -110) == approx(0.047619)


def test_no_vig_favourite_and_dog():
    # -150 -> 0.6000, +130 -> 0.43478; sum 1.03478 -> 0.57983 / 0.42017
    fav, dog = no_vig_probs(-150, 130)
    assert fav == approx(0.6 / 1.0347826)
    assert dog == approx(0.4347826 / 1.0347826)
    assert fav + dog == approx(1.0)


def test_no_vig_is_symmetric_and_below_raw_implied():
    a, b = no_vig_probs(-455, 350)
    b2, a2 = no_vig_probs(350, -455)
    assert (a, b) == (approx(a2), approx(b2))
    assert a < implied_prob(-455) and b < implied_prob(350)      # removing vig lowers both


def test_no_vig_with_no_vig_changes_nothing():
    assert no_vig_probs(-150, 150) == (approx(0.6), approx(0.4))


# ---------------------------------------------------------------- moneyline CLV

def test_moneyline_clv_worked_example():
    # Bet +150 (decimal 2.5). Close: my side +110, other side -130.
    # implied 0.47619 and 0.56522 -> fair 0.47619 / 1.04141 = 0.45726.  CLV = 2.5 * 0.45726 - 1 = +14.31%
    fair = no_vig_probs(110, -130)[0]
    assert fair == approx(0.45726)
    assert moneyline_clv(150, fair) == approx(0.14314)


def test_moneyline_clv_zero_at_the_fair_close_and_negative_at_the_vigged_close():
    assert moneyline_clv(150, 0.40) == approx(0.0)                # +150 is exactly fair for 40%
    fair = no_vig_probs(-110, -110)[0]
    assert moneyline_clv(-110, fair) == approx(-1 / 22)           # 1.9091 * 0.5 - 1 = -4.55%: you paid the vig


def test_moneyline_clv_equals_ratio_of_probabilities():
    # CLV% = fair_close / my_implied - 1 is the same formula written differently.
    assert moneyline_clv(-200, 0.70) == approx(0.70 / implied_prob(-200) - 1)


def test_moneyline_clv_sign_follows_line_movement():
    assert moneyline_clv(130, no_vig_probs(110, -130)[0]) > 0     # dog shortened after I bet: good
    assert moneyline_clv(110, no_vig_probs(130, -150)[0]) < 0     # dog drifted after I bet: bad


# ---------------------------------------------------------------- spread CLV and key numbers

def test_spread_same_line_same_price_is_the_vig():
    r = spread_clv(-3.5, -110, -3.5)
    assert r["points"] == 0 and r["prob"] == approx(0.0)
    assert r["ev"] == approx(-1 / 22)                             # 0.5 * 0.9091 - 0.5
    assert r["win"] + r["push"] + r["lose"] == approx(1.0)


def test_spread_points_sign():
    assert spread_clv(3.5, -110, 2.5)["points"] == 1.0            # dog: got more points than the close
    assert spread_clv(-2.5, -110, -3.5)["points"] == 1.0          # favourite: laid fewer than the close
    assert spread_clv(-3.5, -110, -2.5)["points"] == -1.0         # laid more than the close: bad
    assert spread_clv(-3.5, -110, -2.5)["ev"] < spread_clv(-2.5, -110, -2.5)["ev"]


def test_favourite_and_dog_views_agree():
    # Crossing the 3 is worth the same whether you hold the dog (+3.5 vs +2.5) or the favourite (-2.5 vs -3.5).
    assert spread_clv(3.5, -110, 2.5)["prob"] == approx(spread_clv(-2.5, -110, -3.5)["prob"], 1e-3)


def test_key_numbers_three_beats_seven_beats_five():
    through_3 = spread_clv(3.5, -110, 2.5)["prob"]
    through_7 = spread_clv(7.5, -110, 6.5)["prob"]
    through_5 = spread_clv(5.5, -110, 4.5)["prob"]
    through_9 = spread_clv(9.5, -110, 8.5)["prob"]
    assert through_3 > through_7 > through_5 > through_9 > 0
    assert 0.07 < through_3 < 0.10      # P(favourite wins by exactly 3); about 9% in 2015-2025 games
    assert 0.04 < through_7 < 0.07
    assert through_5 < 0.035
    assert through_3 > 2.5 * through_5  # the whole point: not every point is equal


def test_half_point_onto_a_key_number_is_half_the_push():
    # +3 instead of +2.5: the games that land on 3 become pushes instead of losses.
    r = spread_clv(3, -110, 2.5)
    assert r["push"] == approx(spread_clv(3.5, -110, 2.5)["prob"], 1e-3)
    assert r["prob"] == approx(r["push"] / 2, 1e-6)


def test_spread_respects_closing_juice():
    # Close -3 with my side at -120 / other side +100: my side's fair cover prob is above 50%,
    # so holding -3 at -110 is better than it looks from the line alone.
    fair = no_vig_probs(-120, 100)[0]
    assert fair == approx(0.52174)
    assert spread_clv(-3, -110, -3, fair)["ev"] > spread_clv(-3, -110, -3, 0.5)["ev"]
    # And at the fair price for that probability the EV is zero by construction.
    assert spread_clv(-3.5, prob_to_american(0.5), -3.5, 0.5)["ev"] == approx(0.0, 1e-3)


def test_spread_ev_is_win_times_payout_minus_lose():
    r = spread_clv(3.5, -105, 2.5)
    assert r["ev"] == approx(r["win"] * (100 / 105) - r["lose"], 1e-9)


def test_more_points_is_never_worse():
    evs = [spread_clv(line, -110, 0.5)["ev"] for line in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 7.5)]
    assert evs == sorted(evs)


# ---------------------------------------------------------------- totals

def test_total_points_direction():
    assert total_clv("over", 44.5, -110, 45.5)["points"] == 1.0   # over a lower number than the close: good
    assert total_clv("over", 45.5, -110, 44.5)["points"] == -1.0
    assert total_clv("under", 45.5, -110, 44.5)["points"] == 1.0  # under a higher number than the close: good
    assert total_clv("over", 44.5, -110, 45.5)["ev"] > total_clv("over", 44.5, -110, 44.5)["ev"]
    assert total_clv("under", 45.5, -110, 44.5)["ev"] > total_clv("under", 44.5, -110, 44.5)["ev"]


def test_total_same_line_is_the_vig_and_a_point_is_worth_a_few_percent():
    assert total_clv("under", 44.5, -110, 44.5)["ev"] == approx(-1 / 22)
    assert 0.015 < total_clv("over", 44.5, -110, 45.5)["prob"] < 0.045


def test_total_rejects_bad_side():
    with pytest.raises(ValueError):
        total_clv("home", 44.5, -110, 44.5)


# ---------------------------------------------------------------- grading

def test_profit_units():
    assert profit_units("W", -110) == approx(100 / 110)
    assert profit_units("W", 150, stake=2) == approx(3.0)
    assert profit_units("L", 150, stake=2) == -2
    assert profit_units("P", -110) == 0


@pytest.mark.parametrize("bet_type, is_home, side, line, hs, as_, expected", [
    ("spread", True, "KC", -3.5, 27, 24, "L"),        # home won by 3, laying 3.5
    ("spread", True, "KC", -3, 27, 24, "P"),
    ("spread", True, "KC", -2.5, 27, 24, "W"),
    ("spread", False, "NYJ", 3.5, 27, 24, "W"),       # away lost by 3, getting 3.5
    ("spread", False, "NYJ", 3, 27, 24, "P"),
    ("spread", False, "NYJ", 2.5, 27, 24, "L"),
    ("moneyline", True, "KC", None, 27, 24, "W"),
    ("moneyline", False, "NYJ", None, 27, 24, "L"),
    ("moneyline", False, "NYJ", None, 20, 20, "P"),   # tie game: stake back
    ("total", None, "over", 50.5, 27, 24, "W"),       # 51 points
    ("total", None, "under", 50.5, 27, 24, "L"),
    ("total", None, "over", 51, 27, 24, "P"),
    ("total", None, "under", 51.5, 27, 24, "W"),
])
def test_grade_bet(bet_type, is_home, side, line, hs, as_, expected):
    assert grade_bet(bet_type, is_home, side, line, hs, as_) == expected


# ---------------------------------------------------------------- sample size

def test_mean_ci():
    # values 0.01..0.05: mean 0.03, sd 0.015811, se 0.007071, CI = 0.03 +/- 1.96 * 0.007071
    ci = mean_ci([0.01, 0.02, 0.03, 0.04, 0.05])
    assert ci["n"] == 5 and ci["mean"] == approx(0.03)
    assert ci["lo"] == approx(0.03 - 1.96 * 0.0158114 / math.sqrt(5))
    assert ci["hi"] == approx(0.03 + 1.96 * 0.0158114 / math.sqrt(5))
    assert ci["n_needed"] == math.ceil((1.96 * 0.0158114 / 0.03) ** 2)      # 2


def test_mean_ci_edge_cases():
    assert mean_ci([])["mean"] is None
    assert mean_ci([0.02])["lo"] is None                     # one observation: no interval
    assert mean_ci([-0.02, -0.01, 0.0])["n_needed"] is None  # non-positive mean: never "significant edge"
    assert mean_ci([0.01, float("nan"), 0.03])["n"] == 2     # blanks ignored
