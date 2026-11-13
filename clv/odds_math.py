"""All the betting math: odds conversion, vig removal, and CLV. Pure functions, no file or network
access, so every formula here is covered by tests/test_odds_math.py.

Conventions used everywhere in this package
-------------------------------------------
* Prices are American odds (-110, +150).
* A spread bet's `line` is the handicap of the side you bet: -3.5 means you lay 3.5, +3.5 means
  you get 3.5. A total bet's `line` is the total itself.
* Positive CLV is always good for you.
"""
from __future__ import annotations

import math

import numpy as np

from .config import MARGIN_SIGMA, TOTAL_SIGMA
from .key_numbers import MARGIN_K, MARGIN_W, TOTAL_K, TOTAL_W


# ----------------------------------------------------------------------------------------------
# Odds conversion
# ----------------------------------------------------------------------------------------------

def american_to_decimal(price: float) -> float:
    """Decimal odds = total returned per 1 staked.
        +150 -> 1 + 150/100 = 2.50        -120 -> 1 + 100/120 = 1.8333
    """
    price = float(price)
    if price >= 100:
        return 1 + price / 100
    if price <= -100:
        return 1 + 100 / -price
    raise ValueError(f"not a valid American price: {price}")


def decimal_to_american(decimal: float) -> int:
    """2.50 -> +150, 1.8333 -> -120. Even money (2.0) is +100."""
    if decimal <= 1:
        raise ValueError(f"decimal odds must be above 1: {decimal}")
    return round((decimal - 1) * 100) if decimal >= 2 else round(-100 / (decimal - 1))


def implied_prob(price: float) -> float:
    """Break-even win probability of a price, vig included:  p = 1 / decimal.
        -110 -> 0.5238        +150 -> 0.4000
    """
    return 1 / american_to_decimal(price)


def prob_to_american(p: float) -> int:
    """The fair (no-vig) price of a probability: 0.60 -> -150."""
    return decimal_to_american(1 / p)


# ----------------------------------------------------------------------------------------------
# Vig removal
# ----------------------------------------------------------------------------------------------

def no_vig_probs(price_a: float, price_b: float) -> tuple[float, float]:
    """Fair probabilities for the two sides of a market (multiplicative / proportional method).

    The two implied probabilities add up to more than 1; the excess is the book's margin
    (the "overround"). Dividing each by their sum removes it:

        fair_a = implied_a / (implied_a + implied_b)

    Example: -110 / -110 -> 0.5238 + 0.5238 = 1.0476 -> 0.50 / 0.50.
             -150 / +130 -> 0.6000 + 0.4348 = 1.0348 -> 0.5798 / 0.4202.

    (Other methods exist - "power", "Shin" - that shift a little more of the vig onto longshots.
    For NFL sides, where prices are rarely beyond +/-500, the difference is a few tenths of a percent.)
    """
    a, b = implied_prob(price_a), implied_prob(price_b)
    return a / (a + b), b / (a + b)


def overround(price_a: float, price_b: float) -> float:
    """The book's margin: implied_a + implied_b - 1 (0.0476 for -110/-110)."""
    return implied_prob(price_a) + implied_prob(price_b) - 1


# ----------------------------------------------------------------------------------------------
# Moneyline CLV
# ----------------------------------------------------------------------------------------------

def moneyline_clv(my_price: float, close_fair_prob: float) -> float:
    """CLV of a moneyline bet, as expected return per unit staked if the no-vig close is the truth.

        CLV% = my_decimal_odds * fair_closing_prob - 1
             = fair_closing_prob / my_implied_prob - 1          (the same thing)

    Example: bet +150 (decimal 2.50, implied 40.0%); the close is -130/+110 on the other side,
    which de-vigs to 45.7% for my side:  2.50 * 0.457 - 1 = +14.3%.
    Zero means I got exactly the fair closing price; betting INTO the closing vig scores about -2%.
    """
    return american_to_decimal(my_price) * close_fair_prob - 1


# ----------------------------------------------------------------------------------------------
# Spread and total CLV (with key numbers)
# ----------------------------------------------------------------------------------------------
#
# A point of CLV is not a fixed amount of win probability: moving from -3.5 to -2.5 crosses the 3
# (about 8% of such games land there), moving from -5.5 to -4.5 crosses the 5 (under 3%).
# So the value of my line is computed from a full distribution of the final result:
#
#   1. Shape. P(result = k) is proportional to  Normal(k; mu, sigma) * key_number_weight(k)
#      over whole-number results k. sigma is how far results land from the closing line
#      historically (12.7 for margins, 13.2 for totals). The weights are in key_numbers.py.
#   2. Centre. mu is chosen so that this distribution agrees with the closing market: the
#      probability of covering the CLOSING line (pushes excluded) equals the no-vig closing
#      probability (0.50 when both sides close at the same price).
#   3. Value. Under that distribution, work out win / push / lose for MY line, then
#
#        CLV points = my line - closing line                (handicaps of the side I bet)
#        CLV prob   = cover%(my line) - cover%(closing line), where cover% = P(win) + P(push)/2
#        CLV EV     = P(win) * (my_decimal - 1) - P(lose)   (expected return per unit staked;
#                                                             a push returns the stake)
#
# CLV EV is the number comparable to moneyline CLV%: both are "expected profit per unit if the
# no-vig closing line is right".

def _outcome_probs(k, pmf, sign: int, line: float) -> tuple[float, float, float]:
    """(win, push, lose) for a bet that wins when  sign * result + line > 0."""
    x = sign * k + line
    push = np.isclose(x, 0)
    return float(pmf[(x > 0) & ~push].sum()), float(pmf[push].sum()), float(pmf[(x < 0) & ~push].sum())


def _pmf(k, weights, mu: float, sigma: float):
    p = np.exp(-0.5 * ((k - mu) / sigma) ** 2) * weights
    return p / p.sum()


def _fit_mu(k, weights, sigma: float, sign: int, close_line: float, close_fair: float) -> float:
    """Find the centre mu at which P(win | no push) on the closing line equals close_fair (bisection)."""
    if not 0 < close_fair < 1:
        raise ValueError(f"closing fair probability must be between 0 and 1: {close_fair}")
    centre = -sign * close_line                 # where the bet is roughly 50/50
    lo, hi = centre - 40, centre + 40
    for _ in range(60):
        mid = (lo + hi) / 2
        win, _, lose = _outcome_probs(k, _pmf(k, weights, mid, sigma), sign, close_line)
        too_low = win / (win + lose) < close_fair
        # The win probability rises with mu for sign=+1 and falls with mu for sign=-1.
        if too_low == (sign > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _line_clv(k, weights, sigma, sign, my_line, my_price, close_line, close_fair) -> dict:
    mu = _fit_mu(k, weights, sigma, sign, close_line, close_fair)
    pmf = _pmf(k, weights, mu, sigma)
    win, push, lose = _outcome_probs(k, pmf, sign, my_line)
    cwin, cpush, _ = _outcome_probs(k, pmf, sign, close_line)
    return {
        "points": my_line - close_line,
        "win": win, "push": push, "lose": lose,
        "cover_prob": win + push / 2,
        "close_cover_prob": cwin + cpush / 2,
        "prob": (win + push / 2) - (cwin + cpush / 2),
        "ev": win * (american_to_decimal(my_price) - 1) - lose,
    }


def spread_clv(my_line: float, my_price: float, close_line: float, close_fair_prob: float = 0.5) -> dict:
    """CLV of a spread bet. Lines are the handicap of the side I bet, at pick time and at the close.

    close_fair_prob = no-vig probability that my side covers the CLOSING line (0.5 if the close was
    -110 both ways). Returns points, prob, ev and the win/push/lose probabilities at my line.

    Example: I took +3.5 at -110 and it closed +2.5 at -110/-110. points = +1.0, and because that
    point crosses the 3, prob is about +8.0% and ev about +10.8% (vs -4.5% for betting the close).
    The same one point from +5.5 to +4.5 is worth only +2.7% prob and +0.6% ev.
    """
    return _line_clv(MARGIN_K, MARGIN_W, MARGIN_SIGMA, +1, my_line, my_price, close_line, close_fair_prob)


def total_clv(side: str, my_total: float, my_price: float, close_total: float, close_fair_prob: float = 0.5) -> dict:
    """CLV of a total bet; side is "over" or "under". points is positive when my number was better
    (over: closing total above mine; under: closing total below mine)."""
    if side not in ("over", "under"):
        raise ValueError(f"side must be 'over' or 'under': {side}")
    # Over wins when  total - line > 0  (sign +1, line -> -total).
    # Under wins when line - total > 0  (sign -1, line -> +total).
    sign = +1 if side == "over" else -1
    return _line_clv(TOTAL_K, TOTAL_W, TOTAL_SIGMA, sign, -sign * my_total, my_price, -sign * close_total,
                     close_fair_prob)


# ----------------------------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------------------------

def profit_units(result: str, price: float, stake: float = 1.0) -> float:
    """Profit in units: a win pays stake * (decimal - 1), a loss costs the stake, a push returns it."""
    if result == "W":
        return stake * (american_to_decimal(price) - 1)
    if result == "L":
        return -stake
    if result == "P":
        return 0.0
    raise ValueError(f"result must be W, L or P: {result}")


def grade_bet(bet_type: str, side_is_home: bool | None, side: str, line: float | None,
              home_score: float, away_score: float) -> str:
    """W / L / P for a finished game.

        spread:    my side's margin + my line  > 0 win, = 0 push, < 0 loss
        moneyline: my side's margin            > 0 win, = 0 push (tie game: stake returned), < 0 loss
        total:     over wins when the combined score is above the line; under is the mirror image
    """
    if bet_type == "total":
        x = (home_score + away_score) - line
        x = x if side == "over" else -x
    else:
        margin = (home_score - away_score) * (1 if side_is_home else -1)
        x = margin + (line if bet_type == "spread" else 0)
    return "P" if math.isclose(x, 0, abs_tol=1e-9) else ("W" if x > 0 else "L")


# ----------------------------------------------------------------------------------------------
# Sample size
# ----------------------------------------------------------------------------------------------

def mean_ci(values, z: float = 1.96) -> dict:
    """Mean with a normal-approximation 95% confidence interval:

        se = sd / sqrt(n)        CI = mean +/- 1.96 * se

    n_needed is how many observations it takes before a true mean of this size would sit 1.96
    standard errors from zero:  n = (1.96 * sd / mean)^2.  None when the mean is not positive.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "lo": None, "hi": None, "n_needed": None}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    lo, hi = (mean - z * se, mean + z * se) if se is not None else (None, None)
    needed = math.ceil((z * sd / mean) ** 2) if sd and mean > 0 else None
    return {"n": n, "mean": mean, "sd": sd, "se": se, "lo": lo, "hi": hi, "n_needed": needed}
