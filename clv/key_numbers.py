"""Key-number weights: how much more (or less) often NFL games land on each exact number than a
smooth bell curve centred on the closing line would predict.

    weight(k) = (games that landed on k) / (games the plain normal model expected to land on k)

where the plain normal model gives each game a bell curve centred on its own closing line. A weight
of 2.56 at 3 means a final margin of exactly 3 happens about 2.6x as often as the bell curve says,
which is why a half point through 3 is worth far more than one through 5.

Built from every game 2015-2025 (3,165 games; 2015 is when the longer extra point changed how
often games land on 1, 2, 5 and 6). To rebuild after more seasons:

    python -m clv.key_numbers

and paste the printed tables over the ones below. Margin weights are symmetric (keyed by the
absolute margin). Total weights are much noisier (about 60 games per number), so they are shrunk
halfway toward 1 and should be read as rough.

Sanity check: with these weights a 3-point favourite wins by exactly 3 about 8% of the time; the
2015-2025 figure for games closing between 2.5 and 3.5 is about 9%. So the value of the 3 is, if
anything, slightly understated.
"""
from __future__ import annotations

import numpy as np

MARGIN_WEIGHTS = {
    0: 0.20, 1: 0.80, 2: 0.83, 3: 2.56, 4: 0.88, 5: 0.86, 6: 1.33, 7: 1.68, 8: 0.91, 9: 0.42,
    10: 1.06, 11: 0.55, 12: 0.54, 13: 0.59, 14: 1.42, 15: 0.57, 16: 0.82, 17: 1.27, 18: 0.91,
    19: 0.58, 20: 0.98, 21: 1.15, 22: 0.73, 23: 0.94, 24: 1.35, 25: 0.95, 26: 0.96, 27: 0.87, 28: 1.50,
}   # margins above 28: weight 1 (too few games to estimate)

TOTAL_WEIGHTS = {   # already shrunk halfway toward 1
    30: 1.17, 31: 0.88, 32: 0.92, 33: 1.21, 34: 1.10, 35: 0.90, 36: 1.05, 37: 1.19, 38: 0.88, 39: 0.96,
    40: 1.15, 41: 1.12, 42: 0.84, 43: 1.11, 44: 1.13, 45: 0.95, 46: 0.96, 47: 1.07, 48: 0.97, 49: 0.96,
    50: 0.96, 51: 1.22, 52: 0.89, 53: 0.80, 54: 0.99, 55: 1.01, 56: 0.80, 57: 1.09, 58: 1.03,
}   # totals outside 30-58: weight 1

# Arrays the probability code indexes into.
MARGIN_K = np.arange(-80, 81)
MARGIN_W = np.array([MARGIN_WEIGHTS.get(abs(int(k)), 1.0) for k in MARGIN_K])
TOTAL_K = np.arange(0, 141)
TOTAL_W = np.array([TOTAL_WEIGHTS.get(int(k), 1.0) for k in TOTAL_K])


def build_tables(first_season: int = 2015, prior: float = 8) -> tuple[dict, dict]:
    """Recompute both tables from nflverse results. `prior` pulls thin counts toward a weight of 1."""
    import nflreadpy as nfl
    from scipy.stats import norm
    from .config import MARGIN_SIGMA, TOTAL_SIGMA
    s = nfl.load_schedules().to_pandas()
    s = s[(s.season >= first_season) & s.result.notna() & s.spread_line.notna() & s.total_line.notna()]

    def weights(values, lines, sigma, lo, hi, symmetric):
        k = np.arange(lo, hi + 1)
        counts = np.array([(values == x).sum() for x in k], dtype=float)
        # Games the plain normal model expects on each number: sum over games of P(k - .5 < result < k + .5).
        expected = np.array([(norm.cdf((x + .5 - lines) / sigma) - norm.cdf((x - .5 - lines) / sigma)).sum()
                             for x in k])
        if symmetric:                                   # home/away doesn't matter for key numbers
            counts, expected = (counts + counts[::-1]) / 2, (expected + expected[::-1]) / 2
        return k, (counts + prior) / (expected + prior)

    # nflverse: result = home margin, spread_line = points the home team is favoured by.
    k, w = weights(s.result.astype(int).values, s.spread_line.values, MARGIN_SIGMA, -45, 45, True)
    margin = {int(a): round(float(b), 2) for a, b in zip(k, w) if 0 <= a <= 28}
    k, w = weights(s.total.astype(int).values, s.total_line.values, TOTAL_SIGMA, 10, 95, False)
    total = {int(a): round(float(1 + (b - 1) / 2), 2) for a, b in zip(k, w) if 30 <= a <= 58}
    print(f"{len(s)} games since {first_season}")
    return margin, total


if __name__ == "__main__":
    m, t = build_tables()
    print("MARGIN_WEIGHTS =", m)
    print("TOTAL_WEIGHTS =", t)
