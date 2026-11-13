# Closing Line Value tracker

Measures whether the model's picks get a better number than the market's final one. Winning or
losing 55% over 100 bets says almost nothing; beating the closing line consistently says a lot,
and says it much sooner.

It lives in its own package (`clv/`) and only *reads* the model's ledger (`data/predictions.csv`).
Nothing in `nflpred/` imports it, so betting lines still never reach the model.

## Setup

Nothing is required: without a key the tracker uses the lines in the nflverse schedule file, which
the pipeline already downloads.

For multi-book consensus lines and best available prices, add a free key from
[The Odds API](https://the-odds-api.com) (500 credits a month; each run uses 2, and the scheduled
pipeline makes roughly 60 runs a month):

```powershell
# on the laptop (new terminals pick it up)
setx ODDS_API_KEY "your-key-here"
```

For the scheduled GitHub runs: repo **Settings → Secrets and variables → Actions → New repository
secret**, name `ODDS_API_KEY`. The key is only ever read from that environment variable; it is not
written to any file or log.

Tests:

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest
```

## Weekly workflow

It runs by itself inside the existing pipeline. `run.py full` / `refresh` do all four steps, `run.py score`
does the last three:

| Step | What happens | By hand |
|---|---|---|
| 1. Log picks | Records the current market, then logs the model's side of the spread and of the moneyline for every game that doesn't have picks yet, at the best price available, with book and timestamp. | `python -m clv log` |
| 2. Capture the close | For games that have kicked off: the last Odds API snapshot taken within 90 minutes of kickoff if there is one, otherwise the nflverse closing line once the game is final. | `python -m clv close` |
| 3. Grade | W / L / push and profit in units. | `python -m clv grade` |
| 4. Report | Console summary, `site/data/clv.json`, and the **CLV** page of the site. | `python -m clv report` |

`python -m clv weekly` runs all four; `python -m clv status` shows upcoming games (with their
`game_id`) and anything still waiting on a close or a result.

**A pick is logged once and then frozen**, the way a real bet is. Later model re-runs (Thursday and
Sunday refreshes) don't move it. `python -m clv log --replace` re-logs games that haven't kicked off.

### Bets you actually place

```powershell
python -m clv add --game 2026_10_BUF_NYJ --type spread --side BUF --line -8.5 --price -110 --book DraftKings --stake 2
python -m clv add --game 2026_10_BUF_NYJ --type moneyline --side NYJ --price 340 --book FanDuel
python -m clv add --game 2026_10_BUF_NYJ --type total --side over --line 44.5 --price -110
```

`--line` is the handicap of the side you bet (`-8.5` = laying 8.5, `+8.5` = getting 8.5), or the
total. These go in a separate "placed" track, reported apart from the automatic "model" track.
They get closing lines and grades the same way.

### Typing in a closing line (manual fallback)

```powershell
python -m clv set-close --pick 2026_10_BUF_NYJ:spread:placed:1 --line -9.5 --price -108 --opp-price -112
python -m clv set-close --pick 2026_10_BUF_NYJ:moneyline:placed:1 --price 280 --opp-price -350
```

`--price` is the closing price on your side, `--opp-price` on the other side; both are needed to
remove the vig. A close entered by hand is never overwritten.

### Backfilling past weeks

Fill in a CSV like [`backfill_template.csv`](backfill_template.csv) and run
`python -m clv backfill my_past_picks.csv`. Required columns: `game_id, bet_type, side, price`
(plus `line` for spreads and totals). Optional: `book, line_ts, stake, track, model_number,
model_edge, close_line, close_price, close_opp_price`. Rows without a closing line get the
nflverse close automatically. Loading the same file twice adds nothing.

Weeks 1–9 of 2026 can't be backfilled automatically: the lines that were available when those
predictions were made weren't recorded, and CLV needs both ends.

## Where the data lives, and why CSV

| File | Contents |
|---|---|
| `data/clv_picks.csv` | One row per bet: what was taken, the close, CLV, result. Column meanings are commented in `tracker.py`. |
| `data/clv_odds_snapshots.csv` | Append-only log of the market on every run. |
| `site/data/clv.json` | The report the CLV page reads. Rebuilt from the picks file every time. |

CSV rather than SQLite because GitHub Actions commits this repo's data: a CSV diffs line by line, so
every pick and every later change to it shows in git history and merges cleanly, while a SQLite file is
a binary blob git can't diff or merge. At a few hundred rows a season nothing here needs a database,
and you can open the picks in Excel.

## Reading the numbers

- **CLV EV** (the headline) — expected profit per unit staked *if the no-vig closing line is the
  true price*. Zero means you got exactly a fair price. Betting the closing line itself scores about
  −2% (moneyline) to −4.5% (spread at −110), because you pay the vig. Sustained positive CLV EV is
  the evidence of an edge.
- **CLV pts** — spreads and totals only: points better than the close. Not all points are equal,
  which is why CLV EV is the number to judge by.
- **Beat close %** — share of bets with a better number than the close (or the same number at a
  better price). About 50% is what guessing gets you.
- **Expected vs. actual units** — the sum of stake × CLV EV against real profit, over the same bets.
  Good CLV with bad results is bad luck; bad CLV with good results is good luck that won't last.
- **By size of the model's edge** — if the model is right about the games where it disagrees with
  the market most, CLV should rise with the bucket. If it falls, the big disagreements are the
  model's mistakes, not the market's.
- **Is it real yet?** — a 95% confidence interval on average CLV, computed across *games* (a spread
  and a moneyline pick on the same game aren't independent), and how many games it would take to be
  statistically meaningful at the current average.

### The formulas (all in [`odds_math.py`](odds_math.py), tested in `tests/test_odds_math.py`)

```
decimal odds       +150 -> 1 + 150/100 = 2.50          -120 -> 1 + 100/120 = 1.833
implied prob       1 / decimal
no-vig prob        implied_a / (implied_a + implied_b)             (multiplicative method)
moneyline CLV      my_decimal * fair_close_prob - 1
spread/total CLV   points = my line - closing line
                   EV     = P(win at my line) * (my_decimal - 1) - P(lose at my line)
```

The spread probabilities come from a distribution of the final margin: a bell curve (σ = 12.7
points) multiplied by key-number weights measured from 2015–2025 results, centred so that it agrees
with the no-vig closing line. With it, moving from +2.5 to +3.5 is worth about 8% of win
probability, from +6.5 to +7.5 about 5%, and from +4.5 to +5.5 under 3%. The weights are in
[`key_numbers.py`](key_numbers.py); `python -m clv.key_numbers` rebuilds them.

## Things to know

- **When the pick is logged matters.** Picks are logged on the first pipeline run that sees the game,
  usually Tuesday. Lines move most on injury news, which the Tuesday model doesn't have, so this is a
  demanding test.
- **The automatic track bets every game**, including ones where the model and market nearly agree.
  Use the edge-size table to see whether a threshold would help.
- **Which book a pick is priced at** is set by `PICK_BOOKS` in `config.py` (currently Fanatics).
  The Odds API only serves Fanatics to paid plans, so on the free tier picks are priced at the median
  of the regulated US books in `PROXY_BOOKS` and the book column says "(proxy)". Offshore books are
  never used for pick prices. The closing line always uses every book. To track real Fanatics
  prices on the free tier, log the bets you place with `python -m clv add ... --book Fanatics`.
- **nflverse lines come from a single source**, not a consensus; they are used when no key is set.
- **Closing line source is recorded per pick** (`close_source`: `oddsapi`, `nflverse`, `manual`,
  `backfill`). GitHub starts scheduled runs hours late, so a snapshot inside the 90-minute window is
  uncommon; most closes will come from nflverse. Running `python -m clv snapshot` on the laptop shortly
  before kickoff records a true consensus close (then commit `data/clv_odds_snapshots.csv`).
- **Totals** are supported for hand-entered bets only; the model doesn't predict totals. Their
  key-number weights are rough.
- Settings (close window, default stake, σ, edge buckets) are in [`config.py`](config.py).
