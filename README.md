# NFL game predictions

Win probability and predicted point spread for every upcoming NFL game, published to a
mobile-friendly dashboard on GitHub Pages and refreshed automatically by GitHub Actions.
No betting lines or odds are used anywhere.

- **This week:** `index.html` — matchups, kickoff (Central Time), win %, spread, QB starter flags, uncertainty notes
- **Track record:** `history.html` — pick record, spread error, running Brier score by week
- **Backtest:** [MODEL_REPORT.md](MODEL_REPORT.md)

## How it works

```
nflverse data ─▶ cached summaries ─▶ pregame features ─▶ ridge model ─▶ ledger ─▶ JSON ─▶ static site
 (nflreadpy)      data/*.parquet       features.py        models.py    predictions.csv  site/data
```

1. **Data** (`nflpred/data.py`) — play-by-play since 1999 is boiled down to one row per team-game
   (EPA/play and success rate, split pass/run, offense and defense) and one row per QB-game
   (dropbacks, EPA). These small parquet files are committed, so runs only re-download the current season.
2. **Ratings & features** (`nflpred/features.py`, `qb.py`, `elo.py`)
   - Team ratings: recency-weighted average of this season's games (half-life 6 games), blended with
     last season's rating regressed 50% to the league mean; the prior fades as games are played.
   - QB rating: shrunken EPA per dropback, decaying over ~600 dropbacks. Two features use it:
     the starter's rating, and the *change* vs. the QBs behind the team's recent stats (this is what
     catches a backup starting).
   - Elo, home field, rest difference, bye weeks, travel distance, and a divisional-game interaction.
   - Every feature uses only information from before kickoff.
3. **Starting QB for upcoming games** (`qb.project_starters`) — latest nflverse depth chart, minus anyone
   Out/Doubtful on the injury report or on IR/inactive rosters. Flags: backup starting, QB1 questionable,
   change from last game.
4. **Model** (`nflpred/models.py`) — ridge regression predicts the home margin. Win probability is
   `Φ(margin / σ)` with σ ≈ 13 points from training residuals, so spread and win % always agree.
   Training weights recent seasons more (half-life 6 seasons).
5. **Ledger** (`data/predictions.csv`) — every run writes predictions for games that haven't kicked off;
   once a game starts its prediction is frozen. Scores are filled in after games finish.
6. **Site** (`site/`) — plain HTML/CSS/JS reading `site/data/current.json` and `history.json`.

## Run locally (PowerShell)

```powershell
cd C:\Users\Murph\Coding\nfl_predictor
python -m venv .venv                     # first time only
.\.venv\Scripts\pip install -r requirements.txt

.\.venv\Scripts\python run.py full       # refresh data + predict this week + rebuild site data
.\.venv\Scripts\python run.py score      # fill in final scores
.\.venv\Scripts\python run.py backtest   # walk-forward backtest -> MODEL_REPORT.md
.\.venv\Scripts\python run.py full --no-refresh   # use cached data only (offline)

# look at the dashboard
cd site; ..\.venv\Scripts\python -m http.server 8000   # then open http://localhost:8000
```

## Automation

`.github/workflows/pipeline.yml` runs on this schedule (all Central Time, Sept–Feb):

| When | Command | Why |
|---|---|---|
| Tue 8:00 AM | `full` | week is set; refresh all data, retrain, predict, publish |
| Thu 3:00 PM | `refresh` | latest injury report/QB status before Thursday Night Football |
| Sun 10:40 AM | `refresh` | after inactives for the noon games |
| Mon 8:00 AM | `score` | score Sunday's games into the track record |

You can also run it by hand: GitHub → **Actions** → **NFL pipeline** → **Run workflow**, or
`gh workflow run pipeline.yml -f command=full`.

**If a run fails:** the deploy step never runs, so the old dashboard stays up. The run shows a red ❌
in the Actions tab, GitHub emails you, the workflow opens an issue on the repo, and the dashboard
shows a warning banner if it hasn't updated in 8 days.

### Changing the schedule
Cron times are **UTC** and GitHub doesn't adjust for daylight saving, so each job has two lines:
months `9,10` use Central Daylight (UTC−5) and `11,12,1,2` use Central Standard (UTC−6).
To move a run, change the hour on both lines: CT hour + 5 for the Sep–Oct line, + 6 for the Nov–Feb line.
Example: Thursday 2 PM CT → `"0 19 * 9,10 4"` and `"0 20 * 1,2,11,12 4"`. The last field is the weekday
(0 = Sunday). Daylight time ends on the first Sunday of November, so in years when that's later than
Nov 1, runs in those few days fire an hour early. Also, GitHub's scheduled runs are often 5–30
minutes late at busy times.

## Retraining

There's no saved model file. Training takes about a second, so **every run retrains** on all completed
games. To change the model:

1. Edit settings in `nflpred/config.py` (half-lives, prior weight, ridge alpha, Elo K…) or the
   feature list in `nflpred/features.py` (`FEATURES`; try ideas via `EXPERIMENTAL`).
2. Run `python run.py backtest` and compare `MODEL_REPORT.md` with the previous version (`git diff`).
3. Keep the change only if out-of-sample Brier/log loss improve, then commit and push.

To add a candidate model to the backtest, add it to `candidate_models()` in `nflpred/report.py`.

## Caveats
- The backtest uses the QB who actually started (known at kickoff). Tuesday predictions use a
  projected starter, so they're noisier in weeks with QB uncertainty; the Sunday re-run fixes most of that.
- Early-season weeks lean heavily on last season's ratings and are less accurate.
- Weeks labeled **backfill** on the track record page were predicted after the fact (model trained only on
  prior seasons, pregame features only), to give the 2026 track record a starting point.
