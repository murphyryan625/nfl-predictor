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

`.github/workflows/pipeline.yml` runs on this schedule, Sept–Feb. Times are Central Standard Time
(Nov–Feb); in Sep–Oct, during daylight time, each attempt is one hour later on the clock:

| When (scheduled attempts, CST) | Command | Why |
|---|---|---|
| Tue 4:17 AM, 8:17 AM | `full` | week is set; refresh all data, retrain, predict, publish |
| Thu 6:17 AM, 10:17 AM, 2:17 PM | `refresh` | latest injury report/QB status before Thursday Night Football |
| Fri 9:17 PM, Sat 9:17 AM | `refresh` | Friday's final injury report (Out/Doubtful); Saturday games late in the season |
| Sun 3:17 AM, 6:17 AM, 9:17 AM, 10:47 AM | `refresh` | last update before the noon games |
| Mon 6:17 AM, 12:17 PM | `score` | score Sunday's games into the track record |

**Why several attempts per job:** GitHub starts scheduled workflows late — in early Nov 2026 every run
started 4–9 hours after its cron time, so the single "Sunday 10:40 AM" refresh actually ran at 2:44 PM,
after the noon games had kicked off. Repeating a run is harmless (predictions freeze at kickoff and a run
takes ~2 minutes), so each job is scheduled a few times, hours apart, and whichever attempts land before
kickoff do the work. If you need a run at an exact time, trigger it yourself (below) — manual runs start
immediately.

You can also run it by hand: GitHub → **Actions** → **NFL pipeline** → **Run workflow**, or
`gh workflow run pipeline.yml -f command=full`.

**If a run fails:** the deploy step never runs, so the old dashboard stays up. The run shows a red ❌
in the Actions tab, GitHub emails you, the workflow opens an issue on the repo, and the dashboard
shows a warning banner if it hasn't updated in 8 days.

### Changing the schedule
Cron times are **UTC** and GitHub doesn't adjust for daylight saving: Central is UTC−5 in Sep–Oct and
UTC−6 in Nov–Feb, so the CST times above are an hour later on the clock in Sep–Oct (CST = UTC hour − 6).
Given how late GitHub starts runs, that hour doesn't matter. Each line is `minute hours day-of-month months weekday`;
the **weekday** (last field, 0 = Sunday) decides the command: 2 (Tue) → `full`, 1 (Mon) → `score`,
anything else → `refresh`. To add an attempt, add an hour to the list, e.g. Thursday
`"17 12,16,20,22 * 1,2,9-12 4"`.

## Retraining

There's no saved model file. Training takes about a second, so **every run retrains** on all completed
games. To change the model:

1. Edit settings in `nflpred/config.py` (half-lives, prior weight, ridge alpha, Elo K…) or the
   feature list in `nflpred/features.py` (`FEATURES`; try ideas via `EXPERIMENTAL`).
2. Run `python run.py backtest` and compare `MODEL_REPORT.md` with the previous version (`git diff`).
3. Keep the change only if out-of-sample Brier/log loss improve, then commit and push.

To add a candidate model to the backtest, add it to `candidate_models()` in `nflpred/report.py`.

**Already tried (Nov 2026) — none moved out-of-sample Brier by more than ±0.0003, which is noise over
4,363 games, so the model was left as is:** opponent-adjusted team ratings; rating half-lives from 4 to 20
games and other prior weights/regression; a points-scored/allowed rating; Elo K and season regression;
QB shrinkage, half-life and replacement level; ridge alpha 1–300; rescaling σ; blending ridge with
gradient boosting or the logistic model; and new play-by-play stats (EPA capped per play to mute
turnover luck, garbage-time down-weighting, special-teams EPA, explosive-play rate, early-down success,
sack rate, turnovers). The remaining gap to the betting market is mostly information this model doesn't
have (non-QB injuries, coaching and roster changes), not tuning.

## Caveats
- The backtest uses the QB who actually started (known at kickoff). Tuesday predictions use a
  projected starter, so they're noisier in weeks with QB uncertainty; the Sunday re-run fixes most of that.
- Early-season weeks lean heavily on last season's ratings and are less accurate.
- Weeks labeled **backfill** on the track record page were predicted after the fact (model trained only on
  prior seasons, pregame features only), to give the 2026 track record a starting point.
