"""Central settings. Change numbers here rather than hunting through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # compact cached tables (committed to git)
SITE_DIR = ROOT / "site"            # static dashboard published to GitHub Pages
SITE_DATA = SITE_DIR / "data"       # JSON the dashboard reads

FIRST_SEASON = 1999                 # first season with nflverse play-by-play
FIRST_TEST_SEASON = 2010            # walk-forward backtest starts predicting here
MIN_TRAIN_SEASON = 2002             # training rows start here (need a prior season first)

# --- team ratings (EPA / success rate) ---
RECENCY_HALF_LIFE_GAMES = 6         # within-season weighting: a game 6 games ago counts half
PRIOR_GAMES_WEIGHT = 4              # last season's rating counts like this many games in week 1
PRIOR_REGRESSION = 0.5              # carry 50% of last season's rating (rest regresses to league mean)

# --- QB ratings ---
QB_PRIOR_DROPBACKS = 250            # shrinkage: a QB's rating is pulled toward replacement with this weight
QB_HALF_LIFE_DROPBACKS = 600        # older dropbacks count less (about 1 season of passing)
QB_REPLACEMENT_EPA = -0.12          # EPA/dropback assumed for unknown/backup QBs

# --- Elo ---
ELO_K = 20
ELO_HOME_ADV = 48                   # Elo points
ELO_SEASON_REGRESSION = 1 / 3       # pull toward 1505 between seasons
ELO_POINTS_PER_POINT = 25           # 25 Elo points ~ 1 point of spread

# --- model training ---
TRAIN_HALF_LIFE_SEASONS = 6         # a season 6 years back counts half as much (the game changes; HFA has shrunk)
RIDGE_ALPHA = 10.0

# --- dashboard ---
TIMEZONE = "America/Chicago"
