"""Settings for the CLV tracker. Change numbers here rather than in the code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# CLV_DATA_DIR / CLV_SITE_DATA let tests (or a dry run) write somewhere other than the real files.
DATA_DIR = Path(os.environ.get("CLV_DATA_DIR") or ROOT / "data")
SITE_DATA = Path(os.environ.get("CLV_SITE_DATA") or ROOT / "site" / "data")

PICKS_CSV = DATA_DIR / "clv_picks.csv"                # one row per bet (the durable record)
SNAPSHOTS_CSV = DATA_DIR / "clv_odds_snapshots.csv"   # append-only log of market lines seen on each run
REPORT_JSON = SITE_DATA / "clv.json"                  # what site/clv.html reads

# --- The Odds API (https://the-odds-api.com, free tier = 500 credits/month) ---
ODDS_API_KEY_ENV = "ODDS_API_KEY"                     # the key is read from this environment variable only
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_API_REGIONS = "us"
ODDS_API_MARKETS = "h2h,spreads"                      # each call costs (markets x regions) = 2 credits

# --- which books a pick is priced at ---
# The price logged for a pick should be one I could actually have bet. If any PICK_BOOKS are in the
# feed, the pick uses their line and the best price among them. If none are (Fanatics is only
# served to PAID Odds API plans), the pick is priced at the median of PROXY_BOOKS instead and
# labelled as a proxy. The closing line always uses every book: more books = a sharper fair price.
# Names are the feed's bookmaker titles.
PICK_BOOKS = ["Fanatics"]
PROXY_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "BetRivers", "Caesars"]   # regulated US books only

# --- closing line ---
# A market snapshot counts as "the close" only if it was taken this close to kickoff. Otherwise the
# closing line published by nflverse after the game is used.
CLOSE_WINDOW_MINUTES = 90

# --- picks ---
DEFAULT_STAKE = 1.0                                   # units per auto-logged pick

# --- spread / total distribution (see odds_math.py and key_numbers.py) ---
MARGIN_SIGMA = 12.7                                   # sd of (final margin - closing spread), 2015-2025
TOTAL_SIGMA = 13.2                                    # sd of (final total - closing total), 2015-2025

# --- report buckets: (label, lower bound inclusive, upper bound exclusive) ---
SPREAD_EDGE_BUCKETS = [("0-1 pts", 0, 1), ("1-2 pts", 1, 2), ("2-3.5 pts", 2, 3.5), ("3.5+ pts", 3.5, 1e9)]
ML_EDGE_BUCKETS = [("0-2%", 0, .02), ("2-5%", .02, .05), ("5-10%", .05, .10), ("10%+", .10, 1e9)]
