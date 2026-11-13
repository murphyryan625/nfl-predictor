"""Point the tracker at a throwaway folder BEFORE clv.config is imported, so tests can never
touch the real data/ files, and make the repo root importable."""
import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="clv_tests_"))
os.environ["CLV_DATA_DIR"] = str(_tmp / "data")
os.environ["CLV_SITE_DATA"] = str(_tmp / "site")
os.environ.pop("ODDS_API_KEY", None)                    # tests never call the network
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
