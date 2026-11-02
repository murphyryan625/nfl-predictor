"""Quarterback ratings and starting-QB detection.

Rating = shrunken EPA per dropback:
    (decayed EPA sum + PRIOR * replacement) / (decayed dropbacks + PRIOR)
so a QB with few dropbacks sits near replacement level, and a long-time starter's
rating is mostly his own recent play.

For completed games the schedule tells us who actually started (nflverse
`home_qb_id` / `away_qb_id`). For upcoming games we project the starter from the
latest depth chart, injury report and roster status — see `project_starters`.
"""
from __future__ import annotations

import bisect
import pandas as pd

from .config import QB_PRIOR_DROPBACKS, QB_HALF_LIFE_DROPBACKS, QB_REPLACEMENT_EPA


class QBRatings:
    """Answers: 'how good was QB x, using only games before date d?'"""

    def __init__(self, qb_games: pd.DataFrame, schedule: pd.DataFrame):
        dates = schedule.set_index("game_id")["gameday"]
        qg = qb_games.assign(gameday=qb_games["game_id"].map(dates)).dropna(subset=["gameday"])
        qg = qg.sort_values("gameday")
        self.history: dict[str, tuple[list, list]] = {}   # qb -> (dates, ratings after each game)
        self.names: dict[str, str] = {}
        state: dict[str, tuple[float, float]] = {}
        for r in qg.itertuples(index=False):
            s_epa, s_n = state.get(r.qb_id, (0.0, 0.0))
            decay = 0.5 ** (r.dropbacks / QB_HALF_LIFE_DROPBACKS)  # older play fades as new play arrives
            s_epa, s_n = s_epa * decay + r.epa_sum, s_n * decay + r.dropbacks
            state[r.qb_id] = (s_epa, s_n)
            d, v = self.history.setdefault(r.qb_id, ([], []))
            d.append(r.gameday)
            v.append((s_epa + QB_PRIOR_DROPBACKS * QB_REPLACEMENT_EPA) / (s_n + QB_PRIOR_DROPBACKS))
            self.names[r.qb_id] = r.name

    def rating(self, qb_id, before) -> float:
        if not isinstance(qb_id, str) or qb_id not in self.history:
            return QB_REPLACEMENT_EPA
        dates, vals = self.history[qb_id]
        i = bisect.bisect_left(dates, before) - 1   # last game strictly before `before`
        return vals[i] if i >= 0 else QB_REPLACEMENT_EPA

    def dropbacks_known(self, qb_id) -> bool:
        return isinstance(qb_id, str) and qb_id in self.history


# --------------------------------------------------------------------------
# Projecting this week's starters
# --------------------------------------------------------------------------
UNAVAILABLE = {"Out", "Doubtful"}
UNAVAILABLE_ROSTER = {"RES", "INA", "SUS", "CUT", "PUP", "NON", "EXE", "RET"}  # IR, inactive, suspended...


def project_starters(season: int, week: int, teams: list[str], schedule: pd.DataFrame) -> dict[str, dict]:
    """Best guess at each team's starting QB for `week`, with flags explaining it.

    Order of evidence:
      1. Latest depth chart QB1/QB2/QB3 for the team.
      2. Drop anyone listed Out/Doubtful on this week's injury report or on IR/inactive rosters.
      3. If the depth chart is missing, fall back to last game's starter.
    Flags tell the dashboard when something is off (backup starting, QB1 questionable, change from last week).
    """
    import nflreadpy as nfl
    from .data import TEAM_FIX

    def safe(fn):
        try:
            df = fn([season]).to_pandas()
            if "team" in df.columns:
                df["team"] = df["team"].replace(TEAM_FIX)
            return df
        except Exception as e:
            print(f"  warning: {fn.__name__} unavailable ({e})")
            return pd.DataFrame()

    depth, inj, ros = safe(nfl.load_depth_charts), safe(nfl.load_injuries), safe(nfl.load_rosters_weekly)

    # Latest depth chart snapshot per team, QBs only
    if not depth.empty:
        depth = depth[depth["pos_abb"] == "QB"].copy()
        depth["dt"] = pd.to_datetime(depth["dt"])
        latest = depth.groupby("team")["dt"].transform("max")
        depth = depth[depth["dt"] == latest].sort_values(["team", "pos_rank"])
    inj_wk = inj[(inj.get("week") == week) & (inj.get("position") == "QB")] if not inj.empty else inj
    ros_wk = ros[ros.get("week") == ros["week"].max()] if not ros.empty else ros

    played = schedule[(schedule.season == season) & schedule.home_score.notna()]
    out = {}
    for team in teams:
        flags = []
        # last game's actual starter
        tg = played[(played.home_team == team) | (played.away_team == team)].tail(1)
        last_id = last_name = None
        if len(tg):
            g = tg.iloc[0]
            side = "home" if g.home_team == team else "away"
            last_id, last_name = g[f"{side}_qb_id"], g[f"{side}_qb_name"]

        candidates = []
        if not depth.empty:
            candidates = list(depth[depth.team == team][["gsis_id", "player_name"]].itertuples(index=False, name=None))
        if not candidates and last_id is not None:
            candidates = [(last_id, last_name)]
            flags.append("no depth chart — using last game's starter")

        starter = None
        for i, (pid, name) in enumerate(candidates):
            status = None
            if not inj_wk.empty:
                r = inj_wk[inj_wk.gsis_id == pid]
                status = r["report_status"].iloc[0] if len(r) else None
            rstat = None
            if not ros_wk.empty:
                r = ros_wk[(ros_wk.gsis_id == pid) & (ros_wk.team == team)]
                rstat = r["status"].iloc[0] if len(r) else None
            if status in UNAVAILABLE or rstat in UNAVAILABLE_ROSTER:
                flags.append(f"{name} unavailable ({status or rstat})")
                continue
            if status == "Questionable":
                flags.append(f"{name} questionable")
            starter = (pid, name)
            if i > 0:
                flags.append("backup QB projected to start")
            break
        if starter is None:
            starter = (last_id, last_name)
            flags.append("could not confirm starter")
        if last_id is not None and starter[0] != last_id:
            flags.append(f"QB change from last game ({last_name})")
        out[team] = {"qb_id": starter[0], "qb_name": starter[1], "flags": flags}
    return out
