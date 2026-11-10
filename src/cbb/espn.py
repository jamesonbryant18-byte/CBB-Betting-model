"""
ESPN, straight from the source, for the days the hoopR archive has not caught up.

The archive refreshes nightly, so a morning run can be missing last night's late
games, and its schedule still calls them "scheduled". `top_up` re-reads ESPN's
scoreboard for the last few days and today, and pulls the box score of any
finished game that has none. Every row it produces has exactly the columns
`data.py` produces from the archive (tests/test_data.py checks one game both ways).
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import CACHE, NCAA_TOURNAMENT_ID

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
LOOKBACK_DAYS = 3

_STAT_KEYS = {  # ESPN box-score key -> our column(s)
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb",), "defensiveRebounds": ("dreb",),
    "assists": ("ast",), "steals": ("stl",), "blocks": ("blk",),
    "totalTurnovers": ("tov",), "turnovers": ("tov",), "fouls": ("pf",),
    "points": ("pts",), "minutes": ("min",),
}


def _get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def today_et() -> pd.Timestamp:
    """Today's date in US Eastern time, the calendar every slate uses."""
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def _split(value: str, n: int) -> list:
    parts = str(value).split("-")
    if len(parts) != n:
        return [np.nan] * n
    return [pd.to_numeric(x, errors="coerce") for x in parts]


# ─────────────────────────── scoreboard ───────────────────────────

def scoreboard(date: pd.Timestamp) -> pd.DataFrame:
    """Every D1 game ESPN lists on an ET date, in data._schedule's columns."""
    d = _get(f"{BASE}/scoreboard?dates={date:%Y%m%d}&groups=50&limit=500")
    rows = []
    for e in d.get("events", []):
        c = e["competitions"][0]
        side = {x["homeAway"]: x for x in c["competitors"]}
        if "home" not in side or "away" not in side:
            continue
        h, a = side["home"], side["away"]
        tip = pd.to_datetime(e["date"], utc=True)
        name = c["status"]["type"]["name"]
        status = {"STATUS_FINAL": "final", "STATUS_SCHEDULED": "scheduled"}.get(name, "other")
        notes = " ".join(n.get("headline", "") for n in c.get("notes", []))

        def rank(x):
            r = (x.get("curatedRank") or {}).get("current")
            return np.nan if r in (None, 99) else float(r)

        rows.append({
            "game_id": int(e["id"]), "season": int(e["season"]["year"]),
            "date": tip.tz_convert("America/New_York").tz_localize(None).normalize(),
            "tip": tip,
            "home_id": int(h["id"]), "away_id": int(a["id"]),
            "home": h["team"].get("location", ""), "away": a["team"].get("location", ""),
            "home_full": h["team"].get("displayName", ""), "away_full": a["team"].get("displayName", ""),
            "home_score": float(h["score"]) if status == "final" else np.nan,
            "away_score": float(a["score"]) if status == "final" else np.nan,
            "neutral": bool(c.get("neutralSite", False)),
            "conf_game": bool(c.get("conferenceCompetition", False)),
            "home_conf": pd.to_numeric(h["team"].get("conferenceId"), errors="coerce"),
            "away_conf": pd.to_numeric(a["team"].get("conferenceId"), errors="coerce"),
            "home_rank": rank(h), "away_rank": rank(a),
            "postseason": int(e["season"].get("type", 2)) == 3,
            "ncaa": (str(c.get("tournamentId")) == str(NCAA_TOURNAMENT_ID)
                     or "Men's Basketball Championship" in notes),
            "notes": notes,
            "periods": float(c["status"].get("period") or np.nan) if status == "final" else np.nan,
            "status": status,
        })
    return pd.DataFrame(rows)


# ─────────────────────────── box score ───────────────────────────

def summary_box(game_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team rows, player rows) for one finished game, cached on disk."""
    path = CACHE / "espn" / f"summary_{game_id}.json"
    if path.exists():
        d = json.loads(path.read_text())
    else:
        d = _get(f"{BASE}/summary?event={game_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"boxscore": d.get("boxscore", {}), "header": d.get("header", {})}))
        time.sleep(0.2)
    teams, players = [], []
    for t in d.get("boxscore", {}).get("teams", []):
        row = {"game_id": int(game_id), "team_id": int(t["team"]["id"])}
        stats = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
        for key, cols in _STAT_KEYS.items():
            if key in stats and cols[0] not in row:
                vals = _split(stats[key], len(cols)) if len(cols) == 2 else [pd.to_numeric(stats[key], errors="coerce")]
                row.update(dict(zip(cols, vals)))
        teams.append(row)
    for t in d.get("boxscore", {}).get("players", []):
        team_id = int(t["team"]["id"])
        for block in t.get("statistics", []):
            keys = block.get("keys") or block.get("names") or []
            for a in block.get("athletes", []):
                row = {"game_id": int(game_id), "team_id": team_id,
                       "player_id": int(a["athlete"]["id"]),
                       "player": a["athlete"].get("displayName", ""),
                       "starter": bool(a.get("starter", False)),
                       "did_not_play": bool(a.get("didNotPlay", False))}
                for k, v in zip(keys, a.get("stats") or []):
                    cols = _STAT_KEYS.get(k)
                    if not cols:
                        continue
                    vals = _split(v, 2) if len(cols) == 2 else [pd.to_numeric(v, errors="coerce")]
                    row.update(dict(zip(cols, vals)))
                players.append(row)
    return pd.DataFrame(teams), pd.DataFrame(players)


# ─────────────────────────── top up the current season ───────────────────────────

def top_up(games: pd.DataFrame, box: pd.DataFrame, players: pd.DataFrame):
    """Refresh the last few days from ESPN and fill missing box scores."""
    from .data import BOX_STATS
    today = today_et()
    done = games[games.status == "final"]
    start = (done.date.max() if len(done) else today) - timedelta(days=LOOKBACK_DAYS)
    teams = set(games.home_id) | set(games.away_id)
    fresh = []
    for day in pd.date_range(start, today):
        try:
            sb = scoreboard(day)
        except Exception as ex:                            # noqa: BLE001
            print(f"  [espn] scoreboard {day:%Y-%m-%d} failed: {ex}")
            continue
        if len(sb):
            fresh.append(sb[sb.home_id.isin(teams) & sb.away_id.isin(teams)])
    if fresh:
        sb = pd.concat(fresh, ignore_index=True)
        games = pd.concat([games[~games.game_id.isin(sb.game_id)], sb[games.columns]],
                          ignore_index=True)
    missing = games[(games.status == "final") & (games.date < today + timedelta(days=1))
                    & ~games.game_id.isin(box.game_id)]
    new_t, new_p = [], []
    for gid in missing.game_id:
        try:
            t, p = summary_box(int(gid))
        except Exception as ex:                            # noqa: BLE001
            print(f"  [espn] box score {gid} failed: {ex}")
            continue
        new_t.append(t)
        new_p.append(p)
    if new_t:
        t = pd.concat(new_t, ignore_index=True)
        box = pd.concat([box, t.reindex(columns=["game_id", "team_id"] + BOX_STATS)], ignore_index=True)
    if new_p:
        p = pd.concat(new_p, ignore_index=True)
        p["played"] = p["min"].fillna(0).gt(0)
        p["season"] = int(games.season.max())
        players = pd.concat([players, p.reindex(columns=players.columns)], ignore_index=True)
    if len(missing):
        print(f"  [espn] topped up {len(new_t)} of {len(missing)} box scores the archive lacked")
    return games, box, players
