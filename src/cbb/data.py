"""
Game results, box scores and player minutes.

Source: ESPN, by way of the sportsdataverse hoopR archive on GitHub (one file
per season, 2003 to today, refreshed nightly). The current season is topped up
straight from ESPN for any finished game the archive has not caught yet, through
`espn.py`, which produces exactly the same columns.

Two tables come out, and everything else is built on them:

  games    one row per game: date, teams, score, neutral court, conferences,
           AP ranks, NCAA tournament flag, and each side's box-score totals
  players  one row per player per game: minutes, box score, played or not

Only Division I vs Division I games are kept. A team is D1 in a season if it
plays at least `MIN_D1_GAMES` games that season (non-D1 visitors play one or two).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE, CURRENT_SEASON, NCAA_TOURNAMENT_ID

HOOPR = ("https://github.com/sportsdataverse/hoopR-mbb-data/raw/main/mbb/"
         "{kind}/parquet/{prefix}_{season}.parquet")
KINDS = {"schedules": "mbb_schedule", "team_box": "team_box", "player_box": "player_box"}
MIN_D1_GAMES = 10
CURRENT_MAX_AGE_HOURS = 6.0

BOX_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast",
             "stl", "blk", "tov", "pf"]

_TEAM_BOX_COLS = {
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "assists": "ast",
    "steals": "stl", "blocks": "blk", "total_turnovers": "tov", "fouls": "pf",
}
_PLAYER_BOX_COLS = {
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "assists": "ast",
    "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf",
    "points": "pts", "minutes": "min",
}


# ─────────────────────────── download + cache ───────────────────────────

def _cache_path(kind: str, season: int) -> Path:
    return CACHE / "hoopr" / f"{kind}_{season}.parquet"


def fetch(kind: str, season: int, refresh: bool = False) -> Path:
    """Download one season file if missing (or stale, for the current season)."""
    path = _cache_path(kind, season)
    stale = (season == CURRENT_SEASON and path.exists()
             and time.time() - path.stat().st_mtime > CURRENT_MAX_AGE_HOURS * 3600)
    if path.exists() and not refresh and not stale:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = HOOPR.format(kind=kind, prefix=KINDS[kind], season=season)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    pd.read_parquet(tmp, columns=["game_id"])          # refuse a truncated file
    tmp.replace(path)
    return path


# ─────────────────────────── normalize ───────────────────────────

def _rank(s: pd.DataFrame, col: str) -> pd.Series:
    if col not in s:
        return pd.Series(np.nan, index=s.index)
    return pd.to_numeric(s[col], errors="coerce").where(lambda r: r.between(1, 25))


def _schedule(season: int) -> pd.DataFrame:
    s = pd.read_parquet(fetch("schedules", season))
    status = s["status_type_name"].map(
        {"STATUS_FINAL": "final", "STATUS_SCHEDULED": "scheduled"}).fillna("other")
    tip = pd.to_datetime(s["start_date"], utc=True, errors="coerce")
    notes = s["notes_headline"].fillna("").astype(str)
    out = pd.DataFrame({
        "game_id": s["game_id"].astype("int64"),
        "season": season,
        "date": tip.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize(),
        "tip": tip,
        "home_id": s["home_id"].astype("int64"),
        "away_id": s["away_id"].astype("int64"),
        "home": s["home_location"].astype(str),
        "away": s["away_location"].astype(str),
        "home_full": s["home_display_name"].astype(str),
        "away_full": s["away_display_name"].astype(str),
        "home_score": pd.to_numeric(s["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(s["away_score"], errors="coerce"),
        "neutral": s["neutral_site"].fillna(False).astype(bool),
        "conf_game": s["conference_competition"].fillna(False).astype(bool),
        "home_conf": pd.to_numeric(s["home_conference_id"], errors="coerce"),
        "away_conf": pd.to_numeric(s["away_conference_id"], errors="coerce"),
        # AP rank at game time; the archive has it from 2022-23 on, odds.py
        # fills earlier seasons from Action Network.
        "home_rank": _rank(s, "home_current_rank"),
        "away_rank": _rank(s, "away_current_rank"),
        "postseason": s["season_type"].eq(3),
        "ncaa": (pd.to_numeric(s["tournament_id"], errors="coerce").eq(NCAA_TOURNAMENT_ID)
                 | notes.str.contains("Men's Basketball Championship", regex=False)),
        "notes": notes,
        "periods": pd.to_numeric(s["status_period"], errors="coerce"),
        "status": status,
    })
    out.loc[out.status != "final", ["home_score", "away_score", "periods"]] = np.nan
    return out.drop_duplicates("game_id")


def _team_box(season: int) -> pd.DataFrame:
    t = pd.read_parquet(fetch("team_box", season))
    if "total_turnovers" not in t:
        t["total_turnovers"] = t["turnovers"]
    t = t.rename(columns=_TEAM_BOX_COLS)
    cols = ["game_id", "team_id"] + BOX_STATS
    t = t[cols].copy()
    t[BOX_STATS] = t[BOX_STATS].apply(pd.to_numeric, errors="coerce")
    t["game_id"] = t["game_id"].astype("int64")
    t["team_id"] = t["team_id"].astype("int64")
    return t.drop_duplicates(["game_id", "team_id"])


def _player_box(season: int) -> pd.DataFrame:
    p = pd.read_parquet(fetch("player_box", season))
    p = p.rename(columns=_PLAYER_BOX_COLS)
    stats = BOX_STATS + ["pts", "min"]
    out = p[["game_id", "team_id", "athlete_id", "athlete_display_name", "starter",
             "did_not_play"] + stats].copy()
    out = out.rename(columns={"athlete_id": "player_id", "athlete_display_name": "player"})
    out[stats] = out[stats].apply(pd.to_numeric, errors="coerce")
    out["played"] = out["min"].fillna(0).gt(0)
    out["season"] = season
    for c in ("game_id", "team_id", "player_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["game_id", "team_id", "player_id"])
    for c in ("game_id", "team_id", "player_id"):
        out[c] = out[c].astype("int64")
    return out.drop_duplicates(["game_id", "player_id"])


def possessions(fga, oreb, tov, fta):
    """Standard possession estimate (the 0.475 free-throw weight KenPom uses)."""
    return fga - oreb + tov + 0.475 * fta


def _attach_box(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    for side in ("home", "away"):
        b = box.rename(columns={s: f"{side}_{s}" for s in BOX_STATS})
        games = games.merge(b.rename(columns={"team_id": f"{side}_id"}),
                            on=["game_id", f"{side}_id"], how="left")
    poss_h = possessions(games.home_fga, games.home_oreb, games.home_tov, games.home_fta)
    poss_a = possessions(games.away_fga, games.away_oreb, games.away_tov, games.away_fta)
    games["poss"] = (poss_h + poss_a) / 2
    minutes = 40 + 5 * (games["periods"].fillna(2) - 2).clip(lower=0)
    games["poss40"] = games["poss"] * 40 / minutes       # pace per 40 minutes
    games["has_box"] = games["poss"].notna() & games["poss"].gt(40)
    return games


def _d1_only(games: pd.DataFrame) -> pd.DataFrame:
    n = pd.concat([games.home_id, games.away_id]).value_counts()
    d1 = set(n[n >= MIN_D1_GAMES].index)
    return games[games.home_id.isin(d1) & games.away_id.isin(d1)]


def load_season(season: int, top_up: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, players) for one season, D1 vs D1 only."""
    games = _d1_only(_schedule(season))
    box = _team_box(season)
    players = _player_box(season)
    if top_up and season == CURRENT_SEASON:
        from . import espn
        games, box, players = espn.top_up(games, box, players)
    games = _attach_box(games, box)
    games = games.sort_values(["date", "tip", "game_id"]).reset_index(drop=True)
    players = players[players.game_id.isin(games.game_id)].reset_index(drop=True)
    return games, players


def load(seasons, top_up: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Many seasons stacked. Cached per season on disk as parquet after the first build."""
    gs, ps = [], []
    for s in seasons:
        built = CACHE / "built" / f"{s}.parquet"
        built_p = CACHE / "built" / f"{s}_players.parquet"
        if s != CURRENT_SEASON and built.exists() and built_p.exists():
            g, p = pd.read_parquet(built), pd.read_parquet(built_p)
        else:
            g, p = load_season(s, top_up=top_up)
            if s != CURRENT_SEASON:
                built.parent.mkdir(parents=True, exist_ok=True)
                g.to_parquet(built)
                p.to_parquet(built_p)
        gs.append(g)
        ps.append(p)
    games = pd.concat(gs, ignore_index=True)
    players = pd.concat(ps, ignore_index=True)
    return games, players


def finals(games: pd.DataFrame) -> pd.DataFrame:
    """Played games with a score."""
    return games[(games.status == "final") & games.home_score.notna()]
