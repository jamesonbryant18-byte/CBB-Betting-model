"""
Player value, who is missing, and how much of a roster came back.

PLAYER VALUE (owner: "give more weight to players that provide more for the
team"). A player's value is his share of what his team produces when he plays:

    production  = Game Score (points, shooting, rebounds, assists, steals,
                  blocks, minus misses, turnovers and fouls), floored at 0
    value       = w * his share of team production + (1 - w) * his share of
                  team minutes, w = ModelParams.value_production_weight

averaged over his games for this team this season, shrunk toward last season's
value (two games' worth) so one hot night does not make a star. A team's
players add up to about 1.0, so a value of 0.25 means a quarter of the team.

MISSING VALUE, per team per game = the sum of value of ROTATION players who
did not play. A rotation player averages ROTATION_MINUTES or more when he plays
(his games before this one, shrunk toward last season like value). Deep bench
players are left out on purpose: they get in when a game is a blowout and sit
when it is close (2015-2019: players under 4 minutes a game played in 61% of
20-point games and 12% of the rest; players over 15 minutes, 90% either way),
so counting them as "missing" leaks the final score instead of measuring who
was hurt.
  miss_actual  who actually sat (used to adjust PAST games when rating teams)
  miss_proxy   who sat the team's previous game (what is knowable before tip;
               the live run adds his own injury notes on top)
  miss_fresh   the part of miss_proxy from absences FRESH_GAMES games old or
               less (news the ratings have not absorbed yet; a diagnostic
               scripts/metric_tests.py tests, not used in the margin)

CONTINUITY: the share of last season's production on this team that is back on
this season's roster (players listed in any box score so far, played or not).

Every number for a game uses only games before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_VALUE = 0.02             # below this a player is not counted as missing
ROTATION_MINUTES = 15.0      # and below this many minutes a game (see above)
FRESH_GAMES = 3              # an absence this many games old or less is "fresh"
SHRINK_GAMES = 2.0           # last season's value counts as this many games
MAX_MISSING = 0.7
CONTINUITY_DEFAULT = 0.5     # before a team's first box score of the season


def game_score(p: pd.DataFrame) -> pd.Series:
    return (p.pts + 0.4 * p.fgm - 0.7 * p.fga - 0.4 * (p.fta - p.ftm) + 0.7 * p.oreb
            + 0.3 * p.dreb + p.stl + 0.7 * p.ast + 0.7 * p.blk - 0.4 * p.pf - p.tov)


def _shares(players: pd.DataFrame, production_weight: float) -> pd.DataFrame:
    """Per player-game: production and minutes share of his team that game."""
    p = players.copy()
    p["prod"] = np.where(p.played, game_score(p.fillna(0)).clip(lower=0), 0.0)
    p["mins"] = np.where(p.played, p["min"].fillna(0), 0.0)
    tot = p.groupby(["game_id", "team_id"])[["prod", "mins"]].transform("sum")
    p["s_prod"] = np.where(tot["prod"] > 0, p["prod"] / tot["prod"].replace(0, np.nan), 0.0)
    p["s_min"] = np.where(tot["mins"] > 0, p["mins"] / tot["mins"].replace(0, np.nan), 0.0)
    p["share"] = production_weight * p.s_prod + (1 - production_weight) * p.s_min
    return p


def season_values(players: pd.DataFrame, production_weight: float) -> pd.DataFrame:
    """End-of-season value and minutes per (player, team): means over games played."""
    p = _shares(players, production_weight)
    p = p[p.played]
    v = p.groupby(["player_id", "team_id"]).agg(value=("share", "mean"), minutes=("mins", "mean"),
                                                 games=("share", "size"))
    return v.reset_index()


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "date", "tip", "status"]
    tg = pd.concat([
        games[cols + ["home_id"]].rename(columns={"home_id": "team_id"}),
        games[cols + ["away_id"]].rename(columns={"away_id": "team_id"}),
    ], ignore_index=True)
    tg["final"] = tg.status.eq("final")
    tg = tg.sort_values(["team_id", "date", "tip", "game_id"]).reset_index(drop=True)
    tg["k"] = tg.groupby("team_id").cumcount()
    return tg


def availability(games: pd.DataFrame, players: pd.DataFrame,
                 last_season: pd.DataFrame | None, production_weight: float) -> pd.DataFrame:
    """
    One season. Returns one row per (game_id, team_id):
      miss_actual, miss_proxy, continuity
    plus the player-level grid (for display) as attribute `.attrs["grid"]`.

    `games` must hold every game of the season in order (finals and scheduled);
    `players` the season's player rows; `last_season` = season_values(previous).
    """
    tg = _team_games(games)
    p = _shares(players, production_weight).merge(tg[["game_id", "team_id", "k"]], on=["game_id", "team_id"])
    first = p.groupby(["team_id", "player_id"]).k.min().rename("k_first").reset_index()
    grid = tg.merge(first, on="team_id")
    grid = grid[grid.k >= grid.k_first]
    grid = grid.merge(p[["game_id", "team_id", "player_id", "player", "played", "share", "mins"]],
                      on=["game_id", "team_id", "player_id"], how="left")
    grid["listed"] = grid.played.notna()
    # A team-game with no player rows at all is a missing box score, not a
    # night the whole roster sat: it says nothing about who played.
    grid["boxed"] = grid.groupby(["game_id", "team_id"]).listed.transform("any")
    grid["played"] = grid.played.fillna(False).astype(bool)
    grid["share"] = grid.share.where(grid.played, 0.0).fillna(0.0)
    grid["mins"] = grid.mins.where(grid.played, 0.0).fillna(0.0)
    grid = grid.sort_values(["team_id", "player_id", "k"]).reset_index(drop=True)

    g = grid.groupby(["team_id", "player_id"], sort=False)
    grid["n_prev"] = g.played.cumsum() - grid.played
    grid["sum_prev"] = g.share.cumsum() - grid.share
    grid["min_prev"] = g.mins.cumsum() - grid.mins
    # Did he play the team's most recent FINISHED game before this one? (A
    # Friday game predicted on Monday looks back to the last game actually
    # played, not to Wednesday's game that has not happened yet.)
    status = grid.played.astype(float).where(grid.final & grid.boxed)
    keys = [grid.team_id, grid.player_id]
    grid["played_last"] = status.groupby(keys, sort=False).shift(1)
    grid["played_last"] = grid.played_last.groupby(keys, sort=False).ffill()
    # games in a row he had missed going into this one (known games only)
    known = status.notna()
    kg = grid[known]
    kkeys = [kg.team_id, kg.player_id]
    run_id = status[known].eq(1.0).groupby(kkeys, sort=False).cumsum()
    missed_thru = status[known].eq(0.0).astype(float).groupby([*kkeys, run_id], sort=False).cumsum()
    grid["streak"] = missed_thru.reindex(grid.index).groupby(keys, sort=False).shift(1)
    grid["streak"] = grid.streak.groupby(keys, sort=False).ffill().fillna(0.0)

    last_any = pd.DataFrame(columns=["value", "minutes"], dtype=float)
    last_same = pd.Series(dtype=float)
    if last_season is not None and len(last_season):
        best = last_season.sort_values("games").drop_duplicates("player_id", keep="last")
        last_any = best.set_index("player_id")[["value", "minutes"]]
        last_same = last_season.set_index(["team_id", "player_id"]).value
    grid["last"] = grid.player_id.map(last_any.value).fillna(0.0)
    grid["last_min"] = grid.player_id.map(last_any.minutes).fillna(0.0)
    n = grid.n_prev
    grid["value"] = (grid.sum_prev + SHRINK_GAMES * grid["last"]) / (n + SHRINK_GAMES)
    grid["minutes"] = (grid.min_prev + SHRINK_GAMES * grid.last_min) / (n + SHRINK_GAMES)
    rot = (grid.value >= MIN_VALUE) & (grid.minutes >= ROTATION_MINUTES)

    grid["out_actual"] = rot & ~grid.played & grid.boxed
    grid["out_proxy"] = rot & grid.played_last.eq(0.0)
    fresh = grid.out_proxy & (grid.streak <= FRESH_GAMES)
    agg = grid.assign(a=grid.value * grid.out_actual, p=grid.value * grid.out_proxy, f=grid.value * fresh)
    out = agg.groupby(["game_id", "team_id"]).agg(miss_actual=("a", "sum"), miss_proxy=("p", "sum"),
                                                  miss_fresh=("f", "sum"))
    out = out.clip(upper=MAX_MISSING)

    if len(last_same):
        key = pd.MultiIndex.from_arrays([grid.team_id, grid.player_id])
        grid["last_same"] = last_same.reindex(key).fillna(0.0).to_numpy()
        before = grid[grid.k > grid.k_first]          # on the roster before this game
        cont = before.groupby(["game_id", "team_id"]).last_same.sum().clip(upper=1.0)
        out["continuity"] = cont
        has_prev = last_season.team_id.unique()
        out = out.reset_index()
        out["continuity"] = np.where(out.team_id.isin(has_prev),
                                     out.continuity.fillna(CONTINUITY_DEFAULT), 0.0)
    else:
        out = out.reset_index()
        out["continuity"] = CONTINUITY_DEFAULT
    out.attrs["grid"] = grid
    return out


def by_game(avail: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home/away columns of availability for each game."""
    a = avail.drop(columns=[], errors="ignore")
    g = games[["game_id", "home_id", "away_id"]]
    for side in ("home", "away"):
        x = a.rename(columns={"team_id": f"{side}_id", "miss_actual": f"{side}_miss_actual",
                              "miss_proxy": f"{side}_miss_proxy", "miss_fresh": f"{side}_miss_fresh",
                              "continuity": f"{side}_continuity"})
        g = g.merge(x, on=["game_id", f"{side}_id"], how="left")
    return g
