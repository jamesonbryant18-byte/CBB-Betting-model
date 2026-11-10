"""
The backtest: replay seasons one day at a time through the SAME Model the daily
run uses. There is no second implementation of the model anywhere in this repo.

For every date with games: rate every team on what was known the morning of
that date, predict that date's games, move on. The output is one row per game
with the projection, the closing line, the result, and the bettable flag.
"""
from __future__ import annotations

import time

import pandas as pd

from . import data, odds
from .config import FIRST_SEASON, RULES, ModelParams, load_params
from .model import Model


def load_inputs(seasons, first_season: int = FIRST_SEASON):
    """
    (games, players, lines) from `first_season` through the last of `seasons`.

    Always from the first season on file, whatever is being replayed: each
    season's preseason prior is last season's final rating, which came from
    the season before that, and so on. Starting the chain later changes every
    prediction after it, so the backtest, the tuning and the live run would
    each be a slightly different model.
    """
    seasons = list(seasons)
    start = first_season
    games, plays = data.load(range(start, max(seasons) + 1))
    lines = odds.closing_lines(games, range(max(2020, start), max(seasons) + 1))
    games = fill_ranks(games, lines)
    return games, plays, lines


def fill_ranks(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """AP ranks before 2022-23 come from Action Network's game records."""
    if not len(lines) or "close_an_home_rank" not in lines:
        return games
    r = lines.set_index("game_id")[["close_an_home_rank", "close_an_away_rank"]]
    g = games.copy()
    for side in ("home", "away"):
        fill = g.game_id.map(r[f"close_an_{side}_rank"])
        g[f"{side}_rank"] = g[f"{side}_rank"].fillna(fill.where(fill.between(1, 25)))
    return g


def bettable(g: pd.DataFrame) -> pd.Series:
    """
    Games he will bet: at least one Power-5 (incl. Big East) team or an AP
    top-25 team; regular season, plus the NCAA tournament. No conference
    tournaments, NIT or other postseason events.
    """
    power = g.home_conf.isin(RULES.power_conferences) | g.away_conf.isin(RULES.power_conferences)
    ranked = (g.home_rank <= RULES.max_rank) | (g.away_rank <= RULES.max_rank)
    conf_tourney = (~g.postseason & g.conf_game & g.notes.str.contains(
        "Tournament|Championship", case=False, regex=True))
    season_ok = (~g.postseason & ~conf_tourney) | g.ncaa
    return (power | ranked) & season_ok


def run(seasons, params: ModelParams | None = None, inputs=None,
        verbose: bool = True, model: Model | None = None,
        who_sits: str = "proxy") -> pd.DataFrame:
    """
    `who_sits`: "proxy" = what is knowable before tip (missed the last game),
    the only honest choice for a backtest of bets; "actual" = who really sat,
    used once, to measure how much perfect injury news would be worth.
    """
    params = params or load_params()
    games, plays, lines = inputs if inputs is not None else load_inputs(seasons)
    model = model or Model(games, plays, lines, params)
    out = []
    t0 = time.time()
    for s in seasons:
        g = model.season_games(s)
        fin = g[(g.status == "final") & g.home_score.notna()]
        missing = None
        if who_sits == "actual":
            a = model.availability(s)
            missing = pd.DataFrame({"game_id": a.index, "home_miss": a.home_miss_actual.to_numpy(),
                                    "away_miss": a.away_miss_actual.to_numpy()})
        for day, slate in fin.groupby("date"):
            state = model.rate(s, day)
            out.append(model.predict(slate, state, missing=missing))
        if verbose:
            print(f"  replayed {s}: {len(fin)} games ({time.time() - t0:.0f}s)", flush=True)
    pred = pd.concat(out, ignore_index=True)
    return attach_outcomes(pred, games, lines)


def attach_outcomes(pred: pd.DataFrame, games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    keep = ["game_id", "home", "away", "home_score", "away_score", "home_conf", "away_conf",
            "home_rank", "away_rank", "conf_game", "postseason", "ncaa", "notes", "tip"]
    out = pred.merge(games[keep], on="game_id", how="left")
    out["result"] = out.home_score - out.away_score
    out["bettable"] = bettable(out)
    if len(lines):
        out = out.merge(lines, on="game_id", how="left")
    return out
