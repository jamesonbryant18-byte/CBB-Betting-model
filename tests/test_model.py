"""The model's core promises: no peeking, one code path, signs that read right."""
from dataclasses import replace

import numpy as np
import pandas as pd

from cbb.config import ModelParams
from cbb.model import Model

PARAMS = replace(ModelParams(), market_lambda=5.0, avail_points=10.0)


def _blank_from(games, players, lines, asof):
    """What a live run on `asof` would have: the schedule, but no results yet."""
    g = games.copy()
    future = g.date >= asof
    g.loc[future, ["home_score", "away_score", "poss", "poss40", "periods"]] = np.nan
    g.loc[future, "status"] = "scheduled"
    g.loc[future, "has_box"] = False
    p = players[~players.game_id.isin(g.game_id[future])]
    ln = lines[~lines.game_id.isin(g.game_id[future])]
    return g, p, ln


def test_prediction_ignores_everything_on_or_after_the_date(league):
    games, players, lines = league
    asof = games[games.season == 2020].date.sort_values().iloc[len(games[games.season == 2020]) // 2]
    slate = games[(games.season == 2020) & (games.date == asof)]

    full = Model(games, players, lines, PARAMS)
    a = full.predict(slate, full.rate(2020, asof))

    g, p, ln = _blank_from(games, players, lines, asof)
    live = Model(g, p, ln, PARAMS)
    b = live.predict(g[g.game_id.isin(slate.game_id)], live.rate(2020, asof))

    cols = ["margin", "p_home", "r_home", "r_away", "hfa", "home_miss", "away_miss", "poss"]
    np.testing.assert_allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-9)


def test_replay_and_live_are_the_same_call(league):
    """replay.run must produce exactly what Model.rate + Model.predict produce."""
    from cbb import replay
    games, players, lines = league
    r = replay.run([2020], PARAMS, inputs=(games, players, lines), verbose=False)
    day = games[games.season == 2020].date.sort_values().unique()[20]
    m = Model(games, players, lines, PARAMS)
    direct = m.predict(games[(games.season == 2020) & (games.date == day)], m.rate(2020, day))
    got = r[r.date == day].set_index("game_id").loc[direct.game_id]
    np.testing.assert_allclose(got.margin.to_numpy(), direct.margin.to_numpy(), atol=1e-12)


def test_better_team_is_favored_and_home_court_helps(league):
    games, players, lines = league
    m = Model(games, players, lines, replace(ModelParams(), market_lambda=0.0))
    s = m.rate(2020, games[games.season == 2020].date.max())
    r = s.rating.as_series()
    best, worst = r.idxmax(), r.idxmin()
    g = pd.DataFrame({"game_id": [1, 2], "season": 2020, "date": s.asof,
                      "home_id": [best, worst], "away_id": [worst, best], "neutral": [False, False]})
    pr = m.predict(g, s, missing=pd.DataFrame({"game_id": [1, 2], "home_miss": 0.0, "away_miss": 0.0}))
    assert pr.margin.iloc[0] > 0 > pr.margin.iloc[1]
    assert s.rating.extra > 0                                   # home court is an advantage
    assert abs(pr.margin.iloc[0] + pr.margin.iloc[1] - 2 * s.rating.extra) < 1e-9


def test_a_missing_player_lowers_his_team(league):
    games, players, lines = league
    m = Model(games, players, lines, PARAMS)
    s = m.rate(2020, games[games.season == 2020].date.max())
    g = pd.DataFrame({"game_id": [1, 1], "season": 2020, "date": s.asof,
                      "home_id": [100, 100], "away_id": [101, 101], "neutral": False})
    full = m.predict(g.iloc[:1], s, missing=pd.DataFrame({"game_id": [1], "home_miss": [0.0], "away_miss": [0.0]}))
    short = m.predict(g.iloc[:1], s, missing=pd.DataFrame({"game_id": [1], "home_miss": [0.25], "away_miss": [0.0]}))
    assert np.isclose(full.margin.iloc[0] - short.margin.iloc[0], 0.25 * PARAMS.avail_points)
