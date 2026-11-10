"""Bet rules: his limits, the 8-and-8 card, and grading."""
from dataclasses import replace

import numpy as np
import pandas as pd

from cbb import bets
from cbb.config import RULES, ModelParams


def _game(i, date, margin, home_spread, home_ml=-150, away_ml=130, bettable=True):
    return {"game_id": i, "date": pd.Timestamp(date), "home": f"H{i}", "away": f"A{i}",
            "margin": margin, "sigma": 10.0, "bettable": bettable,
            "close_home_spread": home_spread, "close_home_spread_price": -110.0,
            "close_away_spread_price": -110.0, "close_home_ml": home_ml, "close_away_ml": away_ml}


def test_no_edge_when_the_model_is_not_trusted():
    df = pd.DataFrame([_game(1, "2026-11-16", 10.0, -2.5)])
    c = bets.candidates(df, replace(ModelParams(), blend_weight=0.0))
    assert (c.ev < 0).all()                       # only the vig is left


def test_whole_number_spreads_are_never_bet():
    df = pd.DataFrame([_game(1, "2026-11-16", 12.0, -3.0)])
    c = bets.candidates(df, ModelParams())
    assert (c.market == "moneyline").all()


def test_moneyline_limits():
    df = pd.DataFrame([_game(1, "2026-11-16", -15.0, 12.5, home_ml=600, away_ml=-900)])
    c = bets.candidates(df, ModelParams())
    ml = c[c.market == "moneyline"]
    assert ml.price.between(RULES.ml_max_favorite, RULES.ml_max_underdog).all()


def test_unbettable_games_are_skipped():
    df = pd.DataFrame([_game(1, "2026-11-16", 12.0, -2.5, bettable=False)])
    assert len(bets.candidates(df, ModelParams())) == 0


def test_card_eight_per_window_first_come_first_served():
    rows = []
    for d, n in (("2026-11-16", 6), ("2026-11-18", 6), ("2026-11-21", 10)):     # Mon, Wed, Sat
        for k in range(n):
            rows.append({"game_id": len(rows), "date": pd.Timestamp(d), "p_bet": 0.6 + k / 100})
    c = bets.card(pd.DataFrame(rows))
    per = c.groupby("date").size().to_dict()
    assert per[pd.Timestamp("2026-11-16")] == 6       # Monday takes what it has
    assert per[pd.Timestamp("2026-11-18")] == 2       # Wednesday gets what is left of 8
    assert per[pd.Timestamp("2026-11-21")] == 8       # weekend has its own 8
    assert c[c.date == pd.Timestamp("2026-11-21")].p_bet.min() >= 0.62   # the best 8


def test_grade_spread_and_moneyline():
    b = pd.DataFrame({"game_id": [1, 1, 2], "market": ["spread", "spread", "moneyline"],
                      "side": ["home", "away", "away"], "line": [-3.5, 3.5, np.nan],
                      "price": [-110.0, -110.0, 150.0]})
    g = bets.grade(b, pd.Series({1: 4.0, 2: -1.0}))
    assert list(g.outcome) == ["W", "L", "W"]
    assert np.isclose(g.profit.iloc[2], 1.5)
