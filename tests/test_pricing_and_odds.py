"""Signs, de-vig and the line flip: the places a betting model silently goes wrong."""
import numpy as np
import pandas as pd

from cbb import odds, pricing


def test_devig_even_market_is_fifty_fifty():
    a, b = pricing.devig(-110, -110)
    assert np.isclose(a, 0.5) and np.isclose(b, 0.5)


def test_devig_favorite():
    a, b = pricing.devig(-200, 170)
    assert a > 0.62 and np.isclose(a + b, 1.0)


def test_half_point_cover_equals_normal_curve():
    from scipy.stats import norm
    c, push = pricing.cover_prob(5.0, 10.0, -3.5)      # home -3.5, projected +5
    assert np.isclose(c[0], norm.cdf((5.0 - 3.5) / 10.0), atol=1e-3)
    assert push[0] == 0.0


def test_whole_number_line_has_push_mass():
    c, push = pricing.cover_prob(3.0, 10.0, -3.0)
    assert 0.02 < push[0] < 0.06


def test_no_ties_in_win_probability():
    assert np.isclose(pricing.win_prob(0.0, 10.0)[0], 0.5)


def test_ticket_sign_convention():
    # LSU +8.5 at home against Alabama: the line says LSU loses by 8.5
    g = pd.DataFrame({"game_id": [1], "date": [pd.Timestamp("2024-02-10")], "home_id": [99], "away_id": [333]})
    o = pd.DataFrame({"an_id": [7], "book": ["fanduel"], "an_home": [981], "an_away": [987],
                      "date": [pd.Timestamp("2024-02-10")], "home_spread": [8.5],
                      "home_spread_price": [-120.0], "away_spread_price": [-102.0],
                      "home_ml": [275.0], "away_ml": [-350.0], "total": [168.5],
                      "captured": [pd.NaT], "start": [pd.NaT], "an_home_rank": [np.nan],
                      "an_away_rank": [np.nan]})
    straight = odds.attach(g, o, "fanduel", cw={981: 99, 987: 333})
    assert straight.home_spread.iloc[0] == 8.5 and straight.home_ml.iloc[0] == 275


def test_line_is_flipped_when_sources_disagree_on_home():
    """A neutral game Action Network lists the other way round."""
    g = pd.DataFrame({"game_id": [1], "date": [pd.Timestamp("2024-02-10")], "home_id": [333], "away_id": [99]})
    o = pd.DataFrame({"an_id": [7], "book": ["fanduel"], "an_home": [981], "an_away": [987],
                      "date": [pd.Timestamp("2024-02-10")], "home_spread": [8.5],
                      "home_spread_price": [-120.0], "away_spread_price": [-102.0],
                      "home_ml": [275.0], "away_ml": [-350.0], "total": [168.5],
                      "captured": [pd.NaT], "start": [pd.NaT], "an_home_rank": [np.nan],
                      "an_away_rank": [np.nan]})
    f = odds.attach(g, o, "fanduel", cw={981: 99, 987: 333}).iloc[0]
    # ESPN's home team (333) is Action Network's away team: Alabama -8.5
    assert f.home_spread == -8.5
    assert f.home_spread_price == -102.0 and f.away_spread_price == -120.0
    assert f.home_ml == -350.0 and f.away_ml == 275.0


def test_v2_main_line_not_alternate():
    """v2 lists alternates beside the main line (real FanDuel shape, VT-Providence 2025-11-08)."""
    from cbb.odds import _main_market
    o = lambda m, side, value, odds: {"market_id": m, "side": side, "value": value, "odds": odds, "is_live": False}
    spread = [o(1, "home", 1.5, -112), o(1, "away", -1.5, -118),          # main
              o(2, "home", -1.5, 120), o(2, "home", 4.5, -200),           # alternates, one side each
              o(3, "home", -10.5, 400), o(3, "away", 10.5, -600)]         # a mirrored alternate pair
    main = _main_market(spread, "home", "away", opposite=True)
    assert main["home"]["value"] == 1.5 and main["away"]["odds"] == -118
    assert _main_market([o(2, "home", -1.5, 120)], "home", "away", opposite=True) == {}
