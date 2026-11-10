"""Who counts as missing (players.py)."""
import numpy as np
import pandas as pd

from cbb import players as pl
from cbb.data import BOX_STATS


def _season(sat: dict, no_box: set = frozenset(), n_games: int = 8):
    """One team (id 1) vs a rotating opponent. Player 10 is a 32-minute starter,
    player 19 a 5-minute bench player. `sat` = {game_id: [player ids who sat]}."""
    games, rows = [], []
    for k in range(n_games):
        gid = 100 + k
        games.append({"game_id": gid, "date": pd.Timestamp("2026-11-03") + pd.Timedelta(days=3 * k),
                      "tip": pd.Timestamp("2026-11-03", tz="UTC") + pd.Timedelta(days=3 * k),
                      "status": "final", "home_id": 1, "away_id": 50 + k})
        if gid in no_box:
            continue
        for pid, mins in [(10, 32.0), *[(11 + j, 30.0 - 2 * j) for j in range(7)], (19, 5.0)]:
            out = pid in sat.get(gid, [])
            rows.append({"game_id": gid, "team_id": 1, "player_id": pid, "player": f"P{pid}",
                         "starter": pid < 15, "did_not_play": out,
                         **{s: (0.0 if out else 2.0) for s in BOX_STATS},
                         "pts": 0.0 if out else (20.0 if pid == 10 else 6.0),
                         "min": 0.0 if out else mins, "played": not out})
    return pd.DataFrame(games), pd.DataFrame(rows)


def _miss(games, players):
    a = pl.availability(games, players, None, 1.0)
    return a[a.team_id == 1].set_index("game_id")


def test_starter_who_sits_is_missing_then_proxied():
    m = _miss(*_season({105: [10]}))
    assert m.loc[105, "miss_actual"] > 0.1           # he sat game 105
    assert m.loc[104, "miss_actual"] == 0
    assert m.loc[106, "miss_proxy"] > 0.1            # known before 106: he missed the last game
    assert m.loc[105, "miss_proxy"] == 0             # not knowable before 105


def test_fresh_absence_fades_after_a_few_games():
    m = _miss(*_season({g: [10] for g in range(102, 108)}))   # out from game 102 on
    assert m.loc[103, "miss_fresh"] > 0.1                    # 1 game missed so far
    assert m.loc[105, "miss_fresh"] > 0.1                    # 3 missed
    assert m.loc[106, "miss_fresh"] == 0 and m.loc[106, "miss_proxy"] > 0.1   # 4 missed: old news


def test_deep_bench_player_never_counts():
    """He gets in only in some games; that is game script, not injury news."""
    m = _miss(*_season({103: [19], 104: [19], 105: [19]}))
    assert np.allclose(m.loc[[103, 104, 105, 106], ["miss_actual", "miss_proxy"]], 0.0)


def test_missing_box_score_is_not_a_missing_roster():
    m = _miss(*_season({}, no_box={104}))
    assert m.reindex([104, 105]).fillna(0)[["miss_actual", "miss_proxy"]].to_numpy().max() == 0.0
