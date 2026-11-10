"""A small synthetic league with the exact columns data.load() produces."""
import numpy as np
import pandas as pd
import pytest

from cbb.data import BOX_STATS, possessions


def make_league(seasons=(2019, 2020), n_teams=24, days=50, seed=0):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 8, n_teams)
    team_ids = np.arange(100, 100 + n_teams)
    conf = np.repeat([2, 7, 99], n_teams // 3)
    games, players = [], []
    gid = 1
    for s in seasons:
        start = pd.Timestamp(f"{s - 1}-11-10")
        for d in range(days):
            date = start + pd.Timedelta(days=d)
            order = rng.permutation(n_teams)
            for i in range(0, n_teams - 1, 2):
                h, a = order[i], order[i + 1]
                neutral = rng.random() < 0.1
                mu = strength[h] - strength[a] + (0 if neutral else 3)
                poss = rng.normal(68, 4)
                margin = int(round(rng.normal(mu, 10)))
                margin = margin if margin != 0 else 1
                hs = int(round(poss * 1.05 + margin / 2))
                g = {"game_id": gid, "season": s, "date": date,
                     "tip": pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=23),
                     "home_id": team_ids[h], "away_id": team_ids[a],
                     "home": f"T{h}", "away": f"T{a}", "home_full": f"Team {h}", "away_full": f"Team {a}",
                     "home_score": float(hs), "away_score": float(hs - margin), "neutral": neutral,
                     "conf_game": conf[h] == conf[a], "home_conf": float(conf[h]), "away_conf": float(conf[a]),
                     "home_rank": np.nan, "away_rank": np.nan, "postseason": False, "ncaa": False,
                     "notes": "", "periods": 2.0, "status": "final"}
                for side, t in (("home", h), ("away", a)):
                    fga, oreb, tov, fta = poss * 0.85, 9.0, 12.0, 18.0
                    g.update({f"{side}_fga": fga, f"{side}_oreb": oreb, f"{side}_tov": tov, f"{side}_fta": fta})
                    for st in BOX_STATS:
                        g.setdefault(f"{side}_{st}", 5.0)
                    for j in range(10):
                        pid = team_ids[t] * 100 + j
                        dnp = rng.random() < (0.5 if j == 0 and d > days // 2 else 0.05)
                        mins = 0.0 if dnp else float(max(0, rng.normal(32 - 3 * j, 3)))
                        players.append({"game_id": gid, "team_id": team_ids[t], "player_id": pid,
                                        "player": f"P{pid}", "starter": j < 5, "did_not_play": dnp,
                                        **{k: (0.0 if dnp else float(rng.poisson(2))) for k in BOX_STATS},
                                        "pts": 0.0 if dnp else float(rng.poisson(18 - 1.5 * j)),
                                        "min": mins, "played": mins > 0, "season": s})
                games.append(g)
                gid += 1
    games = pd.DataFrame(games)
    ph = possessions(games.home_fga, games.home_oreb, games.home_tov, games.home_fta)
    pa = possessions(games.away_fga, games.away_oreb, games.away_tov, games.away_fta)
    games["poss"] = (ph + pa) / 2
    games["poss40"] = games["poss"]
    games["has_box"] = True
    lines = pd.DataFrame({"game_id": games.game_id,
                          "close_home_spread": -(games.home_score - games.away_score).clip(-20, 20) / 2})
    return games, pd.DataFrame(players), lines


@pytest.fixture(scope="session")
def league():
    return make_league()
