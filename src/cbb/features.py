"""
Advanced metrics, as they stood the morning of each game.

Every value for a game is computed only from games before that game's date.
That is the whole difference between a backtest and a fantasy: end-of-season
KenPom or Torvik numbers would hand a December game its own future.

Per team, opponent-adjusted (a ridge fit like the ratings, offense and defense
separately, recent games weighted more):
  efg      effective field-goal %          (fgm + 0.5 * 3pm) / fga
  tov      turnover %                      turnovers / possessions
  orb      offensive rebound %             oreb / (oreb + opponent dreb)
  ftr      free-throw rate                 fta / fga
  3par     three-point attempt rate        3pa / fga

Shooting luck, points per game (KenPom: defenses barely control opponents'
3-point and free-throw percentages):
  def3_luck  points opponents made above league-average 3P% on their 3PA
  off3_luck  points this team made above league-average 3P% on its 3PA
  ft_luck    points opponents made above league-average FT% on their FTA

Schedule and form:
  rest       days since the team's last game (capped at 7)
  form       how far the team beat its projections over its last 5 games
  home_edge  this home team's average beat-the-projection at home, the 3
             seasons before (shrunk; 0 on a neutral court)

Each feature is returned as home minus away (`f_<name>`), plus the raw sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .ratings import recency_weights

FACTORS = ["efg", "tov", "orb", "ftr", "3par"]
FACTOR_PENALTY = 6.0
HALF_LIFE = 60.0
LUCK_SHRINK_GAMES = 5.0


def team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per finished game: each team's offensive line and its opponent's."""
    g = games[(games.status == "final") & games.has_box]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = pd.DataFrame({
            "game_id": g.game_id, "date": g.date, "season": g.season,
            "team_id": g[f"{side}_id"], "opp_id": g[f"{opp}_id"],
            "fgm": g[f"{side}_fgm"], "fga": g[f"{side}_fga"], "fg3m": g[f"{side}_fg3m"],
            "fg3a": g[f"{side}_fg3a"], "ftm": g[f"{side}_ftm"], "fta": g[f"{side}_fta"],
            "oreb": g[f"{side}_oreb"], "opp_dreb": g[f"{opp}_dreb"], "tov": g[f"{side}_tov"],
            "poss": g.poss,
        })
        rows.append(r)
    t = pd.concat(rows, ignore_index=True)
    t["efg"] = (t.fgm + 0.5 * t.fg3m) / t.fga
    t["tov"] = t.tov / t.poss
    t["orb"] = t.oreb / (t.oreb + t.opp_dreb)
    t["ftr"] = t.fta / t.fga
    t["3par"] = t.fg3a / t.fga
    return t.replace([np.inf, -np.inf], np.nan)


def _adjusted(rows: pd.DataFrame, stat: str, team_ids: np.ndarray, w: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Opponent-adjusted offense and defense for one stat: y = mean + off[team] + def[opp]."""
    n = len(team_ids)
    idx = pd.Series(np.arange(n), index=team_ids)
    ok = rows[stat].notna().to_numpy()
    t = idx.reindex(rows.team_id.to_numpy()).to_numpy()
    o = idx.reindex(rows.opp_id.to_numpy()).to_numpy()
    ok &= ~(np.isnan(t) | np.isnan(o))
    y = rows[stat].to_numpy()[ok]
    mu = float(np.average(y, weights=w[ok])) if ok.any() else 0.0
    m = int(ok.sum())
    X = sparse.csr_matrix((np.ones(2 * m), (np.r_[np.arange(m), np.arange(m)],
                                            np.r_[t[ok].astype(int), n + o[ok].astype(int)])),
                          shape=(m, 2 * n))
    XtW = X.T.multiply(w[ok]).tocsr()
    A = (XtW @ X).toarray() + np.eye(2 * n) * FACTOR_PENALTY
    beta = np.linalg.solve(A, XtW @ (y - mu))
    return pd.Series(beta[:n], index=team_ids), pd.Series(beta[n:], index=team_ids)


def season_features(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Morning-of-game features for every game of one season."""
    g = games[games.season == season].sort_values(["date", "tip", "game_id"])
    rows = team_rows(g)
    team_ids = np.unique(np.r_[g.home_id.to_numpy(), g.away_id.to_numpy()])
    out = []
    for day, slate in g.groupby("date"):
        past = rows[rows.date < day]
        f = {"game_id": slate.game_id.to_numpy()}
        if len(past) >= 50:
            w = recency_weights(past.date, day, HALF_LIFE)
            for stat in FACTORS:
                off, dfn = _adjusted(past, stat, team_ids, w)
                for side in ("home", "away"):
                    ids = slate[f"{side}_id"].to_numpy()
                    f[f"{side}_off_{stat}"] = off.reindex(ids).to_numpy()
                    f[f"{side}_def_{stat}"] = dfn.reindex(ids).to_numpy()
        out.append(pd.DataFrame(f))
    feats = pd.concat(out, ignore_index=True)
    feats = feats.merge(_luck(g, rows), on="game_id", how="left")
    feats = feats.merge(_rest(g), on="game_id", how="left")
    return feats


def _luck(g: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Shooting luck per team, from its earlier games this season, shrunk toward 0."""
    r = rows.copy()
    # League averages from earlier days only (a season-wide average would let
    # a November game see March).
    day = r.groupby("date")[["fg3m", "fg3a", "ftm", "fta"]].sum().sort_index()
    before = day.cumsum().shift(1)
    lg3 = (before.fg3m / before.fg3a).fillna(0.34)
    lgft = (before.ftm / before.fta).fillna(0.70)
    r["pts3_above"] = 3 * (r.fg3m - r.date.map(lg3) * r.fg3a)    # this team's offense
    r["ft_above"] = r.ftm - r.date.map(lgft) * r.fta
    # the same quantities from the opponent's point of view = this team's defense
    opp = r[["game_id", "team_id", "pts3_above", "ft_above"]].rename(
        columns={"team_id": "opp_id", "pts3_above": "opp3_above", "ft_above": "oppft_above"})
    r = r.merge(opp, on=["game_id", "opp_id"])
    r = r.sort_values(["team_id", "date", "game_id"])
    gb = r.groupby("team_id")
    n = gb.cumcount()
    out = pd.DataFrame({"game_id": r.game_id, "team_id": r.team_id})
    for col, name in (("pts3_above", "off3_luck"), ("opp3_above", "def3_luck"), ("oppft_above", "ft_luck")):
        prev = gb[col].cumsum() - r[col]
        out[name] = prev / (n + LUCK_SHRINK_GAMES)
    return _sides(g, out, ["off3_luck", "def3_luck", "ft_luck"])


def _rest(g: pd.DataFrame) -> pd.DataFrame:
    t = pd.concat([g[["game_id", "date", "home_id"]].rename(columns={"home_id": "team_id"}),
                   g[["game_id", "date", "away_id"]].rename(columns={"away_id": "team_id"})])
    t = t.sort_values(["team_id", "date", "game_id"])
    t["rest"] = t.groupby("team_id").date.diff().dt.days.clip(upper=7).fillna(7)
    return _sides(g, t[["game_id", "team_id", "rest"]], ["rest"])


def _sides(g: pd.DataFrame, per_team: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = g[["game_id", "home_id", "away_id"]]
    for side in ("home", "away"):
        x = per_team.rename(columns={"team_id": f"{side}_id", **{c: f"{side}_{c}" for c in cols}})
        out = out.merge(x, on=["game_id", f"{side}_id"], how="left")
    return out.drop(columns=["home_id", "away_id"])


def add_residual_features(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Features built from the model's own earlier misses (needs replay output):
    form (last 5 games) and team-specific home edge (previous 3 seasons).
    """
    p = pred.sort_values(["date", "game_id"]).copy()
    p["resid"] = p.result - p.margin
    t = pd.concat([
        pd.DataFrame({"game_id": p.game_id, "date": p.date, "team_id": p.home_id, "r": p.resid}),
        pd.DataFrame({"game_id": p.game_id, "date": p.date, "team_id": p.away_id, "r": -p.resid}),
    ]).sort_values(["team_id", "date", "game_id"])
    seasons = p.set_index("game_id").season
    t["season"] = t.game_id.map(seasons)
    gb = t.groupby(["team_id", "season"]).r
    t["form"] = gb.transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    p = p.merge(t[["game_id", "team_id", "form"]].rename(columns={"team_id": "home_id", "form": "home_form"}),
                on=["game_id", "home_id"], how="left")
    p = p.merge(t[["game_id", "team_id", "form"]].rename(columns={"team_id": "away_id", "form": "away_form"}),
                on=["game_id", "away_id"], how="left")

    home = p[~p.neutral].groupby(["home_id", "season"]).resid.agg(["sum", "count"]).reset_index()
    edges = []
    for s in sorted(p.season.unique()):
        prev = home[home.season.between(s - 3, s - 1)].groupby("home_id")[["sum", "count"]].sum()
        e = (prev["sum"] / (prev["count"] + 50.0)).rename("home_edge").reset_index()
        e["season"] = s
        edges.append(e)
    p = p.merge(pd.concat(edges), on=["home_id", "season"], how="left")
    p["home_edge"] = np.where(p.neutral, 0.0, p.home_edge.fillna(0.0))
    return p


def diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away versions of every two-sided feature, as f_<name>."""
    out = pd.DataFrame(index=df.index)
    for stat in FACTORS:
        for side in ("off", "def"):
            h, a = f"home_{side}_{stat}", f"away_{side}_{stat}"
            if h in df:
                out[f"f_{side}_{stat}"] = df[h] - df[a]
    for name in ("off3_luck", "def3_luck", "ft_luck", "rest", "form", "miss_fresh"):
        if f"home_{name}" in df:
            out[f"f_{name}"] = df[f"home_{name}"] - df[f"away_{name}"]
    if "home_edge" in df:
        out["f_home_edge"] = df["home_edge"]
    if {"margin", "poss"} <= set(df):
        out["f_pace_scale"] = df["margin"] * (df["poss"] / 68.0 - 1.0)
        out["f_margin_scale"] = df["margin"]
    if {"margin", "conf_game"} <= set(df):
        out["f_conf_compress"] = df["margin"] * df["conf_game"].astype(float)
    return out
