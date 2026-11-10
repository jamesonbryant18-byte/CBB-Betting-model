"""
Team power ratings in points: opponent-adjusted, recent games weighted more.

One solver does every fit in this model: a weighted ridge regression where each
team's rating is pulled toward a prior mean instead of toward zero.

    minimize   sum_g w_g (y_g - x_g . beta)^2  +  sum_j penalty_j (beta_j - prior_j)^2

`penalty` is measured in games: a penalty of 3 means the prior counts as much as
three full-weight games of evidence about that team.

Three uses:
  margin   y = home margin (availability-adjusted), x = +home, -away, +HFA
  market   y = the margin the closing spread implied, same design
  tempo    y = possessions per 40 minutes, x = +home, +away, +intercept
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Fit:
    team_ids: np.ndarray
    values: np.ndarray        # rating per team, aligned with team_ids
    extra: float              # HFA (margin, market) or league mean (tempo)

    def lookup(self, team_ids, default: float = 0.0) -> np.ndarray:
        idx = pd.Series(np.arange(len(self.team_ids)), index=self.team_ids)
        pos = idx.reindex(np.asarray(team_ids)).to_numpy()
        out = np.full(len(pos), default, dtype=float)
        ok = ~np.isnan(pos)
        out[ok] = self.values[pos[ok].astype(int)]
        return out

    def as_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.team_ids)


def recency_weights(dates: pd.Series, asof: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (asof - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.power(0.5, np.clip(age, 0, None) / half_life_days)


def solve(home_idx, away_idx, extra_col, y, w, n_teams, penalty, prior,
          away_sign: float = -1.0) -> np.ndarray:
    """Weighted ridge toward `prior`. Returns n_teams + 1 coefficients."""
    m = len(y)
    rows = np.concatenate([np.arange(m), np.arange(m), np.arange(m)])
    cols = np.concatenate([home_idx, away_idx, np.full(m, n_teams)])
    vals = np.concatenate([np.ones(m), np.full(m, away_sign), extra_col])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n_teams + 1))
    XtW = X.T.multiply(w).tocsr()
    A = (XtW @ X).toarray() + np.diag(penalty)
    b = XtW @ y + penalty * prior
    return np.linalg.solve(A, b)


def fit(games: pd.DataFrame, y: np.ndarray, team_ids: np.ndarray, prior: np.ndarray,
        penalty: float, extra_prior: float, extra_penalty: float, w: np.ndarray,
        kind: str = "margin") -> Fit:
    """
    Fit ratings for `team_ids` from `games` (needs home_id, away_id, neutral).

    kind="margin": y ~ r_home - r_away + HFA * (not neutral)
    kind="tempo":  y ~ t_home + t_away + league mean
    """
    team_ids = np.asarray(team_ids)
    n = len(team_ids)
    full_prior = np.append(np.asarray(prior, dtype=float), extra_prior)
    full_pen = np.append(np.full(n, float(penalty)), float(extra_penalty))
    if len(games) == 0:
        return Fit(team_ids, full_prior[:-1].copy(), float(extra_prior))
    idx = pd.Series(np.arange(n), index=team_ids)
    h = idx.reindex(games.home_id.to_numpy()).to_numpy()
    a = idx.reindex(games.away_id.to_numpy()).to_numpy()
    ok = ~(np.isnan(h) | np.isnan(a)) & np.isfinite(y)
    if kind == "margin":
        extra_col = (~games.neutral.to_numpy()).astype(float)[ok]
        sign = -1.0
    else:
        extra_col = np.ones(int(ok.sum()))
        sign = 1.0
    beta = solve(h[ok].astype(int), a[ok].astype(int), extra_col, np.asarray(y)[ok],
                 np.asarray(w)[ok], n, full_pen, full_prior, away_sign=sign)
    return Fit(team_ids, beta[:-1], float(beta[-1]))
