"""
Odds math, and turning a projected margin into win and cover chances.

Sign convention (see odds.py): `home_spread` is the number on a home ticket.
The home side covers when  margin + home_spread > 0.

Margins are modeled as a normal distribution around the projection, put on
whole points. Basketball has no ties, so the mass at exactly 0 (a tie after 40
minutes, i.e. overtime) is split evenly between winning and losing by 1. On a
half-point line this gives the same cover chance as the plain normal curve; on a
whole number it also gives the push chance.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_GRID = np.arange(-90, 91)


def american_to_prob(odds):
    """Implied probability of American odds, vig included."""
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def prob_to_american(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.where(p > 0.5, -100.0 * p / (1 - p), 100.0 * (1 - p) / p)


def profit_per_dollar(odds):
    """What $1 wins at these odds."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / -o)


def devig(odds_a, odds_b):
    """
    Fair probabilities from ONE book's two-sided price (multiplicative).

    Both prices must come from the same book at the same time. Taking the best
    price on each side from different books shrinks the book's cut and makes
    every edge look bigger than it is (a real NFL bug).
    """
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    return pa / (pa + pb), pb / (pa + pb)


def expected_value(p_win, odds, p_push=0.0):
    """Profit per $1 staked, on average."""
    return p_win * profit_per_dollar(odds) - (1 - p_win - p_push)


# ─────────────────────────── margin distribution ───────────────────────────

def margin_pmf(mu, sigma):
    """P(final margin = k) for k on _GRID, one row per game. No ties."""
    mu = np.atleast_1d(np.asarray(mu, dtype=float))[:, None]
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))[:, None]
    hi = norm.cdf((_GRID + 0.5 - mu) / sigma)
    lo = norm.cdf((_GRID - 0.5 - mu) / sigma)
    pmf = hi - lo
    zero = int(np.where(_GRID == 0)[0][0])
    tie = pmf[:, zero].copy()
    pmf[:, zero] = 0.0
    pmf[:, zero + 1] += tie / 2
    pmf[:, zero - 1] += tie / 2
    return pmf / pmf.sum(axis=1, keepdims=True)


def win_prob(mu, sigma):
    """P(home wins)."""
    pmf = margin_pmf(mu, sigma)
    return pmf[:, _GRID > 0].sum(axis=1)


def cover_prob(mu, sigma, home_spread):
    """(P(home covers), P(push)) for a home ticket at `home_spread`."""
    pmf = margin_pmf(mu, sigma)
    need = -np.atleast_1d(np.asarray(home_spread, dtype=float))[:, None]
    cover = (pmf * (_GRID[None, :] > need)).sum(axis=1)
    push = (pmf * (_GRID[None, :] == need)).sum(axis=1)
    return cover, push
