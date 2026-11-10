"""
The model: one prediction path, used by the backtest AND the daily run.

    model = Model(games, players, lines, params)
    state = model.rate(season, asof)          # everything known before `asof`
    preds = model.predict(target_games, state)

`rate` only ever sees games dated before `asof` and lines of those games, so a
replay of a past date and a live run on today's date are the same computation.
tests/test_model.py proves it: deleting every game on or after `asof` from the
inputs leaves the prediction unchanged to the last decimal.

How a rating is built, for a season, as of a date:
  1. Preseason prior: last season's final rating, pulled toward the average of
     the team's current conference. How much is kept depends on how much of last
     season's production is back (players.py continuity).
  2. Market ratings: the ratings that best explain the closing spreads of this
     season's games so far, pulled toward the preseason prior.
  3. Performance ratings: the ratings that best explain this season's results
     (each past margin adjusted for who sat), pulled toward a blend of 1 and 2.
  4. Tempo ratings (possessions per 40 minutes), for the spread of outcomes.

A projected margin is  rating_home - rating_away + home court (if not neutral)
minus the value of players expected out, times avail_points.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import players as pl
from . import pricing, ratings
from .config import ModelParams

NEW_TEAM_GAP = 6.0        # a team new to D1 starts this far below its conference
TEMPO_CARRY = 0.7         # share of last season's pace kept
TEMPO_PENALTY = 4.0
AVG_POSS = 68.0


@dataclass
class State:
    season: int
    asof: pd.Timestamp
    team_ids: np.ndarray
    prior: np.ndarray
    rating: ratings.Fit
    market: ratings.Fit | None
    tempo: ratings.Fit
    n_games: int
    n_lines: int


class Model:
    def __init__(self, games: pd.DataFrame, players: pd.DataFrame,
                 lines: pd.DataFrame | None, params: ModelParams):
        self.p = params
        g = games.copy()
        g["margin"] = g.home_score - g.away_score
        self.games = g
        self.lines = (lines.set_index("game_id").close_home_spread
                      if lines is not None and len(lines) else pd.Series(dtype=float))
        self._players = players
        self._avail = {}
        self._end = {}
        self._season_games = {}

    def with_params(self, params: ModelParams) -> "Model":
        """Same data, different parameters; shares the (parameter-free) availability cache."""
        m = Model.__new__(Model)
        m.p, m.games, m.lines, m._players = params, self.games, self.lines, self._players
        m._avail, m._end, m._season_games = self._avail, {}, self._season_games
        return m

    # ─────────── inputs per season ───────────

    def season_games(self, season: int) -> pd.DataFrame:
        if season not in self._season_games:
            self._season_games[season] = self.games[self.games.season == season]
        return self._season_games[season]

    def availability(self, season: int) -> pd.DataFrame:
        """Per game: home/away miss_actual, miss_proxy, continuity (cached)."""
        key = (season, self.p.value_production_weight)
        if key not in self._avail:
            w = self.p.value_production_weight
            g = self.season_games(season)
            p = self._players[self._players.season == season]
            prev = self._players[self._players.season == season - 1]
            last = pl.season_values(prev, w) if len(prev) else None
            a = pl.availability(g, p, last, w)
            self._avail[key] = pl.by_game(a, g).set_index("game_id")
        return self._avail[key]

    @lru_cache(maxsize=None)
    def teams(self, season: int) -> np.ndarray:
        g = self.season_games(season)
        return np.unique(np.concatenate([g.home_id.to_numpy(), g.away_id.to_numpy()]))

    @lru_cache(maxsize=None)
    def conference(self, season: int) -> pd.Series:
        """Each team's conference this season (from its schedule, known in advance)."""
        g = self.season_games(season)
        c = pd.concat([pd.DataFrame({"t": g.home_id, "c": g.home_conf}),
                       pd.DataFrame({"t": g.away_id, "c": g.away_conf})]).dropna()
        return c.groupby("t").c.agg(lambda x: x.mode().iloc[0])

    # ─────────── the preseason prior ───────────

    def end_of_season(self, season: int) -> tuple[pd.Series, pd.Series]:
        """(rating, tempo) the day after the season's last game."""
        if season not in self._end:
            g = self.season_games(season)
            if len(g) == 0 or g.status.eq("final").sum() == 0:
                self._end[season] = (pd.Series(dtype=float), pd.Series(dtype=float))
            else:
                last = g[g.status == "final"].date.max() + pd.Timedelta(days=1)
                s = self.rate(season, last)
                self._end[season] = (s.rating.as_series(), s.tempo.as_series())
        return self._end[season]

    def preseason_prior(self, season: int, asof: pd.Timestamp, team_ids) -> tuple[np.ndarray, np.ndarray]:
        last_r, last_t = (self.end_of_season(season - 1) if season - 1 in set(self.games.season)
                          else (pd.Series(dtype=float), pd.Series(dtype=float)))
        team_ids = np.asarray(team_ids)
        if not len(last_r):                       # first season loaded: no history
            return np.zeros(len(team_ids)), np.zeros(len(team_ids))
        prev = last_r.reindex(team_ids)
        conf = self.conference(season).reindex(team_ids).to_numpy()
        conf_mean = prev.groupby(conf).transform("mean").fillna(prev.mean())
        prev = prev.fillna(conf_mean - NEW_TEAM_GAP)
        cont = self._continuity(season, asof, team_ids)
        carry = np.clip(self.p.carry_base + self.p.carry_continuity * cont, 0.0, 1.0)
        prior = conf_mean.to_numpy() + carry * (prev.to_numpy() - conf_mean.to_numpy())
        tempo_prior = TEMPO_CARRY * last_t.reindex(team_ids).fillna(0.0).to_numpy()
        return prior, tempo_prior

    @lru_cache(maxsize=None)
    def _continuity_rows(self, season: int) -> pd.DataFrame:
        a = self.availability(season)
        g = self.season_games(season)[["game_id", "date", "home_id", "away_id"]].set_index("game_id")
        rows = pd.concat([
            pd.DataFrame({"t": g.home_id, "d": g.date, "c": a.home_continuity.reindex(g.index)}),
            pd.DataFrame({"t": g.away_id, "d": g.date, "c": a.away_continuity.reindex(g.index)})])
        return rows.dropna().sort_values("d")

    def _continuity(self, season: int, asof: pd.Timestamp, team_ids) -> np.ndarray:
        """Each team's returning-production share as of its latest game on or before `asof`.

        A row's continuity counts only players listed in box scores BEFORE that
        game, so reading the row of a game on `asof` itself is not a peek.
        """
        rows = self._continuity_rows(season)
        latest = rows[rows.d <= asof].groupby("t").c.last()
        return latest.reindex(team_ids).fillna(pl.CONTINUITY_DEFAULT).to_numpy()

    # ─────────── rating as of a date ───────────

    def rate(self, season: int, asof: pd.Timestamp) -> State:
        p = self.p
        asof = pd.Timestamp(asof).normalize()
        team_ids = self.teams(season)
        prior, tempo_prior = self.preseason_prior(season, asof, team_ids)
        g = self.season_games(season)
        past = g[(g.date < asof) & (g.status == "final") & g.margin.notna()]
        w = ratings.recency_weights(past.date, asof, p.half_life_days)

        # market ratings from closing spreads of earlier games
        market = None
        n_lines = 0
        if p.market_lambda > 0 and len(self.lines):
            line = self.lines.reindex(past.game_id).to_numpy()
            has = np.isfinite(line)
            n_lines = int(has.sum())
            if n_lines:
                market = ratings.fit(past[has], -line[has], team_ids, prior,
                                     p.market_ridge_lambda, p.hfa, p.hfa_lambda, w[has])

        # performance ratings from availability-adjusted margins
        a = self.availability(season).reindex(past.game_id)
        adj = (p.avail_points * (a.home_miss_actual.fillna(0) - a.away_miss_actual.fillna(0))).to_numpy()
        y = past.margin.to_numpy() + adj
        if market is not None:
            lam = p.ridge_lambda + p.market_lambda
            mean = (p.ridge_lambda * prior + p.market_lambda * market.values) / lam
        else:
            lam, mean = p.ridge_lambda, prior
        rating = ratings.fit(past, y, team_ids, mean, lam, p.hfa, p.hfa_lambda, w)

        boxed = past[past.has_box]
        t_mean = float(boxed.poss40.mean()) if len(boxed) else AVG_POSS
        tempo = ratings.fit(boxed, boxed.poss40.to_numpy() - t_mean, team_ids, tempo_prior,
                            TEMPO_PENALTY, 0.0, 1e6, w[past.has_box.to_numpy()], kind="tempo")
        tempo.extra = t_mean
        return State(season, asof, team_ids, prior, rating, market, tempo, len(past), n_lines)

    # ─────────── predict ───────────

    def predict(self, target: pd.DataFrame, state: State,
                missing: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Projections for `target` games (home_id, away_id, neutral, game_id).

        `missing`: per game home_miss / away_miss to use instead of the proxy
        (the live run passes the proxy plus his injury notes; None = proxy).
        """
        p = self.p
        out = target[["game_id", "season", "date", "home_id", "away_id", "neutral"]].copy()
        out["r_home"] = state.rating.lookup(out.home_id, np.nan)
        out["r_away"] = state.rating.lookup(out.away_id, np.nan)
        out["hfa"] = np.where(out.neutral, 0.0, state.rating.extra)
        if missing is None:
            a = self.availability(state.season).reindex(out.game_id)
            out["home_miss"] = a.home_miss_proxy.fillna(0).to_numpy()
            out["away_miss"] = a.away_miss_proxy.fillna(0).to_numpy()
            out["home_miss_fresh"] = a.home_miss_fresh.fillna(0).to_numpy()
            out["away_miss_fresh"] = a.away_miss_fresh.fillna(0).to_numpy()
        else:
            m = missing.set_index("game_id").reindex(out.game_id)
            out["home_miss"] = m.home_miss.fillna(0).to_numpy()
            out["away_miss"] = m.away_miss.fillna(0).to_numpy()
        out["avail_adj"] = -p.avail_points * (out.home_miss - out.away_miss)
        out["margin"] = out.r_home - out.r_away + out.hfa + out.avail_adj
        if state.market is not None:
            out["market_rating_margin"] = (state.market.lookup(out.home_id, np.nan)
                                           - state.market.lookup(out.away_id, np.nan)
                                           + np.where(out.neutral, 0.0, state.market.extra))
        out["poss"] = (state.tempo.extra + state.tempo.lookup(out.home_id, 0.0)
                       + state.tempo.lookup(out.away_id, 0.0))
        out["sigma"] = p.sigma + p.sigma_per_poss * (out.poss - AVG_POSS)
        out["p_home"] = pricing.win_prob(out.margin.to_numpy(), out.sigma.to_numpy())
        out["n_games"] = state.n_games
        out["n_lines"] = state.n_lines
        return out
