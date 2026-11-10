"""
Which games to bet, at what number, and the weekly card.

THE BET PROBABILITY. The NFL model's bets said 58% and won 54%: picking the games
where a model disagrees most with the market also picks the games where the
model is most wrong. So a bet's chance is the market's chance moved only part of
the way toward the model:

    p_bet = p_market + blend_weight * (p_model - p_model_at_the_market_line)

`blend_weight` is fitted (scripts/tune.py --stage blend) as the share of the
model's disagreement with the closing line that actually showed up in results.
At 0 the model never bets; at 1 it trusts itself completely.

  spread     p_market = FanDuel's two spread prices, de-vigged (about 50%)
  moneyline  p_market = FanDuel's two moneylines, de-vigged

A bet needs expected value >= RULES.min_ev at FanDuel's price, on a bettable
game (replay.bettable), at a half-point spread, with a moneyline inside
-350..+250. One bet per game: the spread if it qualifies, else the moneyline.

THE CARD. Monday-Friday share RULES.weekday_max slots, Saturday-Sunday share
RULES.weekend_max. First come, first served: each day's qualifying bets are
ranked by p_bet and take the slots still open in that window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pricing
from .config import RULES, BetRules, ModelParams

BET_COLS = ["game_id", "date", "home", "away", "market", "side", "team", "line", "price",
            "p_model", "p_market", "p_bet", "ev", "margin", "market_margin"]


def _is_half(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.isfinite(x) & (np.abs(np.round(x * 2) % 2) == 1)


def candidates(df: pd.DataFrame, params: ModelParams, rules: BetRules = RULES,
               prefix: str = "close_") -> pd.DataFrame:
    """
    Every side of every bettable game priced at `prefix` lines, with p_bet and EV.

    `df` = model predictions (margin, sigma, p_home) joined to one book's lines
    (prefix + home_spread, home_spread_price, away_spread_price, home_ml, away_ml).
    """
    d = df[df.bettable].copy()
    hs = d[f"{prefix}home_spread"].to_numpy(float)
    w = params.blend_weight
    mm = -hs                                                    # the line's margin
    rows = []

    # spreads
    ok = np.isfinite(hs) & d[f"{prefix}home_spread_price"].notna().to_numpy() & d[f"{prefix}away_spread_price"].notna().to_numpy()
    if rules.half_points_only:
        ok &= _is_half(hs)
    s = d[ok]
    if len(s):
        hs_s = s[f"{prefix}home_spread"].to_numpy(float)
        cover_model, _ = pricing.cover_prob(s.margin, s.sigma, hs_s)
        cover_line, _ = pricing.cover_prob(-hs_s, s.sigma, hs_s)
        pm_h, pm_a = pricing.devig(s[f"{prefix}home_spread_price"], s[f"{prefix}away_spread_price"])
        for side, sign, price, p_mod, p_line, p_mkt in (
                ("home", 1, s[f"{prefix}home_spread_price"], cover_model, cover_line, pm_h),
                ("away", -1, s[f"{prefix}away_spread_price"], 1 - cover_model, 1 - cover_line, pm_a)):
            p_bet = np.clip(p_mkt + w * (p_mod - p_line), 0.001, 0.999)
            rows.append(pd.DataFrame({
                "game_id": s.game_id, "date": s.date, "home": s.home, "away": s.away,
                "market": "spread", "side": side, "team": s[side],
                "line": sign * hs_s, "price": price.to_numpy(float), "p_model": p_mod,
                "p_market": p_mkt, "p_bet": p_bet,
                "ev": pricing.expected_value(p_bet, price.to_numpy(float)),
                "margin": s.margin, "market_margin": -hs_s}))

    # moneylines
    ok = d[f"{prefix}home_ml"].notna().to_numpy() & d[f"{prefix}away_ml"].notna().to_numpy()
    m = d[ok]
    if len(m):
        hs_m = m[f"{prefix}home_spread"].to_numpy(float)
        p_mod = pricing.win_prob(m.margin, m.sigma)
        p_line = np.where(np.isfinite(hs_m), pricing.win_prob(np.nan_to_num(-hs_m), m.sigma), np.nan)
        pm_h, pm_a = pricing.devig(m[f"{prefix}home_ml"], m[f"{prefix}away_ml"])
        # with no spread posted, measure the model against the moneyline's own view
        p_line = np.where(np.isfinite(p_line), p_line, pm_h)
        for side, price, pmod, pline, pmkt in (
                ("home", m[f"{prefix}home_ml"], p_mod, p_line, pm_h),
                ("away", m[f"{prefix}away_ml"], 1 - p_mod, 1 - p_line, pm_a)):
            pr = price.to_numpy(float)
            p_bet = np.clip(pmkt + w * (pmod - pline), 0.001, 0.999)
            rows.append(pd.DataFrame({
                "game_id": m.game_id, "date": m.date, "home": m.home, "away": m.away,
                "market": "moneyline", "side": side, "team": m[side], "line": np.nan,
                "price": pr, "p_model": pmod, "p_market": pmkt, "p_bet": p_bet,
                "ev": pricing.expected_value(p_bet, pr),
                "margin": m.margin, "market_margin": -hs_m}))
    if not rows:
        return pd.DataFrame(columns=BET_COLS)
    c = pd.concat(rows, ignore_index=True)
    in_range = (c.market == "spread") | ((c.price <= rules.ml_max_underdog) & (c.price >= rules.ml_max_favorite))
    return c[in_range].reset_index(drop=True)


def qualifying(c: pd.DataFrame, rules: BetRules = RULES) -> pd.DataFrame:
    """Bets with enough value; one per game (spread first)."""
    q = c[c.ev >= rules.min_ev].copy()
    q["_order"] = np.where(q.market == "spread", 0, 1)
    q = q.sort_values(["game_id", "_order", "ev"], ascending=[True, True, False])
    return q.drop_duplicates("game_id").drop(columns="_order").reset_index(drop=True)


def window_of(date: pd.Series) -> pd.Series:
    """'2026-11-16 weekday' / '2026-11-21 weekend' (the window's first day)."""
    d = pd.to_datetime(date)
    wd = d.dt.weekday
    start = np.where(wd >= 5, d - pd.to_timedelta(wd - 5, unit="D"), d - pd.to_timedelta(wd, unit="D"))
    kind = np.where(wd >= 5, "weekend", "weekday")
    return pd.Series(pd.to_datetime(start).strftime("%Y-%m-%d"), index=date.index) + " " + kind


def card(q: pd.DataFrame, rules: BetRules = RULES, taken: pd.DataFrame | None = None,
         rank_by: str = "p_bet") -> pd.DataFrame:
    """
    Apply the 8-and-8 limits, first come first served.

    `taken` = bets already published earlier in the window (they keep their
    slots). Within a day, bets are ranked by `rank_by` (the chance the bet wins).
    """
    if not len(q):
        return q.assign(window=pd.Series(dtype=str))
    q = q.copy()
    q["window"] = window_of(q.date)
    used = {} if taken is None or not len(taken) else window_of(taken.date).value_counts().to_dict()
    keep = []
    for (win, day), x in q.sort_values("date").groupby(["window", "date"], sort=True):
        cap = rules.weekend_max if win.endswith("weekend") else rules.weekday_max
        room = cap - used.get(win, 0)
        if room <= 0:
            continue
        pick = x.sort_values(rank_by, ascending=False).head(room)
        used[win] = used.get(win, 0) + len(pick)
        keep.append(pick)
    return pd.concat(keep, ignore_index=True) if keep else q.iloc[:0]


def grade(b: pd.DataFrame, results: pd.Series) -> pd.DataFrame:
    """Win/loss/push and profit per $1 for bets, given home margins by game_id."""
    b = b.copy()
    r = b.game_id.map(results).to_numpy(float)
    side = np.where(b.side == "home", 1.0, -1.0)
    spread_score = np.where(b.market == "spread", side * r + b.line.fillna(0).to_numpy(), np.nan)
    ml_score = side * r
    score = np.where(b.market == "spread", spread_score, ml_score)
    b["outcome"] = np.select([score > 0, score < 0], ["W", "L"], "P")
    b.loc[~np.isfinite(r), "outcome"] = None
    b["profit"] = np.select([b.outcome == "W", b.outcome == "L"],
                            [pricing.profit_per_dollar(b.price), -1.0], 0.0)
    return b
