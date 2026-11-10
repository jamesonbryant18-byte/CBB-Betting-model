"""
How good is a set of predictions? One scorecard, used by every test and report.

Every comparison is paired (same games, model vs market) and comes with a
t-statistic, because a difference without its noise is not a finding.

  straight up   model's pick vs the closing favorite (pick'ems left out)
  margin error  mean absolute error, model vs the closing spread
  win %         Brier score, model vs FanDuel's de-vigged moneyline
  beyond line   regress (result - line) on (model - line): the slope is how
                much of the model's disagreement with the line is real.
                0 = the line already knew everything; 1 = trust the model.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import pricing
from .config import PARAMS_FILE, RESEARCH


def _t(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(se), float(m / se) if se > 0 else np.nan


def market_margin(df: pd.DataFrame) -> pd.Series:
    return -df.close_home_spread


def market_p_home(df: pd.DataFrame) -> pd.Series:
    ok = df.close_home_ml.notna() & df.close_away_ml.notna()
    p = pd.Series(np.nan, index=df.index)
    ph, _ = pricing.devig(df.close_home_ml[ok], df.close_away_ml[ok])
    p[ok] = ph
    return p


def scorecard(df: pd.DataFrame, margin_col: str = "margin", p_col: str = "p_home") -> dict:
    d = df[df.result.notna() & df[margin_col].notna()]
    out = {"games": len(d)}

    # straight up, all games
    su = np.sign(d[margin_col]) == np.sign(d.result)
    out["su"] = float(su.mean())
    out["mae"] = float((d[margin_col] - d.result).abs().mean())

    if "close_home_spread" not in d:
        return out
    lined = d[d.close_home_spread.notna()]
    out["games_lined"] = len(lined)
    if len(lined) < 30:
        return out
    mm = market_margin(lined)
    nz = mm != 0
    model_right = (np.sign(lined[margin_col]) == np.sign(lined.result))[nz]
    fav_right = (np.sign(mm) == np.sign(lined.result))[nz]
    out["su_lined"] = float(model_right.mean())
    out["su_favorite"] = float(fav_right.mean())
    out["su_diff"], out["su_diff_se"], out["su_diff_t"] = _t(model_right.astype(float) - fav_right.astype(float))
    disagree = (np.sign(lined[margin_col]) != np.sign(mm)) & nz
    out["disagree_n"] = int(disagree.sum())
    out["disagree_won"] = float((np.sign(lined[margin_col]) == np.sign(lined.result))[disagree].mean()) if disagree.any() else np.nan

    err_m = (lined[margin_col] - lined.result).abs()
    err_l = (mm - lined.result).abs()
    out["mae_model"], out["mae_line"] = float(err_m.mean()), float(err_l.mean())
    out["mae_diff"], out["mae_diff_se"], out["mae_diff_t"] = _t(err_m - err_l)

    # how much of the disagreement with the line is real
    gap = (lined[margin_col] - mm).to_numpy()
    miss = (lined.result - mm).to_numpy()
    ok = np.isfinite(gap) & np.isfinite(miss)
    gap, miss = gap[ok], miss[ok]
    slope = float((gap @ miss) / (gap @ gap)) if (gap @ gap) > 0 else np.nan
    resid = miss - slope * gap
    se = float(np.sqrt((resid @ resid) / (len(gap) - 1) / (gap @ gap))) if len(gap) > 2 else np.nan
    out["beyond_line"], out["beyond_line_se"] = slope, se
    out["beyond_line_t"] = slope / se if se and se > 0 else np.nan

    # win %: Brier vs the de-vigged moneyline
    mp = market_p_home(lined)
    both = mp.notna() & lined[p_col].notna()
    if both.sum() >= 30:
        won = (lined.result > 0).astype(float)[both]
        b_m = (lined[p_col][both] - won) ** 2
        b_k = (mp[both] - won) ** 2
        out["brier_model"], out["brier_market"] = float(b_m.mean()), float(b_k.mean())
        out["brier_diff"], out["brier_diff_se"], out["brier_diff_t"] = _t(b_m - b_k)
        out["games_ml"] = int(both.sum())
    return out


def calibration(df: pd.DataFrame, p_col: str = "p_home", bins=None) -> pd.DataFrame:
    """Said vs won, by the favorite's chance, with the binomial standard error."""
    d = df[df.result.notna() & df[p_col].notna()]
    p_fav = np.maximum(d[p_col], 1 - d[p_col])
    fav_won = np.where(d[p_col] >= 0.5, d.result > 0, d.result < 0)
    bins = bins or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    b = pd.cut(p_fav, bins, include_lowest=True)
    t = pd.DataFrame({"said": p_fav, "won": fav_won.astype(float), "bin": b}).groupby("bin", observed=True)
    out = t.agg(games=("won", "size"), said=("said", "mean"), won=("won", "mean"))
    out["se"] = np.sqrt(out.said * (1 - out.said) / out.games)
    out["gap_in_se"] = (out.won - out.said) / out.se
    return out


def fmt(card: dict) -> str:
    """One readable block."""
    def g(k, f="{:.3f}"):
        v = card.get(k)
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)
    lines = [f"games {card.get('games')}, with a line {card.get('games_lined')}"]
    if "su_lined" in card:
        lines += [
            f"straight up   model {g('su_lined', '{:.1%}')} vs favorite {g('su_favorite', '{:.1%}')}"
            f"  (diff {g('su_diff', '{:+.2%}')}, t {g('su_diff_t', '{:+.1f}')});"
            f" picks against the favorite won {g('disagree_won', '{:.1%}')} of {card.get('disagree_n')}",
            f"margin error  model {g('mae_model', '{:.2f}')} vs line {g('mae_line', '{:.2f}')}"
            f"  (diff {g('mae_diff', '{:+.3f}')}, t {g('mae_diff_t', '{:+.1f}')})",
            f"beyond line   {g('beyond_line', '{:+.3f}')} (se {g('beyond_line_se', '{:.3f}')}, t {g('beyond_line_t', '{:+.1f}')})"
            "  [0 = no information the line lacks]",
        ]
    if "brier_model" in card:
        lines.append(f"win % (Brier) model {g('brier_model', '{:.4f}')} vs market {g('brier_market', '{:.4f}')}"
                     f"  (diff {g('brier_diff', '{:+.4f}')}, t {g('brier_diff_t', '{:+.1f}')}, n {card.get('games_ml')})")
    return "\n".join(lines)


def log_holdout(source: str, params) -> None:
    """Record that the hold-out seasons were looked at, by what, with which parameters."""
    log = RESEARCH / "holdout_log.csv"
    digest = hashlib.sha1(PARAMS_FILE.read_bytes()).hexdigest()[:10] if PARAMS_FILE.exists() else "defaults"
    row = pd.DataFrame([{"opened": f"{datetime.now():%Y-%m-%d %H:%M}", "by": source, "params_sha": digest,
                         "params": json.dumps({k: getattr(params, k) for k in params.FITTED})}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
