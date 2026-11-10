"""
The backtest report: the model as it runs, replayed, graded, written to
research/RESULTS.md. Every number quoted anywhere in this repo comes from here,
so the docs cannot drift from the code (the NFL README did).

    .venv/bin/python scripts/backtest.py              # tune seasons only
    .venv/bin/python scripts/backtest.py --holdout    # + the hold-out, logged

Opening the hold-out is recorded in research/holdout_log.csv with the exact
parameters used, so "how many times was the hold-out looked at" has an answer.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from cbb import bets, evaluate, replay
from cbb.config import (HOLDOUT_SEASONS, RESEARCH, RULES, TUNE_SEASONS,
                        load_params, season_label)


def tiers(df: pd.DataFrame) -> pd.Series:
    p = set(RULES.power_conferences)
    hp, ap = df.home_conf.isin(p), df.away_conf.isin(p)
    return np.select([hp & ap, hp | ap], ["both power", "one power"], "neither power")


def bet_summary(b: pd.DataFrame) -> dict:
    d = b[b.outcome.isin(["W", "L", "P"])]
    if not len(d):
        return {"bets": 0}
    roi = d.profit.mean()
    se = d.profit.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    dec = d[d.outcome != "P"]
    return {"bets": len(d), "won": int((d.outcome == "W").sum()), "lost": int((d.outcome == "L").sum()),
            "push": int((d.outcome == "P").sum()), "roi": roi, "roi_se": se,
            "said": float(dec.p_bet.mean()), "won_rate": float((dec.outcome == "W").mean()),
            "units": float(d.profit.sum())}


def run_bets(pred: pd.DataFrame, params, fanduel_only: bool) -> dict[str, pd.DataFrame]:
    if fanduel_only:                 # he bets FanDuel: never grade a bet at another book's number
        pred = pred[pred.close_book == "fanduel"]
    cand = bets.candidates(pred, params)
    q = bets.qualifying(cand)
    res = pred.set_index("game_id").result
    return {"every qualifying bet": bets.grade(q, res),
            "the card (8 + 8 a week, best chance first)": bets.grade(bets.card(q), res),
            "the card, best expected value first": bets.grade(bets.card(q, rank_by="ev"), res)}


def bet_row(label: str, b: pd.DataFrame) -> str | None:
    s = bet_summary(b)
    if not s["bets"]:
        return None
    return (f"| {label}: {s['bets']} | {s['won']}-{s['lost']}-{s['push']} | {s['roi']:+.1%} | "
            f"{s['roi_se']:.1%} | {s['units']:+.1f} | {s['said']:.1%} | {s['won_rate']:.1%} |")


def bet_kind(b: pd.DataFrame) -> pd.Series:
    fav = np.where(b.market == "spread", b.line < 0, b.price < 0)
    return b.market + np.where(fav, ", favorite", ", underdog")


def windows(card: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    """How full the 8-and-8 windows were, and how many came out ahead."""
    days = pd.Series(pd.to_datetime(pred.date.unique()))
    allw = bets.window_of(days).unique()
    n = card.groupby("window").size().reindex(allw, fill_value=0)
    kind = pd.Series(n.index.str.split(" ").str[1], index=n.index)
    out = ["| window | windows | bets per window | windows with no bet | windows up | windows down |",
           "|---|---|---|---|---|---|"]
    units = card.groupby("window").profit.sum().reindex(allw, fill_value=0.0)
    for k in ("weekday", "weekend"):
        m = kind == k
        out.append(f"| {k} | {int(m.sum())} | {n[m].mean():.1f} | {(n[m] == 0).mean():.0%} | "
                   f"{int((units[m] > 0).sum())} | {int((units[m] < 0).sum())} |")
    return out


def section(name: str, pred: pd.DataFrame, params, fanduel_only: bool = False) -> list[str]:
    card = evaluate.scorecard(pred)
    sets = run_bets(pred, params, fanduel_only)
    out = [f"## {name}", "", "```", evaluate.fmt(card), "```", "",
           "| bets | record | ROI per $1 | SE | units | model said | won |", "|---|---|---|---|---|---|---|"]
    out += [r for label, b in sets.items() if (r := bet_row(label, b))]
    crd = sets["the card (8 + 8 a week, best chance first)"]
    if len(crd):
        out += ["", "The card by kind of bet:", "",
                "| bets | record | ROI per $1 | SE | units | model said | won |", "|---|---|---|---|---|---|---|"]
        out += [r for k, x in crd.groupby(bet_kind(crd)) if (r := bet_row(k, x))]
        out += ["", *windows(crd, pred)]
    out.append("")
    by = []
    for t, x in pred.assign(tier=tiers(pred)).groupby("tier"):
        c = evaluate.scorecard(x)
        if "su_lined" in c:
            by.append(f"| {t} | {c['games_lined']} | {c['su_lined']:.1%} | {c['su_favorite']:.1%} | "
                      f"{c['mae_model']:.2f} | {c['mae_line']:.2f} | {c['beyond_line']:+.2f} ({c['beyond_line_t']:+.1f}) |")
    if by:
        out += ["| games | n | model SU | favorite SU | model error | line error | beyond line (t) |",
                "|---|---|---|---|---|---|---|", *by, ""]
    for s, x in pred.groupby("season"):
        c = evaluate.scorecard(x)
        if "su_lined" in c:
            out.append(f"- {season_label(s)}: SU {c['su_lined']:.1%} vs favorite {c['su_favorite']:.1%}; "
                       f"error {c['mae_model']:.2f} vs line {c['mae_line']:.2f}; beyond line {c['beyond_line']:+.2f}")
    cal = evaluate.calibration(pred[pred.close_home_spread.notna()])
    out += ["", "Calibration (favorite's chance said vs won):", "", "| said | games | won | gap in SE |", "|---|---|---|---|"]
    out += [f"| {r.said:.1%} | {r.games} | {r.won:.1%} | {r.gap_in_se:+.1f} |" for r in cal.itertuples()]
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    params = load_params()
    seasons = TUNE_SEASONS + (HOLDOUT_SEASONS if args.holdout else ())
    pred = replay.run(seasons, params)
    pred = pred[pred.close_home_spread.notna() | pred.close_home_ml.notna()]

    out = ["# Backtest results", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/backtest.py` from `data/params.json`. "
           "Every game is predicted with only what was known that morning (`replay.py`, the same "
           "`Model` the daily run uses). Bets are priced at the closing line of the book in the "
           "data (consensus before 2023-24, FanDuel after), $1 each.", ""]
    out += section(f"Tune seasons {season_label(TUNE_SEASONS[0])} to {season_label(TUNE_SEASONS[-1])} (fit here)",
                   pred[pred.season.isin(TUNE_SEASONS)], params)
    if args.holdout:
        out += section(f"HOLD-OUT {season_label(HOLDOUT_SEASONS[0])} to {season_label(HOLDOUT_SEASONS[-1])} "
                       "(never fit on; bets only at FanDuel's own closing number)",
                       pred[pred.season.isin(HOLDOUT_SEASONS)], params, fanduel_only=True)
        evaluate.log_holdout("backtest.py", params)
    (RESEARCH / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
