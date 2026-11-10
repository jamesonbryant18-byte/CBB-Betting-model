"""
Do advanced metrics make the model better? The pre-registered test.

    .venv/bin/python scripts/metric_tests.py             # screen + tune
    .venv/bin/python scripts/metric_tests.py --holdout   # + the hold-out, logged

Rules, written before any result was looked at:
  1. Replay the model as it runs (data/params.json) over SCREEN + TUNE seasons.
  2. For each metric, fit its coefficient on the model's misses in SCREEN
     seasons (2010-2019) only.
  3. Apply that fixed coefficient to TUNE seasons (2019-20 to 2022-23).
     PASS = margin error falls with t <= -2 there, AND a fit on TUNE alone has
     the same sign.
  4. Market check, TUNE seasons: does (result - closing line) move with the
     metric's adjustment? |t| >= 3 (about 0.05 after ~20 looks) = the market
     misprices it, which is the only kind of help that can make money.
  5. HOLDOUT seasons (2023-24 to 2025-26) only with --holdout: the same fixed
     coefficient, scored once on FanDuel closing lines, logged to
     research/holdout_log.csv. Nothing is refit there.

Writes research/metric_tests.md and adds rows to research/test_database.csv.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

from cbb import evaluate, features, replay
from cbb.config import CACHE, HOLDOUT_SEASONS, RESEARCH, SCREEN_SEASONS, TUNE_SEASONS, load_params

FF = features.FACTORS
TESTS = [
    # name, columns, what it asks
    ("four factors (all 10)", [f"f_{s}_{x}" for s in ("off", "def") for x in FF],
     "Opponent-adjusted shooting, turnovers, rebounding, free throws, 3-point rate, offense and defense"),
    *[(f"{x} (offense + defense)", [f"f_off_{x}", f"f_def_{x}"], f"Opponent-adjusted {x}") for x in FF],
    ("3-point defense luck", ["f_def3_luck"], "Opponents' 3s made above league average (KenPom: mostly luck)"),
    ("3-point offense luck", ["f_off3_luck"], "Own 3s made above league average"),
    ("free-throw defense luck", ["f_ft_luck"], "Opponents' free throws made above league average"),
    ("all shooting luck", ["f_def3_luck", "f_off3_luck", "f_ft_luck"], "The three luck terms together"),
    ("pace scaling", ["f_pace_scale"], "Scale the margin by expected possessions (per-possession ratings)"),
    ("margin stretch/shrink", ["f_margin_scale"], "Is the projected margin too big or too small?"),
    ("conference-game compression", ["f_conf_compress"], "Are conference games closer than projected?"),
    ("rest days", ["f_rest"], "Days since last game, home minus away (capped at 7)"),
    ("recent form", ["f_form"], "Beat-the-projection over the last 5 games"),
    ("team-specific home court", ["f_home_edge"], "This arena's beat-the-projection the 3 seasons before"),
    ("fresh absences (a key player just went out)", ["f_miss_fresh"],
     "Value of players out 3 games or fewer, home minus away: do new injuries need MORE than the "
     "model's flat weight?"),
]


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    return beta, beta / np.sqrt(np.diag(cov))


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def score(d: pd.DataFrame, cols: list, beta: np.ndarray) -> dict:
    """A fixed adjustment applied to seasons `d`: margin error, straight up, market."""
    adj = d[cols].to_numpy(float) @ beta
    new = d.margin + adj
    mae_d, mae_t = paired_t((new - d.result).abs(), (d.margin - d.result).abs())
    su_d, su_t = paired_t(np.sign(new) == np.sign(d.result), np.sign(d.margin) == np.sign(d.result))
    line_miss = (d.result + d.close_home_spread).to_numpy(float)       # result - line margin
    mk_b, mk_t = ols(adj[:, None], line_miss) if np.std(adj) > 0 else (np.array([0.0]), np.array([0.0]))
    return {"mae_change": mae_d, "mae_t": mae_t, "su_change": su_d, "su_t": su_t,
            "market_slope": float(mk_b[0]), "market_t": float(mk_t[0]), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    params = load_params()
    seasons = SCREEN_SEASONS + TUNE_SEASONS + (HOLDOUT_SEASONS if args.holdout else ())
    pred = replay.run(seasons, params)
    pred = features.add_residual_features(pred)
    f = pd.read_parquet(CACHE / "features.parquet")
    df = pred.merge(f, on="game_id", how="left")
    df = pd.concat([df, features.diffs(df)], axis=1)
    df["resid"] = df.result - df.margin
    scr = df[df.season.isin(SCREEN_SEASONS)]
    tun = df[df.season.isin(TUNE_SEASONS) & df.close_home_spread.notna()]
    hold = df[df.season.isin(HOLDOUT_SEASONS) & df.close_home_spread.notna()]

    base_s, base_t = evaluate.scorecard(scr), evaluate.scorecard(tun)
    print("BASE (the model as it runs)")
    print(f" screen {SCREEN_SEASONS[0]}-{SCREEN_SEASONS[-1]}: mae {base_s['mae']:.3f}, su {base_s['su']:.1%}")
    print(" tune:\n" + evaluate.fmt(base_t))

    rows = []
    for name, cols, what in TESTS:
        s = scr.dropna(subset=cols)
        t = tun.dropna(subset=cols)
        beta, t_in = ols(s[cols], s.resid)
        sc = score(t, cols, beta)
        beta_tune, _ = ols(t[cols], t.resid)
        same_sign = bool(np.all(np.sign(beta_tune) == np.sign(beta)))
        passed = (sc["mae_t"] <= -2) and same_sign
        verdict = "PASS" if passed else ("NO HELP" if sc["mae_change"] >= 0 else "TOO WEAK")
        row = {"test": name, "features": len(cols), "screen_beta": np.round(beta, 3).tolist(),
               "screen_t": np.round(t_in, 1).tolist(), **{f"tune_{k}": v for k, v in sc.items()},
               "same_sign": same_sign, "mispriced": abs(sc["market_t"]) >= 3, "verdict": verdict,
               "what": what, "n_screen": len(s)}
        if args.holdout:
            row.update({f"hold_{k}": v for k, v in score(hold.dropna(subset=cols), cols, beta).items()})
        rows.append(row)
        print(f"  {name:32s} mae {sc['mae_change']:+.4f} (t {sc['mae_t']:+.1f})  su {sc['su_change']:+.2%} "
              f"(t {sc['su_t']:+.1f})  sign {'same' if same_sign else 'FLIPS'}  market t {sc['market_t']:+.1f}"
              f"  -> {verdict}" + (f"  | HOLD-OUT mae {row['hold_mae_change']:+.4f} (t {row['hold_mae_t']:+.1f})"
                                   f" market t {row['hold_market_t']:+.1f}" if args.holdout else ""))

    res = pd.DataFrame(rows)
    base_h = evaluate.scorecard(hold) if args.holdout else None
    write_report(res, base_s, base_t, base_h)
    log_tests(res)
    if args.holdout:
        evaluate.log_holdout("metric_tests.py", params)


def write_report(res, base_s, base_t, base_h=None):
    lines = [
        "# Advanced metric tests",
        "",
        f"Run {date.today()} by `scripts/metric_tests.py`. Coefficients fit on "
        f"{SCREEN_SEASONS[0]}-{SCREEN_SEASONS[-1]}, scored unchanged on {TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]} "
        "(season = the year it ends). " + (
            f"Then the SAME coefficients on the hold-out {HOLDOUT_SEASONS[0]}-{HOLDOUT_SEASONS[-1]} "
            "(FanDuel closing lines), opened once and logged." if base_h is not None
            else "The hold-out seasons were not opened."),
        "",
        "PASS = margin error falls with t <= -2 on the tune seasons and the sign holds when refit there.",
        "Market t = does the closing line miss in the direction of the metric (|t| >= 3 = mispriced).",
        "",
        "## The model these were tested on",
        "",
        "```",
        f"screen: mae {base_s['mae']:.3f}, straight up {base_s['su']:.1%}",
        evaluate.fmt(base_t),
        *(["", "hold-out:", evaluate.fmt(base_h)] if base_h is not None else []),
        "```",
        "",
        "## Results",
        "",
        "| metric | margin error change | t | straight-up change | t | sign holds | market t | verdict |"
        + (" hold-out error change | t | hold-out market t |" if base_h is not None else ""),
        "|---|---|---|---|---|---|---|---|" + ("---|---|---|" if base_h is not None else ""),
    ]
    for r in res.itertuples():
        line = (f"| {r.test} | {r.tune_mae_change:+.4f} | {r.tune_mae_t:+.1f} | {r.tune_su_change:+.2%} | "
                f"{r.tune_su_t:+.1f} | {'yes' if r.same_sign else 'no'} | {r.tune_market_t:+.1f} | **{r.verdict}** |")
        if base_h is not None:
            line += f" {r.hold_mae_change:+.4f} | {r.hold_mae_t:+.1f} | {r.hold_market_t:+.1f} |"
        lines.append(line)
    lines += ["", "## What each metric is", ""]
    for r in res.itertuples():
        lines.append(f"- **{r.test}**: {r.what}. Screen coefficient {r.screen_beta} (t {r.screen_t}).")
    (RESEARCH / "metric_tests.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESEARCH / 'metric_tests.md'}")


def log_tests(res):
    path = RESEARCH / "test_database.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    old = old[~old.get("source", pd.Series(dtype=str)).eq("metric_tests.py")] if len(old) else old
    new = pd.DataFrame({
        "date": str(date.today()), "area": "advanced metrics", "test": res.test,
        "how": "fit 2010-2019, scored 2020-2023 (fixed coefficient)",
        "result": [f"error {r.tune_mae_change:+.4f} (t {r.tune_mae_t:+.1f}), straight up "
                   f"{r.tune_su_change:+.2%}, market t {r.tune_market_t:+.1f}"
                   + (f"; hold-out error {r.hold_mae_change:+.4f} (t {r.hold_mae_t:+.1f}), market t "
                      f"{r.hold_market_t:+.1f}" if "hold_mae_change" in res else "") for r in res.itertuples()],
        "verdict": res.verdict, "source": "metric_tests.py", "details": "research/metric_tests.md"})
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
