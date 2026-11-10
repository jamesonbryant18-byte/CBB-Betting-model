"""
Fit the model's parameters, in stages, on seasons the hold-out never touches.

    .venv/bin/python scripts/tune.py --stage base    # ratings, priors: SCREEN seasons
    .venv/bin/python scripts/tune.py --stage market  # market prior:   TUNE seasons
    .venv/bin/python scripts/tune.py --stage avail   # injuries:       SCREEN seasons
    .venv/bin/python scripts/tune.py --stage sigma   # win-% spread:   SCREEN + TUNE
    .venv/bin/python scripts/tune.py --stage blend   # trust vs line:  TUNE seasons

Coordinate search: one parameter at a time over a short grid, keep the best,
repeat until nothing moves. Every evaluation is appended to
research/tuning_log.csv. The result is written to data/params.json only with
--save. HOLDOUT_SEASONS are never replayed here.
"""
import argparse
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from cbb import evaluate, pricing, replay
from cbb.config import (HOLDOUT_SEASONS, RESEARCH, SCREEN_SEASONS, TUNE_SEASONS,
                        load_params, save_params)
from cbb.model import Model

GRIDS = {
    "base": {
        "ridge_lambda": [1.5, 3.0, 5.0, 8.0],
        "half_life_days": [30.0, 45.0, 60.0, 90.0, 120.0, 180.0],
        "hfa": [2.5, 3.0, 3.5],
        "carry_base": [0.3, 0.45, 0.6],
        "carry_continuity": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8],
    },
    "avail": {"avail_points": [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0],
              "value_production_weight": [0.0, 0.5, 1.0]},
    "market": {
        "market_lambda": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
        "ridge_lambda": [0.75, 1.5, 3.0, 5.0],
    },
}
STAGE_SEASONS = {"base": SCREEN_SEASONS, "avail": SCREEN_SEASONS, "market": TUNE_SEASONS}


def evaluate_params(model, params, seasons, log, stage, who_sits="proxy"):
    t = time.time()
    r = replay.run(seasons, params, model=model.with_params(params), verbose=False, who_sits=who_sits)
    card = evaluate.scorecard(r)
    row = {"stage": stage, **{k: getattr(params, k) for k in params.FITTED},
           "mae": card["mae"], "su": card["su"], "mae_vs_line": card.get("mae_diff"),
           "seconds": round(time.time() - t, 1)}
    log.append(row)
    print(f"    mae {card['mae']:.4f}  su {card['su']:.4f}  "
          + " ".join(f"{k}={getattr(params, k)}" for k in GRIDS[stage.split("-")[0]]), flush=True)
    return card["mae"], r


def coordinate_search(model, params, stage, seasons, log):
    best, _ = evaluate_params(model, params, seasons, log, stage)
    for _ in range(3):
        moved = False
        for name, grid in GRIDS[stage].items():
            for v in grid:
                if v == getattr(params, name):
                    continue
                cand = replace(params, **{name: v})
                score, _ = evaluate_params(model, cand, seasons, log, stage)
                if score < best - 1e-4:
                    best, params, moved = score, cand, True
        if not moved:
            break
    return params, best


def fit_sigma(model, params, seasons):
    """Margin spread that makes the win chances most accurate (log loss)."""
    r = replay.run(seasons, params, model=model.with_params(params), verbose=False)
    won = (r.result > 0).to_numpy()
    best = None
    for s in np.arange(9.0, 13.01, 0.25):
        for k in (0.0, 0.05, 0.1, 0.15):
            sig = s + k * (r.poss - 68.0)
            p = np.clip(pricing.win_prob(r.margin.to_numpy(), sig.to_numpy()), 1e-6, 1 - 1e-6)
            ll = -np.mean(np.where(won, np.log(p), np.log(1 - p)))
            if best is None or ll < best[0]:
                best = (ll, float(s), float(k))
    print(f"  sigma {best[1]:.2f} + {best[2]:.2f} per possession above 68  (log loss {best[0]:.4f})")
    return replace(params, sigma=best[1], sigma_per_poss=best[2])


def fit_blend(model, params, seasons):
    """
    How much of the model's disagreement with the closing line came true.

    Regress (result - line) on (model - line) over the TUNE seasons. The slope
    is the share of each disagreement that is real; bets use exactly that share
    (bets.py). Clipped to [0, 1].
    """
    r = replay.run(seasons, params, model=model.with_params(params), verbose=False)
    card = evaluate.scorecard(r)
    w = float(np.clip(card["beyond_line"], 0.0, 1.0))
    print(f"  beyond-line slope {card['beyond_line']:+.3f} (se {card['beyond_line_se']:.3f}, "
          f"t {card['beyond_line_t']:+.1f}) on {card['games_lined']} games -> blend_weight {w:.3f}")
    return replace(params, blend_weight=round(w, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "avail", "market", "sigma", "blend"], required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    assert not set(SCREEN_SEASONS + TUNE_SEASONS) & set(HOLDOUT_SEASONS)

    params = load_params()
    seasons = STAGE_SEASONS.get(args.stage) or (TUNE_SEASONS if args.stage == "blend"
                                                 else SCREEN_SEASONS + TUNE_SEASONS)
    games, plays, lines = replay.load_inputs(seasons)
    model = Model(games, plays, lines, params)
    log = []
    print(f"stage {args.stage} on {seasons[0]}-{seasons[-1]}")
    if args.stage == "sigma":
        params = fit_sigma(model, params, seasons)
    elif args.stage == "blend":
        params = fit_blend(model, params, seasons)
    else:
        params, best = coordinate_search(model, params, args.stage, seasons, log)
        print(f"best mae {best:.4f}: " + ", ".join(f"{k}={getattr(params, k)}" for k in GRIDS[args.stage]))
        if args.stage == "avail":
            # The ceiling: the same model told who REALLY sat. Not usable for bets
            # (unknowable before tip); it measures what better injury news is worth.
            print("  ceiling, if we knew exactly who sits:")
            evaluate_params(model, params, seasons, log, "avail-ceiling", who_sits="actual")
    if log:
        path = RESEARCH / "tuning_log.csv"
        df = pd.DataFrame(log)
        if path.exists():                    # align by column name, so adding a parameter never shifts old rows
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False)
    if args.save:
        save_params(params, f"tune.py --stage {args.stage}, seasons {seasons[0]}-{seasons[-1]}")
        print("saved data/params.json")


if __name__ == "__main__":
    main()
