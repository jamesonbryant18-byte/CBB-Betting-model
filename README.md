# CBB Betting Model

A men's college basketball model that bets FanDuel spreads and moneylines,
$1 a bet, only where its edge over the FanDuel price survives an honest
backtest.

**Status: research phase done; the build plan is a draft under review.**
See [PLAN.md](PLAN.md). Data, the ratings model, tuning and backtests are
built. Nothing in the plan is built until it is approved.

## How it works, in one paragraph

Every team gets a rating in points per game, rebuilt every morning from that
season's results with only what was known before that morning. The rating
starts from last season's rating, pulled toward the conference average. How
much of last season it keeps depends on how much of the roster's production
came back. Each past result is corrected for who sat, weighting players by
how much they produce. A projected margin becomes a win and cover chance.
That chance is blended toward FanDuel's own price by exactly as much as the
model has beaten closing lines in the past. A bet needs enough expected
value at FanDuel's number. Details: the docstring at the top of each file in
`src/cbb/`.

## Which games

At least one team from the ACC, Big East, Big Ten, Big 12 or SEC, or any AP
top-25 team. Regular season plus the NCAA tournament only, with no
conference tournaments and no NIT. At most 8 bets Monday to Friday and 8 on
Saturday and Sunday, first come first served. All rules are in one place:
`BetRules` in `src/cbb/config.py`.

## Where the numbers are

No document in this repo types a result by hand. Every number comes from a
script, so it always matches the code:

| file | made by | what |
|---|---|---|
| `research/RESULTS.md` | `scripts/backtest.py` | the model replayed and graded, bets included |
| `research/metric_tests.md` | `scripts/metric_tests.py` | which advanced metrics help, pre-registered test |
| `research/line_audit.md` | `scripts/audit_lines.py` | can the stored closing lines be trusted |
| `research/line_movement.md` | `scripts/line_movement.py` | does the line move toward the model (bet early vs late) |
| `research/tuning_log.csv` | `scripts/tune.py` | every parameter setting ever tried |
| `research/holdout_log.csv` | `scripts/backtest.py --holdout` | every time the hold-out seasons were opened |
| `research/test_database.csv` | the scripts above | one row per idea tested and its verdict |
| `data/params.json` | `scripts/tune.py --save` | the fitted parameters the model runs on |

## Seasons

A season is named by the year it ends (2026 = 2025-26).

| split | seasons | lines | used for |
|---|---|---|---|
| screen | 2010-2019 | none | fitting ratings, testing metrics |
| tune | 2020-2023 | consensus close | fitting market and trust settings, confirming metrics |
| hold-out | 2024-2026 | FanDuel close | scored once, never fit on |

## Setup and research commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python scripts/fetch.py              # results, box scores, closing lines (slow the first time)
.venv/bin/python -m pytest                     # must pass before anything else
.venv/bin/python scripts/audit_lines.py
.venv/bin/python scripts/tune.py --stage base --save      # then avail, market, sigma, blend
.venv/bin/python scripts/build_features.py
.venv/bin/python scripts/metric_tests.py
.venv/bin/python scripts/backtest.py            # add --holdout only for the final, logged look
.venv/bin/python scripts/line_movement.py
```

## Layout

```
src/cbb/
  config.py     every setting, once (rules by hand, parameters from data/params.json)
  data.py       results and box scores (hoopR archive + ESPN for the current season)
  espn.py       ESPN scoreboard and box scores
  odds.py       closing and live lines (Action Network), team-name crosswalk
  players.py    player value, who is missing, returning production
  ratings.py    the weighted ridge fit behind every rating
  model.py      THE model: rate(season, date) and predict(games)
  replay.py     runs the model day by day over past seasons (the backtest)
  pricing.py    odds, de-vig, win and cover chances
  bets.py       which bets qualify, the 8-and-8 card, grading
  evaluate.py   the scorecard every result is reported with
  features.py   advanced metrics under test (not in the model unless they pass)
scripts/        fetch, tune, test, backtest
tests/          leak test, sign conventions, bet rules
```
