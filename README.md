# CBB Betting Model

Men's college basketball model: picks winners and finds FanDuel spread and
moneyline bets ($1 flat) with an edge that survives honest backtesting.

**Status:** research phase done. The build plan (`docs/PLAN.md`) is a draft
awaiting the owner's approval. Do not build from it until approved, and open
it only when working on the build.

## Model today

A daily opponent-adjusted team rating (points vs an average D1 team), plus
home court, a market rating from earlier closing lines, a preseason rating
carried by returning production, and missing rotation players weighted by
production. One path: `Model.rate(season, date)` + `Model.predict(games)` in
`src/cbb/model.py`, used by every script. Rules: `CLAUDE.md`.

## Seasons

Named by the year they end (2026 = 2025-26). Screen 2010-19 (no lines),
tune 2020-23 (consensus closes), hold-out 2024-26 (FanDuel closes; opened
once, logged in `research/holdout_log.csv`).

## Commands

```bash
.venv/bin/pip install -e ".[test]"      # python 3.9 venv
.venv/bin/python scripts/fetch.py       # data (slow first time; cache in data/cache, not in git)
.venv/bin/python -m pytest              # before every commit
.venv/bin/python scripts/tune.py --stage {base,avail,market,sigma,blend} --save
.venv/bin/python scripts/metric_tests.py
.venv/bin/python scripts/backtest.py    # --holdout only for a final, logged look
```

## Where things are

- `src/cbb/`: `config` (all settings), `data`/`espn`/`odds` (inputs),
  `players`, `ratings`, `model`, `replay` (backtest), `pricing`, `bets`,
  `evaluate`, `features` (metrics under test)
- `data/params.json`: fitted parameters, written only by `tune.py --save`
- `research/`: generated reports (`RESULTS.md`, `metric_tests.md`,
  `line_audit.md`, `line_movement.md`), `tuning_log.csv`, `test_database.csv`.
  Results are quoted from these files, never typed by hand.
