# Working on this repo

Rules that keep this model honest. Each one exists because the NFL model
broke it at some point.

1. **One prediction path.** `Model.rate` + `Model.predict` (src/cbb/model.py)
   is the only code that turns data into a projection. The backtest
   (`replay.py`), the tuning, the metric tests and the daily run all call it.
   Never write a second copy "for the backtest".
2. **No peeking.** `rate(season, asof)` may only use games dated before
   `asof` and the lines of those games. `tests/test_model.py` proves it with a
   leak test. Any new input needs a case in that test.
3. **Hold-out is sacred.** Nothing is fit on `HOLDOUT_SEASONS`. Opening it
   means running `scripts/backtest.py --holdout`, which logs itself to
   `research/holdout_log.csv`. Lead with hold-out numbers when reporting.
4. **Numbers come from scripts.** Docs never type a result by hand. They
   point at `research/RESULTS.md` and the other generated files. If a doc
   needs a number, a script writes it.
5. **Settings live once.** Rules are set by hand in `BetRules`. Fitted
   parameters live in `ModelParams` and reach disk only through
   `scripts/tune.py --save`. `load_params` refuses unknown keys. Never add a
   setting that also lives somewhere else.
6. **Advanced metrics enter the model only by passing**
   `scripts/metric_tests.py`: fit 2010-2019, scored with the fixed coefficient
   on 2020-2023, error must fall with t <= -2, and the sign must hold when
   refit. Log every test in `research/test_database.csv`, failures included.
7. **Ticket sign convention.** `home_spread` is the number printed on a home
   ticket (home -5.5 = home favored by 5.5). Home covers if
   `margin + home_spread > 0`. `tests/test_pricing_and_odds.py` pins it.
8. **His rules:** FanDuel only, spreads and moneylines, $1 flat, no spread
   bets on whole numbers, moneylines only between -350 and +250, 8 weekday
   and 8 weekend bets first come first served, NCAA tournament but no
   conference tournaments or NIT. Change them only when he asks.
9. Run `.venv/bin/python -m pytest` before every commit.
10. The repo is public. No personal details in code, docs or commits.
