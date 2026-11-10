# Backtest results

Generated 2026-11-09 21:28 by `scripts/backtest.py` from `data/params.json`. Every game is predicted with only what was known that morning (`replay.py`, the same `Model` the daily run uses). Bets are priced at the closing line of the book in the data (consensus before 2023-24, FanDuel after), $1 each.

## Tune seasons 2019-20 to 2022-23 (fit here)

```
games 20276, with a line 20270
straight up   model 72.4% vs favorite 72.7%  (diff -0.33%, t -1.9); picks against the favorite won 47.2% of 1176
margin error  model 8.90 vs line 8.81  (diff +0.095, t +6.8)
beyond line   +0.287 (se 0.036, t +7.9)  [0 = no information the line lacks]
win % (Brier) model 0.1805 vs market 0.1793  (diff +0.0012, t +3.3, n 20217)
```

| bets | record | ROI per $1 | SE | units | model said | won |
|---|---|---|---|---|---|---|
| every qualifying bet: 350 | 175-175-0 | +5.5% | 5.9% | +19.4 | 50.2% | 50.0% |
| the card (8 + 8 a week, best chance first): 281 | 138-143-0 | +4.9% | 6.6% | +13.7 | 49.6% | 49.1% |
| the card, best expected value first: 281 | 138-143-0 | +6.1% | 6.7% | +17.2 | 49.3% | 49.1% |

The card by kind of bet:

| bets | record | ROI per $1 | SE | units | model said | won |
|---|---|---|---|---|---|---|
| moneyline, favorite: 5 | 3-2-0 | -1.9% | 40.4% | -0.1 | 62.9% | 60.0% |
| moneyline, underdog: 116 | 46-70-0 | +3.3% | 12.1% | +3.9 | 40.6% | 39.7% |
| spread, favorite: 71 | 34-37-0 | -8.4% | 11.4% | -5.9 | 55.5% | 47.9% |
| spread, underdog: 89 | 55-34-0 | +17.8% | 9.9% | +15.8 | 55.8% | 61.8% |

| window | windows | bets per window | windows with no bet | windows up | windows down |
|---|---|---|---|---|---|
| weekday | 83 | 2.1 | 29% | 21 | 38 |
| weekend | 79 | 1.3 | 44% | 27 | 17 |

| games | n | model SU | favorite SU | model error | line error | beyond line (t) |
|---|---|---|---|---|---|---|
| both power | 3017 | 69.2% | 69.3% | 9.07 | 9.00 | +0.39 (+3.7) |
| neither power | 15326 | 71.3% | 71.8% | 8.80 | 8.71 | +0.27 (+6.2) |
| one power | 1927 | 85.6% | 85.6% | 9.47 | 9.32 | +0.29 (+3.3) |

- 2019-20: SU 72.7% vs favorite 73.7%; error 8.83 vs line 8.73; beyond line +0.13
- 2020-21: SU 71.2% vs favorite 71.6%; error 9.21 vs line 9.06; beyond line +0.19
- 2021-22: SU 72.7% vs favorite 72.8%; error 8.77 vs line 8.71; beyond line +0.46
- 2022-23: SU 72.5% vs favorite 72.5%; error 8.89 vs line 8.81; beyond line +0.23

Calibration (favorite's chance said vs won):

| said | games | won | gap in SE |
|---|---|---|---|
| 54.8% | 5135 | 55.2% | +0.4 |
| 65.0% | 4776 | 64.9% | -0.1 |
| 74.9% | 4174 | 75.2% | +0.5 |
| 84.8% | 3386 | 86.0% | +2.0 |
| 94.9% | 2799 | 95.3% | +1.0 |

## HOLD-OUT 2023-24 to 2025-26 (never fit on; bets only at FanDuel's own closing number)

```
games 17180, with a line 17000
straight up   model 72.6% vs favorite 72.9%  (diff -0.37%, t -2.0); picks against the favorite won 46.9% of 1025
margin error  model 8.97 vs line 8.86  (diff +0.112, t +7.3)
beyond line   +0.110 (se 0.042, t +2.6)  [0 = no information the line lacks]
win % (Brier) model 0.1809 vs market 0.1784  (diff +0.0026, t +6.9, n 16491)
```

| bets | record | ROI per $1 | SE | units | model said | won |
|---|---|---|---|---|---|---|
| every qualifying bet: 409 | 191-218-0 | -5.1% | 5.1% | -20.9 | 51.8% | 46.7% |
| the card (8 + 8 a week, best chance first): 301 | 134-167-0 | -8.7% | 6.0% | -26.3 | 51.0% | 44.5% |
| the card, best expected value first: 301 | 134-167-0 | -8.6% | 6.0% | -26.0 | 50.9% | 44.5% |

The card by kind of bet:

| bets | record | ROI per $1 | SE | units | model said | won |
|---|---|---|---|---|---|---|
| moneyline, favorite: 6 | 2-4-0 | -35.1% | 41.1% | -2.1 | 53.0% | 33.3% |
| moneyline, underdog: 78 | 26-52-0 | -12.1% | 14.5% | -9.4 | 38.6% | 33.3% |
| spread, favorite: 54 | 27-27-0 | -5.1% | 13.0% | -2.8 | 56.0% | 50.0% |
| spread, underdog: 163 | 79-84-0 | -7.4% | 7.5% | -12.0 | 55.3% | 48.5% |

| window | windows | bets per window | windows with no bet | windows up | windows down |
|---|---|---|---|---|---|
| weekday | 69 | 2.5 | 23% | 19 | 34 |
| weekend | 66 | 1.9 | 27% | 18 | 30 |

| games | n | model SU | favorite SU | model error | line error | beyond line (t) |
|---|---|---|---|---|---|---|
| both power | 2894 | 71.3% | 72.2% | 8.98 | 8.82 | -0.10 (-0.9) |
| neither power | 12216 | 70.2% | 70.5% | 8.85 | 8.76 | +0.15 (+2.9) |
| one power | 1890 | 90.0% | 89.9% | 9.73 | 9.56 | +0.12 (+1.3) |

- 2023-24: SU 72.5% vs favorite 72.7%; error 8.94 vs line 8.85; beyond line +0.21
- 2024-25: SU 72.7% vs favorite 73.3%; error 8.93 vs line 8.80; beyond line +0.05
- 2025-26: SU 72.5% vs favorite 72.9%; error 9.04 vs line 8.93; beyond line +0.08

Calibration (favorite's chance said vs won):

| said | games | won | gap in SE |
|---|---|---|---|
| 55.0% | 4248 | 55.4% | +0.5 |
| 64.9% | 4064 | 63.8% | -1.5 |
| 74.8% | 3456 | 75.5% | +0.9 |
| 84.8% | 2653 | 86.8% | +2.8 |
| 95.4% | 2579 | 96.1% | +1.8 |

