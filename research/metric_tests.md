# Advanced metric tests

Run 2026-11-09 by `scripts/metric_tests.py`. Coefficients fit on 2010-2019, scored unchanged on 2020-2023 (season = the year it ends). Then the SAME coefficients on the hold-out 2024-2026 (FanDuel closing lines), opened once and logged.

PASS = margin error falls with t <= -2 on the tune seasons and the sign holds when refit there.
Market t = does the closing line miss in the direction of the metric (|t| >= 3 = mispriced).

## The model these were tested on

```
screen: mae 8.724, straight up 73.7%
games 20270, with a line 20270
straight up   model 72.4% vs favorite 72.7%  (diff -0.33%, t -1.9); picks against the favorite won 47.2% of 1176
margin error  model 8.90 vs line 8.81  (diff +0.095, t +6.8)
beyond line   +0.287 (se 0.036, t +7.9)  [0 = no information the line lacks]
win % (Brier) model 0.1805 vs market 0.1793  (diff +0.0012, t +3.3, n 20217)

hold-out:
games 17000, with a line 17000
straight up   model 72.6% vs favorite 72.9%  (diff -0.37%, t -2.0); picks against the favorite won 46.9% of 1025
margin error  model 8.97 vs line 8.86  (diff +0.112, t +7.3)
beyond line   +0.110 (se 0.042, t +2.6)  [0 = no information the line lacks]
win % (Brier) model 0.1809 vs market 0.1784  (diff +0.0026, t +6.9, n 16491)
```

## Results

| metric | margin error change | t | straight-up change | t | sign holds | market t | verdict | hold-out error change | t | hold-out market t |
|---|---|---|---|---|---|---|---|---|---|---|
| four factors (all 10) | -0.0053 | -0.6 | -0.07% | -0.4 | no | +3.7 | **TOO WEAK** | -0.0026 | -0.3 | +2.8 |
| efg (offense + defense) | -0.0049 | -0.6 | +0.02% | +0.1 | yes | +2.3 | **TOO WEAK** | +0.0076 | +0.8 | +1.7 |
| tov (offense + defense) | -0.0006 | -0.3 | +0.04% | +0.5 | yes | +0.7 | **TOO WEAK** | -0.0002 | -0.1 | -0.7 |
| orb (offense + defense) | +0.0010 | +1.1 | +0.05% | +1.1 | no | -3.7 | **NO HELP** | +0.0044 | +4.3 | -4.0 |
| ftr (offense + defense) | -0.0034 | -2.2 | -0.03% | -0.5 | yes | +2.4 | **PASS** | +0.0014 | +0.8 | +0.8 |
| 3par (offense + defense) | -0.0003 | -0.4 | +0.04% | +1.0 | no | +1.4 | **TOO WEAK** | +0.0004 | +0.6 | +0.6 |
| 3-point defense luck | -0.0048 | -0.5 | -0.08% | -0.6 | yes | +2.5 | **TOO WEAK** | -0.0039 | -0.4 | +2.8 |
| 3-point offense luck | -0.0016 | -0.2 | -0.13% | -1.0 | yes | +3.3 | **TOO WEAK** | -0.0180 | -2.3 | +4.2 |
| free-throw defense luck | -0.0078 | -2.1 | -0.09% | -1.0 | yes | +3.5 | **PASS** | -0.0021 | -0.5 | +2.8 |
| all shooting luck | -0.0122 | -1.1 | -0.04% | -0.2 | yes | +4.6 | **TOO WEAK** | -0.0208 | -1.7 | +5.2 |
| pace scaling | -0.0004 | -2.6 | +0.00% | +nan | yes | -0.4 | **PASS** | +0.0002 | +1.3 | -0.8 |
| margin stretch/shrink | -0.0106 | -1.8 | +0.00% | +nan | yes | -1.1 | **TOO WEAK** | +0.0166 | +2.5 | -1.7 |
| conference-game compression | -0.0084 | -1.8 | +0.00% | +nan | yes | -0.1 | **TOO WEAK** | -0.0084 | -1.6 | -0.1 |
| rest days | -0.0000 | -0.1 | -0.01% | -1.0 | yes | +0.9 | **TOO WEAK** | +0.0003 | +2.3 | -1.0 |
| recent form | +0.0194 | +4.4 | -0.16% | -1.5 | no | -0.5 | **NO HELP** | +0.0229 | +4.8 | -0.4 |
| team-specific home court | -0.0036 | -2.0 | +0.00% | +0.0 | yes | +2.3 | **PASS** | -0.0013 | -0.6 | +0.7 |
| fresh absences (a key player just went out) | +0.0000 | +0.0 | +0.01% | +0.8 | no | -3.2 | **NO HELP** | +0.0002 | +0.9 | -4.0 |

## What each metric is

- **four factors (all 10)**: Opponent-adjusted shooting, turnovers, rebounding, free throws, 3-point rate, offense and defense. Screen coefficient [-29.414, 8.027, 2.409, 4.682, 5.769, 41.741, 6.419, -14.029, -3.274, -7.208] (t [-15.6, 2.9, 1.6, 4.6, 4.9, 21.3, 2.5, -7.2, -3.5, -5.2]).
- **efg (offense + defense)**: Opponent-adjusted efg. Screen coefficient [-24.557, 36.128] (t [-14.5, 20.0]).
- **tov (offense + defense)**: Opponent-adjusted tov. Screen coefficient [18.181, 0.765] (t [7.0, 0.3]).
- **orb (offense + defense)**: Opponent-adjusted orb. Screen coefficient [-3.205, 1.849] (t [-2.3, 1.0]).
- **ftr (offense + defense)**: Opponent-adjusted ftr. Screen coefficient [-0.24, 4.486] (t [-0.2, 5.3]).
- **3par (offense + defense)**: Opponent-adjusted 3par. Screen coefficient [2.057, 0.147] (t [2.0, 0.1]).
- **3-point defense luck**: Opponents' 3s made above league average (KenPom: mostly luck). Screen coefficient [0.666] (t [26.3]).
- **3-point offense luck**: Own 3s made above league average. Screen coefficient [-0.51] (t [-21.9]).
- **free-throw defense luck**: Opponents' free throws made above league average. Screen coefficient [0.896] (t [12.2]).
- **all shooting luck**: The three luck terms together. Screen coefficient [0.629, -0.495, 0.728] (t [24.8, -21.3, 10.0]).
- **pace scaling**: Scale the margin by expected possessions (per-possession ratings). Screen coefficient [-0.038] (t [-0.6]).
- **margin stretch/shrink**: Is the projected margin too big or too small?. Screen coefficient [-0.084] (t [-19.9]).
- **conference-game compression**: Are conference games closer than projected?. Screen coefficient [-0.104] (t [-15.3]).
- **rest days**: Days since last game, home minus away (capped at 7). Screen coefficient [-0.011] (t [-0.4]).
- **recent form**: Beat-the-projection over the last 5 games. Screen coefficient [-0.08] (t [-11.1]).
- **team-specific home court**: This arena's beat-the-projection the 3 seasons before. Screen coefficient [0.332] (t [4.8]).
- **fresh absences (a key player just went out)**: Value of players out 3 games or fewer, home minus away: do new injuries need MORE than the model's flat weight?. Screen coefficient [0.232] (t [0.5]).
