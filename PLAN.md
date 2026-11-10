# The plan: final draft for review

**Status: draft, not approved. Nothing in this plan is built yet** except the
research phase, which is in `research/`. Read it, mark it up, and nothing
gets built until you sign off.

Results are never typed into this file. Where a finding matters, the file
that produced it is named, so the plan can never disagree with the data
(rule 4 in `CLAUDE.md`).

---

## 1. What the model is for

1. **Pick winners.** Straight-up accuracy is the headline number. The bar to
   beat is "always take the betting favorite".
2. **Find bets with a real edge** at FanDuel: spreads and moneylines, $1
   flat, under the bet rules in section 9.
3. **Be especially good at underdogs who keep it closer than the market
   thinks.** This is a strength to build, not the only thing it does.
   Favorites and outright winners matter just as much.
4. **One clean, repeatable codebase.** One command per run, one prediction
   path, nothing typed twice.
5. **A professional Excel dashboard**, visual first, including the
   simulations (section 11).

## 2. What the research already showed

The details are in the generated reports. In short:

| finding | where |
|---|---|
| The ratings backbone is about as accurate as the closing line: no better, no worse. It picks winners at about the favorite's rate. | `research/RESULTS.md`, hold-out section |
| At FanDuel's **closing** number the bets lost money in the hold-out seasons. | `research/RESULTS.md` |
| The line **moves toward the model** between the open and the close, so the model knows things the opening line does not. At the opening number the same bets made money, but not by enough to call it proven. | `research/line_movement.md` |
| No box-score metric (four factors, 3-point rate, rest, form, home court by team) improved the model on seasons it was not fit on. They are ingredients of the scoring margin the ratings already use. | `research/metric_tests.md` |
| Shooting luck (3-point and free-throw percentages) is the one lead: the market over-reacts to it, in both test periods. | `research/metric_tests.md` |
| Bigger underdogs (10 to 20 points) covered slightly more than half the time in 2020-23. That is close to break-even, so it is not a bet on its own, but it is the effect the simulator is built to explain (section 5). | to be written by `scripts/underdogs.py` (phase A) |
| The stored lines are pregame numbers (not in-game), and FanDuel's 2025-26 gap is filled. | `research/line_audit.md` |

## 3. The rules that keep it bulletproof

These are already in `CLAUDE.md` and apply to everything below:

- **One prediction path.** The backtest, the weekly evaluation and the
  live run all call the same `Model`. A "live == replay" check runs the live
  pipeline on a past date and must reproduce the backtest's picks for that
  date exactly.
- **No peeking.** Every number for a game uses only what was known that
  morning, and the leak test covers every new input.
- **Settings live once.** Bet rules are set by hand in `config.py`. Fitted
  values are written only by scripts. Variable switches live in one
  registry.
- **Gates are written down before the results** (section 12). A piece that
  fails its gate is dropped, whatever it cost to build.
- **Every look at the hold-out is logged** (`research/holdout_log.csv`). The
  2024-26 seasons have been looked at once for the backbone. New pieces get
  one logged look there, and **the live 2026-27 season is the final judge.**
- Tests pass before every commit. No personal details in the public repo.

## 4. The living database (updated as the season goes)

Every run, before any prediction, the model brings itself up to date:

1. **Games since the last run:** results, team box scores, player box
   scores and play-by-play from ESPN. If runs were skipped for a few days,
   the next run catches up every game it missed.
2. **The player database** is rebuilt from those box scores. For every
   player on every D1 roster:
   - team, games, minutes, and production per game
   - his value (share of team production)
   - rotation status (15+ minutes a game)
   - games missed in a row, and whether he played the last game
   - previous team and production, for transfers
3. **Team stats** are rebuilt from every game so far:
   - rating (points better than average)
   - offense and defense
   - pace
   - style fingerprint (section 5)
   - shooting luck
   - returning and incoming production
   - who is out
4. **Lines:** FanDuel's current numbers for upcoming games, and closing
   numbers for finished games. These feed the market rating and the
   closing-line check on every bet.
5. **Injury notes:** a small file you can edit (`data/injury_notes.csv`:
   player, team, out or questionable, date) for news the box scores cannot
   know yet, like a star ruled out this morning.
6. **Freshness check.** The dashboard prints "data through <date, time>"
   and a coverage line: games missing a box score, play-by-play or a line.
   If a game on the card is missing data, the run says so and does not bet
   it.

Everything except the injury notes is *derived* from the downloaded raw
data, so the database can be rebuilt from scratch at any time and comes out
identical. Nothing is edited by hand, so nothing drifts.

## 5. The model

### 5.1 The backbone (built and tested in the research phase)

| piece | what it is |
|---|---|
| Team rating | Points better than an average D1 team on a neutral floor, adjusted for every opponent, rebuilt every morning |
| Home court | Refit daily, zero on neutral courts |
| Market rating | Team ratings backed out of the closing lines of every game played so far this season |
| Preseason rating | Last season's rating pulled toward the conference average; how much is kept depends on returning production (plus incoming transfers, phase C) |
| Player availability | Rotation players who are out cost their team points in proportion to their share of production |

The backbone says *how good each team is*. It sets the expected margin.

### 5.2 Style fingerprints (new)

Each team gets a profile from play-by-play, opponent-adjusted the same way
as the rating:

- **Offense:** pace, how often it turns it over, its shot mix (rim, other
  twos, threes), its offensive rebounding and how often it gets to the line.
- **Defense:** the same things, on the other end.
- **Make rates are shrunk hard toward average.** How many threes a team
  takes is a skill. How many happened to fall is largely luck, and that is
  where the market over-reacts.

### 5.3 The game simulator (new: the creative core)

Every game is played 10,000 times, possession by possession, on the
computer:

1. Each possession's outcome comes from the offense's fingerprint against
   the defense's: turnover, shot type, make or miss, offensive rebound,
   free throws.
2. **Game-state behavior is learned from 17 seasons of play-by-play:**
   - a team with a big lead slows down and empties the bench
   - a trailing team speeds up and shoots more threes
   - late in close games the trailing team fouls
   - these are what turn a 20-point projection into a 12-point final, or a
     close game into a backdoor cover
3. **The simulator's average margin is pinned to the backbone's.** It adds
   the *shape* of each game, not a second, competing opinion on who is
   better. So the model never says two different things. Only if the weekly
   evaluation proves the simulator's own average adds information does it
   get a say in the average too.

What comes out, per game:

- win chance for each team
- cover chance at any spread, so it can price FanDuel's number and say
  "bet if FanDuel posts X or better"
- the full range of final margins
- how the game is likely to unfold minute by minute (for the visuals)

**Why this should find underdogs.** The market mostly prices the average
margin. Whether an underdog covers or wins depends on the *shape* of the
game:

- A slow, defense-first underdog shrinks the game and keeps it inside big
  spreads.
- A three-point-heavy underdog swings wider and pulls more outright upsets.
- A favorite that coasts with a lead gives back garbage-time points.

The same machinery works for favorites. It also says when a favorite's
shape means it will win by *more* than the line.

Speed: the simulator is plain array math (numpy). A day's slate takes
seconds. Backtests use fewer runs per game (a setting, same code). A test
checks that the answers match the 10,000-run version within noise.

### 5.4 From simulation to a pick

- **Winner pick:** the team the simulator gives more than 50%.
- **Spread and moneyline chances** come from the simulated outcomes,
  blended toward FanDuel's own price by as much as the model has earned. The
  weekly evaluation keeps that blend honest, so a pick's stated chance
  matches how often such picks win (the NFL model's bets said 58% and won
  54%; that gap is what the blend prevents).
- **Underdog Watch:** every underdog whose simulated cover or upset chance
  beats the market, ranked, bet or not.

## 6. Variables: in, out, on trial

Every variable is a switch in one registry (`src/cbb/registry.py`). The
weekly evaluation (section 7) decides which are on. Each variable can be
tested alone, and each comes with a short note on what it measures.

| variable | status |
|---|---|
| Team rating, home court, market rating, preseason, returning production, availability | **in** (backbone) |
| Style fingerprints and the simulator | **new**, gate B |
| Shooting-luck regression | **new**, built into the fingerprints |
| Incoming transfers (old production, adjusted for old conference level) | **new**, phase C |
| Blowout discount (big wins count less when rating teams) | **new**, phase C |
| Public vs sharp money (ticket % vs money %, line moving against the public) | **new**, phase C (data from 2019-20 on) |
| National TV game | **new**, phase C (data from 2019-20 on) |
| Travel distance and time zones | **new**, phase C |
| Roster experience and height | **new**, phase C, if past-season roster data can be found |
| Four factors, 3-point rate, rest, recent form, team home court, fresh injuries | **out**: failed in `research/metric_tests.md`. They stay in the registry and get re-tested weekly like everything else, so they can earn their way back in. |

## 7. The weekly evaluation engine (built first)

**When:** every Monday run, after the database update. Weekly, not daily:
daily would react to noise.

**What it does:**

1. **Grades the current model** on every game this season: winners vs the
   favorite, margin error vs the line, underdog covers, calibration, and
   closing-line value of the bets, by week and by kind of game.
2. **Explains every loss:** which variables pushed that pick the wrong way.
   This is the "a string of losses went one way" view, shown honestly.
3. **Re-checks every variable, on and off.** Each variable's weight is fit
   on everything before the most recent weeks, then scored on those weeks,
   which it never saw, plus on all past seasons the same way.
4. **Decision rule**, the same bar as the research:
   - a change must help on this season's unseen weeks *and* on past seasons
   - it must improve winner picking or margin accuracy without hurting the
     other
   - it must clear t <= -2
   - at most **one** variable change a week
   - weights refit every week
5. **Logs everything.** Every proposed or made change gets a dated row in
   `research/change_log.csv`: what, why, the numbers, the new model
   version. Every pick in the workbook shows the version that made it.
   Changes can be reversed by the same rule.

**Why it will not chase streaks.** With 20+ variables, one will always
explain last week's losses by chance. The rule asks a different question:
does it flip more picks right than wrong, over thousands of games it was not
fit on? A real effect passes. A streak does not.

**The engine is tested itself before it is trusted.** It is replayed over
past seasons week by week, as if live. Gate A asks whether the changes it
made helped the weeks *after* them.

## 8. Run schedule

| when | run | what it does |
|---|---|---|
| Monday morning | weekly evaluation + weekday card | update the database, run the weekly evaluation, picks for Mon to Fri: bets for games with FanDuel lines, "bet if FanDuel posts X or better" for the rest |
| Saturday morning | weekend card | update, picks for Sat and Sun |
| Morning of a bet day | bet-day run | update lines and injuries; confirm or pass each bet at the live number; slots already taken keep their place |

One command: `.venv/bin/python run.py`. It works out which run it is from
the date, and each run writes a new workbook titled with the date.
Research showed the line tends to move toward the model before tip, so a bet
is best placed when the run publishes it, not at game time.

## 9. Bet rules (already in `config.py`, unchanged)

- FanDuel only, spreads and moneylines, $1 flat.
- Bettable games: at least one ACC, Big East, Big Ten, Big 12 or SEC team,
  or any AP top-25 team.
- Regular season plus the NCAA tournament. No conference tournaments, no
  NIT.
- No spread bets on whole numbers. Moneylines only between -350 and +250.
- At most 8 bets Monday to Friday and 8 on Saturday and Sunday, first come
  first served.
- A bet needs enough expected value at FanDuel's number (`min_ev`). The card
  is not forced to fill: a forced bet just pays the vig.
- Every bet records FanDuel's closing number afterward. Beating the close
  is the early proof of skill, long before the win-loss record means
  anything.

## 10. The workbook

One file per run: `output/CBB_Model_YYYY-MM-DD.xlsx`, built with native
Excel charts (`xlsxwriter`). The bet tracker is rebuilt every run from the
saved picks in `picks/`, so it can never drift from what was actually
published.

| sheet | what is on it |
|---|---|
| **Dashboard** | KPI tiles (record, units, winners vs favorite, underdog covers, beat-the-close rate, model version); cumulative units; weekly record; underdog cover rate by spread size; calibration (said vs won); this window's card |
| **Card** | each bet: time, matchup, bet, FanDuel number, bet-if number, win %, cover %, edge bar, top 3 reasons for the pick |
| **Game Cards** | one panel per bet, with the simulation visuals (section 11) |
| **All Games** | every bettable game: projected score, model vs market, win %, upset chance, colored by edge |
| **Underdog Watch** | underdogs ranked by simulated cover and upset chance vs the market |
| **Teams** | power ratings with trend sparklines, offense, defense, pace, style fingerprint heatmap, luck, returning and incoming production |
| **Injuries** | who is out, his value, points impact, and your notes |
| **Model Lab** | variables on and off, this week's re-check results, change log, accuracy trend by model version |
| **Simulation Lab** | proof the simulator is honest: when it said X%, how often it happened |
| **Bet Tracker** | every bet ever: result, units, closing-line value, running total |

**A prototype comes first** (phase D): the full workbook built from a week
of last season's data, so you approve the look before anything goes live.

## 11. Simulation visuals

The simulator plays every game 10,000 times, so there is a lot to show. In
Excel, on each Game Card:

1. **Outcome range chart.** A histogram of the 10,000 final margins, with
   FanDuel's spread drawn as a line and the cover side shaded. You can see
   how often the underdog stays inside the number.
2. **Game-flow fan chart.** The expected score margin minute by minute
   across 40 minutes, with bands for the likely range: a favorite pulling
   away late, or an underdog hanging around all night.
3. **Sample games.** About 20 individual simulated games drawn as lines,
   so "10,000 simulations" is something you can see, not just a number.
4. **Final-score heatmap.** A grid of home score against away score,
   colored by how often each final came up, with the most likely final
   marked.
5. **Chance meters.** Win %, cover % and upset % as gauges, next to
   FanDuel's implied chance.

On the Simulation Lab sheet: **calibration charts** that show the simulator
earns its keep. Of the games where it said the underdog covers 60% of the
time, how many did?

Two options outside the base plan (see section 14):

- an **animated simulation viewer**: a page that opens in the browser and
  replays simulated games possession by possession. Excel cannot animate.
- a **March bracket simulator**: the NCAA tournament played 10,000 times,
  with each team's odds to reach each round.

## 12. Build order and gates

Each phase has to pass its gate before the next one starts. The gates are
fixed now.

| phase | what gets built | gate (pass/fail, decided in advance) |
|---|---|---|
| **A. Evaluation engine** | variable registry, weekly grading, loss explanations, re-check, decision rule, change log, model versions, `scripts/underdogs.py` | Replayed week by week over past seasons, the changes it makes help the weeks after them on average, and it makes few changes. Unit tests pass. |
| **B. Play-by-play and simulator** | play-by-play loader, style fingerprints, game-state behavior, simulator, simulation tests | On 2020-23 the simulator's win and cover chances beat the current model's (log loss, t <= -2), and its calibration holds within 2 standard errors in every bucket. Then one logged look at 2024-26. **If it fails, it is dropped and the backbone carries on.** |
| **C. New variables** | transfers, blowout discount, public money, TV, travel, experience and height | Each one passes the weekly decision rule on its own. |
| **D. Dashboard prototype** | the full workbook on last season's data | You approve the look. |
| **E. Live runs** | `run.py`, the database update, picks archive, bet tracker, injury notes | Tests pass; the live pipeline run on a past date reproduces the backtest picks exactly; one dry run on today's slate. |

**Success, measured live in 2026-27:**

- winners at or above the favorite's rate
- underdogs the model backs covering above 52.4% (the break-even at -110)
  over 300+ games
- calibration within 2 standard errors
- beating FanDuel's closing number on more than half the bets

## 13. Files this adds

```
run.py                   the one command
src/cbb/pbp.py           play-by-play: download, possessions, game state
src/cbb/style.py         style fingerprints (opponent-adjusted, no peeking)
src/cbb/simulate.py      the game simulator
src/cbb/registry.py      every variable, its switch, what it measures
src/cbb/weekly.py        the weekly evaluation engine
src/cbb/database.py      the player and team database, rebuilt each run
src/cbb/workbook.py      the Excel dashboard
scripts/underdogs.py     underdog research (cover rates by spread and situation)
data/injury_notes.csv    your injury notes (the only hand-edited data)
research/change_log.csv  every weekly change, with its numbers
picks/                   every published card, one file per run
output/                  the dated workbooks (not in git)
```

## 14. Decisions for your review

1. **Weekly changes: automatic, or with your OK?** Suggested: weights refit
   automatically, but turning a variable on or off waits for your OK in the
   first season.
2. **Animated simulation viewer** in the browser, in addition to the Excel
   charts: yes or no?
3. **March bracket simulator:** yes or no?
4. Anything to add, cut or change in sections 1 to 13.

## 15. What could go wrong (and what happens if it does)

| risk | what happens |
|---|---|
| The simulator does not beat the current model | Dropped at gate B. The backbone is already tested and keeps running. |
| The weekly engine chases noise | Gate A tests exactly that before it runs live, and changes need your OK in season 1. |
| A data source goes down (ESPN, Action Network) | The freshness check catches it: no bets on games with missing data, and the dashboard says why. |
| FanDuel posts lines late | The Monday card gives bet-if numbers for later games, and the bet-day run confirms them. |
| The season has already started | Bets begin only when the gates pass. The model can run on paper meanwhile. |
