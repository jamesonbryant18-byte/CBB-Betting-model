"""
Download everything the model reads, and report coverage.

    .venv/bin/python scripts/fetch.py            # all seasons, then odds
    .venv/bin/python scripts/fetch.py --current  # just refresh the current season

Box scores: every season FIRST_SEASON..CURRENT_SEASON (hoopR archive + ESPN top-up).
Lines: every date from 2019-20 on (Action Network), one cached file per day.
Writes data/team_crosswalk.csv (Action Network id -> ESPN id), which is tracked.
"""
import argparse

import pandas as pd

from cbb import data, odds
from cbb.config import CACHE, CURRENT_SEASON, FIRST_SEASON
from cbb.espn import today_et

ODDS_FIRST_SEASON = 2020


def season_dates(season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{season - 1}-11-01")
    end = min(pd.Timestamp(f"{season}-04-10"), today_et())
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", action="store_true", help="only the current season")
    args = ap.parse_args()
    seasons = [CURRENT_SEASON] if args.current else list(range(FIRST_SEASON, CURRENT_SEASON + 1))

    for s in seasons:
        for kind in data.KINDS:
            data.fetch(kind, s, refresh=(s == CURRENT_SEASON))
    games, _ = data.load(seasons)
    g = data.finals(games)
    print("season  games  with box  teams")
    for s, x in g.groupby("season"):
        print(f"{s:>6} {len(x):>6} {x.has_box.mean():>9.1%} {len(set(x.home_id) | set(x.away_id)):>6}")

    odds_seasons = [s for s in seasons if s >= ODDS_FIRST_SEASON]
    frames = []
    for s in odds_seasons:
        start, end = season_dates(s)
        o = odds.history(start, end, verbose=True)
        o["season"] = s
        o.to_parquet(CACHE / f"odds_{s}.parquet")
        frames.append(o)
        print(f"  odds {s}: {o.an_id.nunique()} games, books {o.book.value_counts().to_dict()}")
    if not args.current:
        all_odds = pd.concat(frames, ignore_index=True)
        allg, _ = data.load(range(ODDS_FIRST_SEASON, CURRENT_SEASON + 1))
        cw = odds.crosswalk(all_odds, allg)
        cw.to_csv(odds.CROSSWALK_FILE, index=False)
        print(f"crosswalk: {len(cw)} teams mapped")


if __name__ == "__main__":
    main()
