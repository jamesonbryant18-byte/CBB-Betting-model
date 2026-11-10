"""
Morning-of-game advanced metrics for every season, cached for the metric tests.

    .venv/bin/python scripts/build_features.py
"""
import pandas as pd

from cbb import data, features
from cbb.config import CACHE, CURRENT_SEASON, SCREEN_SEASONS


def main():
    seasons = range(SCREEN_SEASONS[0], CURRENT_SEASON)
    games, _ = data.load(seasons)
    out = []
    for s in seasons:
        out.append(features.season_features(games, s))
        print(f"  features {s}: {len(out[-1])} games", flush=True)
    f = pd.concat(out, ignore_index=True)
    f.to_parquet(CACHE / "features.parquet")
    print(f"wrote {len(f)} rows")


if __name__ == "__main__":
    main()
