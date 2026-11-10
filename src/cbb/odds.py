"""
Betting lines from Action Network (free, no key; needs a browser User-Agent).

  past dates   web/v1 scoreboard: the last number each book had for each game.
               FanDuel, DraftKings and the opener from 2023-24 on; only a
               consensus number for 2019-20 to 2022-23; nothing earlier.
  today        web/v2 scoreboard: live pregame markets per book.

One convention everywhere (the NFL model once printed the wrong side of every
game because two modules disagreed about the sign):

    home_spread  = the number on a HOME ticket.  LSU +8.5 at home -> +8.5.
    market margin (what the line says home wins by) = -home_spread.

Action Network's team ids are mapped to ESPN's by `crosswalk()`, which learns
the mapping from games both sources list on the same day. A neutral-site game
can list the two teams in the opposite order; `attach()` flips those lines so
"home" always means ESPN's home team.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE, DATA
from .pricing import american_to_prob

BOOKS = {15: "consensus", 30: "open", 68: "draftkings", 69: "fanduel"}
URL = ("https://api.actionnetwork.com/web/{v}/scoreboard/ncaab?period=game"
       "&bookIds={books}&division=D1&date={date:%Y%m%d}")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
CROSSWALK_FILE = DATA / "team_crosswalk.csv"
V2_HISTORY_FROM = pd.Timestamp("2025-06-01")     # v2 keeps past markets only this recently

COLUMNS = ["an_id", "start", "an_status", "an_home", "an_away", "an_home_name",
           "an_away_name", "an_home_rank", "an_away_rank", "book", "home_spread",
           "home_spread_price", "away_spread_price", "home_ml", "away_ml", "total",
           "captured"]


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


# ─────────────────────────── fetch ───────────────────────────

def _raw_day(version: str, date: pd.Timestamp, refresh: bool) -> tuple[dict, bool]:
    """(raw scoreboard, came from cache). Cached once every game is final."""
    path = CACHE / "odds" / f"{date:%Y%m%d}{'' if version == 'v1' else '_v2'}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text()), True
    d = _get(URL.format(v=version, books=",".join(map(str, BOOKS)), date=date))
    games = d.get("games", [])
    if games and all(g.get("status") in ("complete", "closed", "cancelled", "postponed") for g in games):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"games": games}))
    return d, False


def history_day(date: pd.Timestamp, refresh: bool = False) -> tuple[pd.DataFrame, bool]:
    """
    Every stored line for a past ET date, and whether it all came from cache.

    Action Network's v1 feed keeps FanDuel's last number from 2023-24 on, but
    for November 2025 to mid-February 2026 it kept only the consensus and the
    opener; its v2 feed still has FanDuel for those dates. So: v1 first, and
    v2 for any book v1 lacks that day.
    """
    d1, c1 = _raw_day("v1", date, refresh)
    rows = parse_v1(d1)
    cached = c1
    if len(d1.get("games", [])) and not (rows.book == "fanduel").any() and date >= V2_HISTORY_FROM:
        d2, c2 = _raw_day("v2", date, refresh)
        v2 = parse_v2(d2)
        rows = pd.concat([rows, v2[~v2.book.isin(rows.book.unique())]], ignore_index=True)
        cached = cached and c2
    return rows, cached


def live_day(date: pd.Timestamp) -> pd.DataFrame:
    """Today's (or tomorrow's) pregame lines, every book in BOOKS."""
    d = _get(URL.format(v="v2", books=",".join(map(str, BOOKS)), date=date))
    return parse_v2(d, captured=pd.Timestamp.now(tz="UTC"))


# ─────────────────────────── parse ───────────────────────────

def _teams(e: dict) -> tuple:
    """(home id, away id, home name, away name, home AP rank, away AP rank)."""
    t = {x["id"]: x for x in e.get("teams", [])}
    h, a = t.get(e["home_team_id"], {}), t.get(e["away_team_id"], {})
    ranks = {r.get("team_id"): r.get("rank") for r in e.get("ranks") or []
             if r.get("poll") == "AP25"}
    return (e["home_team_id"], e["away_team_id"], h.get("full_name", ""),
            a.get("full_name", ""), _num(ranks.get(e["home_team_id"])),
            _num(ranks.get(e["away_team_id"])))


def parse_v1(d: dict) -> pd.DataFrame:
    rows = []
    for e in d.get("games", []):
        hid, aid, hn, an, hr, ar = _teams(e)
        for o in e.get("odds") or []:
            book = BOOKS.get(o.get("book_id"))
            if book is None or o.get("type", "game") != "game":
                continue
            rows.append([e["id"], e["start_time"], e.get("status"), hid, aid, hn, an, hr, ar, book,
                         _num(o.get("spread_home")), _num(o.get("spread_home_line")),
                         _num(o.get("spread_away_line")), _num(o.get("ml_home")),
                         _num(o.get("ml_away")), _num(o.get("total")), o.get("inserted")])
    return _frame(rows)


def parse_v2(d: dict, captured=None) -> pd.DataFrame:
    rows = []
    for e in d.get("games", []):
        hid, aid, hn, an, hr, ar = _teams(e)
        for bid, mk in (e.get("markets") or {}).items():
            book = BOOKS.get(int(bid))
            ev = (mk or {}).get("event") or {}
            if book is None:
                continue
            sp = _main_market(ev.get("spread", []), "home", "away", opposite=True)
            ml = _main_market(ev.get("moneyline", []), "home", "away")
            tot = _main_market(ev.get("total", []), "over", "under", opposite=False)
            if not sp and not ml:
                continue
            rows.append([e["id"], e["start_time"], e.get("status"), hid, aid, hn, an, hr, ar, book,
                         _num(sp.get("home", {}).get("value")), _num(sp.get("home", {}).get("odds")),
                         _num(sp.get("away", {}).get("odds")), _num(ml.get("home", {}).get("odds")),
                         _num(ml.get("away", {}).get("odds")),
                         _num(tot.get("over", {}).get("value")), captured])
    return _frame(rows)


def _main_market(outcomes: list, a: str, b: str, opposite: bool | None = None) -> dict:
    """
    The main line among a book's outcomes, as {side: outcome}.

    v2 lists alternate lines next to the main one (FanDuel: Virginia Tech +1.5
    at -112 is the line; -1.5 at +120 and +4.5 at -200 are alternates). A main
    market is one market_id holding exactly one outcome per side, at mirrored
    numbers (spread: +x/-x; total: the same x), and of those the one priced
    closest to even. Anything else returns {}.
    """
    by_market: dict = {}
    for o in outcomes:
        if not o.get("is_live"):
            by_market.setdefault(o.get("market_id"), []).append(o)
    best, best_gap = {}, np.inf
    for group in by_market.values():
        sides = {o["side"]: o for o in group}
        if len(group) != 2 or set(sides) != {a, b}:
            continue
        va, vb = _num(sides[a].get("value")), _num(sides[b].get("value"))
        if opposite is True and not np.isclose(va, -vb):
            continue
        if opposite is False and not np.isclose(va, vb):
            continue
        pa, pb = american_to_prob(_num(sides[a].get("odds"))), american_to_prob(_num(sides[b].get("odds")))
        gap = abs(pa - pb) if opposite is not None else 0.0      # moneyline: one market, no balance test
        if np.isfinite(gap) and gap < best_gap:
            best, best_gap = sides, gap
    return best


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["captured"] = pd.to_datetime(df["captured"], utc=True, errors="coerce", format="ISO8601")
    df["date"] = df["start"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return df


def history(start: pd.Timestamp, end: pd.Timestamp, pause: float = 0.25,
            verbose: bool = False) -> pd.DataFrame:
    """Every book row for every game between two ET dates (inclusive)."""
    frames = []
    for day in pd.date_range(start, end):
        cached = False
        try:
            rows, cached = history_day(day)
            frames.append(rows)
        except Exception as ex:                            # noqa: BLE001
            print(f"  [odds] {day:%Y-%m-%d} failed: {ex}")
        if not cached:
            time.sleep(pause)
        if verbose and day.day == 1:
            print(f"  [odds] through {day:%Y-%m-%d}", flush=True)
    if not frames:
        return _frame([])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["an_id", "book"], keep="last").reset_index(drop=True)


# ─────────────────────────── team crosswalk ───────────────────────────

def _norm(name: str) -> str:
    s = name.lower().replace("&", " and ").replace("st.", "state")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def crosswalk(odds: pd.DataFrame, games: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """
    Map Action Network team ids to ESPN team ids from games both list.

    Seed: same ET date, and ESPN's "location + nickname" equals Action
    Network's full name for both teams. Then grow: once one team of a game is
    mapped, its opponent that day reveals the other id. Each AN id keeps the
    ESPN id it agreed with most often, if that is at least `min_games` games and
    80% of its votes.
    """
    an = odds.drop_duplicates("an_id")[["an_id", "date", "an_home", "an_away",
                                        "an_home_name", "an_away_name"]]
    g = games[["date", "home_id", "away_id"]]
    by_day = defaultdict(dict)                 # date -> espn team -> opponent
    for r in g.itertuples(index=False):
        by_day[r.date][r.home_id] = r.away_id
        by_day[r.date][r.away_id] = r.home_id

    names = {}
    if "home_full" in games:
        for col_id, col_name in (("home_id", "home_full"), ("away_id", "away_full")):
            names.update(dict(zip(games[col_id], games[col_name].map(_norm))))
    by_name = defaultdict(set)
    for tid, n in names.items():
        by_name[n].add(tid)

    votes = defaultdict(Counter)
    for r in an.itertuples(index=False):
        h = by_name.get(_norm(r.an_home_name), set())
        a = by_name.get(_norm(r.an_away_name), set())
        day = by_day.get(r.date, {})
        for x in h:
            if day.get(x) in a:
                votes[r.an_home][x] += 1
                votes[r.an_away][day[x]] += 1

    mapping = {}
    for _ in range(6):                          # grow from the seed
        mapping = {k: c.most_common(1)[0][0] for k, c in votes.items()
                   if c.most_common(1)[0][1] >= min_games
                   and c.most_common(1)[0][1] >= 0.8 * sum(c.values())}
        grown = defaultdict(Counter)
        for r in an.itertuples(index=False):
            day = by_day.get(r.date, {})
            for known, other in ((r.an_home, r.an_away), (r.an_away, r.an_home)):
                e = mapping.get(known)
                if e is not None and e in day:
                    grown[other][day[e]] += 1
        if all(grown[k] == votes[k] for k in set(grown) | set(votes)):
            break
        votes = grown
    names_an = dict(zip(an.an_home, an.an_home_name)) | dict(zip(an.an_away, an.an_away_name))
    out = pd.DataFrame([{"an_id": k, "espn_id": v, "an_name": names_an.get(k, ""),
                         "espn_name": names.get(v, ""), "games": votes[k][v]}
                        for k, v in mapping.items()])
    return out.sort_values("an_name").reset_index(drop=True)


def load_crosswalk() -> dict:
    cw = pd.read_csv(CROSSWALK_FILE)
    return dict(zip(cw.an_id, cw.espn_id))


# ─────────────────────────── attach to games ───────────────────────────

LINE_COLS = ["home_spread", "home_spread_price", "away_spread_price", "home_ml",
             "away_ml", "total", "captured", "start", "an_home_rank", "an_away_rank"]


def attach(games: pd.DataFrame, odds: pd.DataFrame, book: str,
           cw: dict | None = None, prefix: str = "") -> pd.DataFrame:
    """
    Add one book's line to each ESPN game, oriented to ESPN's home team.

    Matched on ET date and the (unordered) pair of teams. When Action Network
    lists the teams the other way round, the spread is negated and the two
    prices and moneylines swap.
    """
    cw = load_crosswalk() if cw is None else cw
    o = odds[odds.book == book].copy()
    o["e_home"] = o.an_home.map(cw)
    o["e_away"] = o.an_away.map(cw)
    o = o.dropna(subset=["e_home", "e_away"])
    o[["e_home", "e_away"]] = o[["e_home", "e_away"]].astype("int64")

    straight = o.rename(columns={"e_home": "home_id", "e_away": "away_id"})
    flipped = o.rename(columns={"e_home": "away_id", "e_away": "home_id"}).copy()
    flipped["home_spread"] = -flipped["home_spread"]
    flipped[["home_spread_price", "away_spread_price"]] = flipped[["away_spread_price", "home_spread_price"]].values
    flipped[["home_ml", "away_ml"]] = flipped[["away_ml", "home_ml"]].values
    flipped[["an_home_rank", "an_away_rank"]] = flipped[["an_away_rank", "an_home_rank"]].values
    both = pd.concat([straight, flipped], ignore_index=True)
    both = both[["date", "home_id", "away_id"] + LINE_COLS].drop_duplicates(
        ["date", "home_id", "away_id"])
    both = both.rename(columns={c: f"{prefix}{c}" for c in LINE_COLS})
    return games.merge(both, on=["date", "home_id", "away_id"], how="left")


# ─────────────────────────── the closing-lines table ───────────────────────────

# Which book is "the close" for a game: FanDuel when it has the game (what he
# bets), else the consensus number (the only one stored before 2023-24), else
# DraftKings. The opener comes from Action Network's opening-line book.
CLOSE_PREFERENCE = ("fanduel", "consensus", "draftkings")


def closing_lines(games: pd.DataFrame, seasons) -> pd.DataFrame:
    """One row per game with a line: the close (by CLOSE_PREFERENCE) and the opener."""
    frames = []
    for s in seasons:
        path = CACHE / f"odds_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["game_id"])
    if not CROSSWALK_FILE.exists():
        print("  [odds] no team crosswalk yet (run scripts/fetch.py); no lines attached")
        return pd.DataFrame(columns=["game_id"])
    o = pd.concat(frames, ignore_index=True)
    cw = load_crosswalk()
    g = games[["game_id", "date", "home_id", "away_id", "tip"]]
    out = g[["game_id"]].copy()
    picked = None
    for book in CLOSE_PREFERENCE:
        b = attach(g, o, book, cw)
        b = b[b.home_spread.notna() | b.home_ml.notna()].copy()
        b["close_book"] = book
        if picked is None:
            picked = b
        else:
            picked = pd.concat([picked, b[~b.game_id.isin(picked.game_id)]], ignore_index=True)
    picked = picked.rename(columns={c: f"close_{c}" for c in LINE_COLS})
    picked["close_after_tip"] = picked.close_captured > picked.tip
    out = out.merge(picked.drop(columns=["date", "home_id", "away_id", "tip"]), on="game_id", how="left")
    op = attach(g, o, "open", cw, prefix="open_")[["game_id", "open_home_spread", "open_home_ml", "open_away_ml"]]
    out = out.merge(op, on="game_id", how="left")
    return out[out.close_book.notna()].reset_index(drop=True)
