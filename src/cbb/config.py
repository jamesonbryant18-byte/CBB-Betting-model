"""
Every setting lives here, once.

Two kinds of numbers, kept apart on purpose (the NFL model once froze a whole
settings object to disk and silently ran a stale value for weeks):

  * CHOICES  -- the betting rules and the season split. Set by hand, below.
  * FITTED   -- model parameters. Defaults live in `ModelParams`; the fit
                script writes ONLY the fields listed in `ModelParams.FITTED`
                to data/params.json, and `load_params()` refuses any other key.

Seasons are named by the year they END: 2027 is the 2026-27 season.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

# ─────────────────────────── paths ───────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
PICKS = ROOT / "picks"
OUTPUT = ROOT / "output"
RESEARCH = ROOT / "research"
PARAMS_FILE = DATA / "params.json"

# ─────────────────────────── seasons ───────────────────────────
CURRENT_SEASON = 2027
FIRST_SEASON = 2008          # earliest season loaded (2010 needs a prior season)

# The split every test uses. Nothing is fit on HOLDOUT; it is scored once.
SCREEN_SEASONS = tuple(range(2010, 2020))   # results only (no betting lines exist)
TUNE_SEASONS = tuple(range(2020, 2024))     # + consensus closing lines
HOLDOUT_SEASONS = tuple(range(2024, 2027))  # + FanDuel closing lines


def season_label(season: int) -> str:
    """2027 -> '2026-27'."""
    return f"{season - 1}-{str(season)[-2:]}"


# ─────────────────────────── conferences ───────────────────────────
# ESPN conference ids. "Power" = the four football power conferences plus the
# Big East (owner, 2026-11-09: "the big east should count").
POWER_CONFERENCES = {2: "ACC", 4: "Big East", 7: "Big Ten", 8: "Big 12", 23: "SEC"}

# ESPN tournament id of the NCAA tournament (First Four through the final).
NCAA_TOURNAMENT_ID = 22


# ─────────────────────────── betting rules (CHOICES) ───────────────────────────
@dataclass(frozen=True)
class BetRules:
    book: str = "fanduel"            # the only book he bets (locked)
    stake: float = 1.00              # flat $1 a bet (owner, 2026-11-09)

    # Which games can be bet: at least one Power-5 (incl. Big East) team, or
    # any AP top-25 team. Postseason: the NCAA tournament only.
    power_conferences: tuple = tuple(POWER_CONFERENCES)
    max_rank: int = 25

    # How many bets: Monday-Friday share 8 slots, Saturday-Sunday share 8.
    # First come, first served: a bet published earlier in the window keeps
    # its slot (owner, 2026-11-09, option a).
    weekday_max: int = 8
    weekend_max: int = 8

    # Carried from the NFL model, his rules:
    half_points_only: bool = True    # no spread bets on a whole number
    ml_max_underdog: int = 250       # no moneyline dogs longer than +250
    ml_max_favorite: int = -350      # no moneyline favorites shorter than -350

    # A bet needs this much expected value at FanDuel's price, measured with
    # the bet probability (model blended toward the market; see ModelParams).
    min_ev: float = 0.03


RULES = BetRules()


# ─────────────────────────── model parameters (FITTED) ───────────────────────────
@dataclass(frozen=True)
class ModelParams:
    # ── ratings (points of margin per game, opponent-adjusted) ──
    ridge_lambda: float = 3.0        # pull of each team toward its prior, in games of data
    half_life_days: float = 45.0     # a game this many days old counts half
    hfa: float = 3.0                 # home court, points; refit daily from data
    hfa_lambda: float = 200.0        # how firmly the daily HFA stays near `hfa`

    # ── preseason prior: last season's final rating, regressed ──
    carry_base: float = 0.45         # share of last season's rating kept, zero continuity
    carry_continuity: float = 0.35   # extra share kept per unit of returning production

    # ── market prior (ratings backed out of earlier closing spreads) ──
    market_lambda: float = 0.0       # pull toward the market ratings; 0 = off
    market_ridge_lambda: float = 1.0

    # ── player availability (players.py) ──
    avail_points: float = 0.0        # points per 1.0 of team production missing; 0 = off
    value_production_weight: float = 0.5   # player value: production share vs minutes share

    # ── margin -> probability ──
    sigma: float = 10.5              # spread of final margins around the projection
    sigma_per_poss: float = 0.0      # extra sigma per possession above average

    # ── bet probability = market + weight * (model - market) ──
    blend_weight: float = 1.0        # 1 = trust the model fully, 0 = trust the market

    FITTED = ("ridge_lambda", "half_life_days", "hfa", "carry_base",
              "carry_continuity", "market_lambda", "avail_points",
              "value_production_weight", "sigma", "sigma_per_poss", "blend_weight")


def load_params(path: Path = PARAMS_FILE) -> ModelParams:
    """Defaults overlaid with the fitted values on disk. Unknown keys are an error."""
    p = ModelParams()
    if not Path(path).exists():
        return p
    saved = json.loads(Path(path).read_text())
    values = saved.get("values", saved)
    bad = set(values) - set(ModelParams.FITTED)
    if bad:
        raise ValueError(f"{path} has keys that are not fitted parameters: {sorted(bad)}")
    return replace(p, **values)


def save_params(params: ModelParams, note: str, path: Path = PARAMS_FILE) -> None:
    values = {f: getattr(params, f) for f in ModelParams.FITTED}
    Path(path).write_text(json.dumps({"note": note, "values": values}, indent=2) + "\n")


def param_names() -> list[str]:
    return [f.name for f in fields(ModelParams)]
