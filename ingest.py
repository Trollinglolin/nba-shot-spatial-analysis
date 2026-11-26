"""Data ingestion: pull shot logs from nba_api (or Sportradar / synthetic data), normalize, filter.

Usage:
    python ingest.py --source nba_api --seasons 2014-15 2025-26   # default; 30 calls per season, cached
    python ingest.py --source sportradar --games-per-season 120   # needs a Sportradar key in .env
    python ingest.py --source synthetic                           # offline demo data
Output: data/shots.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import constants as C
from features import is_three

COLUMNS = ["game_id", "game_date", "season", "player_id", "player_name", "team_name",
           "period", "minutes_remaining", "seconds_remaining", "action_type", "shot_type",
           "shot_distance", "loc_x", "loc_y", "shot_made_flag"]


# ---------------------------------------------------------------- nba_api
def _season_list(first: str, last: str):
    a, b = int(first[:4]), int(last[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(a, b + 1)]


def fetch_nba_api(seasons, season_type="Regular Season", cache_dir="data/raw/nba_api",
                  pause=0.8, timeout=60, retries=5, proxy=None):
    """Fetch every shot of every team for each season via ShotChartDetail.

    One call per team-season (30 per season) with player_id=0 returns all of that team's
    field-goal attempts with LOC_X/LOC_Y. Each response is cached as CSV, so the run is
    resumable: if a call keeps failing it is skipped and reported, and a rerun only retries
    the missing ones.
    """
    from nba_api.stats.endpoints import shotchartdetail
    from nba_api.stats.static import teams

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = season_type.replace(" ", "")
    frames, failed = [], []
    team_list = teams.get_teams()
    for season in seasons:
        for t in team_list:
            f = cache / f"{season}_{tag}_{t['id']}.csv"
            if f.exists():
                frames.append(pd.read_csv(f, dtype={"GAME_ID": str}))
                continue
            for attempt in range(retries):
                try:
                    d = shotchartdetail.ShotChartDetail(
                        team_id=t["id"], player_id=0, season_nullable=season,
                        season_type_all_star=season_type, context_measure_simple="FGA",
                        timeout=timeout, proxy=proxy,
                    ).get_data_frames()[0]
                    d["season"] = season
                    d.to_csv(f, index=False)
                    frames.append(d)
                    print(f"  {season} {t['abbreviation']}: {len(d):,} shots", flush=True)
                    break
                except Exception as exc:  # timeouts, rate limits, malformed responses
                    wait = 5 * 2 ** attempt
                    print(f"  retry {attempt + 1}/{retries} {season} {t['abbreviation']}: "
                          f"{type(exc).__name__} (waiting {wait}s)", flush=True)
                    time.sleep(wait)
            else:
                failed.append(f"{season} {t['abbreviation']}")
            time.sleep(pause)
    if failed:
        print(f"WARNING: {len(failed)} team-seasons failed (rerun to retry): {', '.join(failed[:20])}")
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("No data fetched from nba_api (stats.nba.com unreachable or blocked?).")
    return pd.concat(frames, ignore_index=True)


def normalize_nba_api(raw: pd.DataFrame, flip_x: bool = False) -> pd.DataFrame:
    r = raw.rename(columns=str.lower)
    out = pd.DataFrame({
        "game_id": r["game_id"], "game_date": pd.to_datetime(r["game_date"].astype(str), format="%Y%m%d"),
        "season": r["season"], "player_id": r["player_id"], "player_name": r["player_name"],
        "team_name": r["team_name"], "period": r["period"],
        "minutes_remaining": r["minutes_remaining"], "seconds_remaining": r["seconds_remaining"],
        "action_type": r["action_type"], "shot_type": r["shot_type"], "shot_distance": r["shot_distance"],
        "loc_x": -r["loc_x"] if flip_x else r["loc_x"], "loc_y": r["loc_y"],
        "shot_made_flag": r["shot_made_flag"],
    })
    return out[COLUMNS]


def fetch_sportradar(first, last, games_per_season, level):
    """Optional alternative source (sampled games; see sportradar.py)."""
    import sportradar
    rows = sportradar.fetch_shots(range(int(first[:4]), int(last[:4]) + 1), games_per_season, level)
    if not rows:
        raise RuntimeError("Sportradar returned no shots (check key / quota).")
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.reindex(columns=COLUMNS)


# ---------------------------------------------------------------- synthetic
ARCHETYPES = {  # mix over [rim, paint, mid, corner3, above-break3]
    "Big": [0.55, 0.20, 0.15, 0.03, 0.07],
    "Wing": [0.25, 0.12, 0.23, 0.12, 0.28],
    "Guard": [0.18, 0.10, 0.22, 0.08, 0.42],
}


def _sample_location(zone_idx, rng):
    n = len(zone_idx)
    x, y = np.zeros(n), np.zeros(n)
    for z in range(5):
        m = zone_idx == z
        k = int(m.sum())
        if not k:
            continue
        if z == 0:    # rim
            r = np.minimum(np.abs(rng.normal(0, 22, k)), 39.0)
            th = rng.uniform(0, np.pi, k)
            x[m], y[m] = r * np.cos(th), r * np.sin(th)
        elif z == 1:  # paint
            x[m] = rng.uniform(-78, 78, k)
            y[m] = rng.uniform(45, 135, k)
        elif z == 2:  # mid-range
            r = rng.uniform(100, 225, k)
            th = rng.uniform(np.radians(10), np.radians(170), k)
            x[m], y[m] = r * np.cos(th), r * np.sin(th)
        elif z == 3:  # corner 3
            x[m] = rng.choice([-1, 1], k) * rng.uniform(222, 245, k)
            y[m] = rng.uniform(-45, 138, k)
        else:         # above-break 3
            r = 242 + rng.exponential(18, k)
            r = np.minimum(r, 330)
            th = rng.uniform(np.radians(22), np.radians(158), k)
            x[m], y[m] = np.clip(r * np.cos(th), -248, 248), r * np.sin(th)
    return x, y


def _base_prob(d_ft, three, corner):
    p = np.interp(d_ft, [0, 4, 8, 14, 20, 23, 25, 28, 35, 50],
                  [0.68, 0.60, 0.44, 0.41, 0.40, 0.37, 0.355, 0.33, 0.25, 0.10])
    return np.clip(p + np.where(corner, 0.025, 0.0), 0.02, 0.95)


def generate_synthetic(n_players=80, seasons=range(2019, 2025), seed=7):
    """Synthetic shots with realistic geometry and a known data-generating process.

    Players are fictional ("Synthetic Guard 01"). Skill offsets differ by zone group so
    xFG% vs FG% differences are recoverable. NOT real NBA data.
    """
    rng = np.random.default_rng(seed)
    kinds = rng.choice(list(ARCHETYPES), n_players, p=[0.3, 0.3, 0.4])
    players = []
    for i, k in enumerate(kinds):
        players.append({
            "player_id": 1000 + i, "player_name": f"Synthetic {k} {i + 1:02d}", "kind": k,
            "team_name": f"Team {i % 30 + 1:02d}",
            "skill_rim": rng.normal(0, 0.18), "skill_mid": rng.normal(0, 0.18),
            "skill_three": rng.normal(0, 0.18),
            "volume": rng.uniform(0.6, 1.4),
        })
    rows = []
    for p in players:
        for s in seasons:
            n = int(np.clip(rng.normal(520, 170) * p["volume"], 120, 1100))
            mix = np.array(ARCHETYPES[p["kind"]]) + rng.normal(0, 0.02, 5)
            mix = np.clip(mix, 0.01, None)
            mix[3:] *= 1 + 0.04 * (s - 2019)          # league-wide drift toward 3s
            mix /= mix.sum()
            zi = rng.choice(5, n, p=mix)
            x, y = _sample_location(zi, rng)
            heave = rng.random(n) < 0.01              # end-of-period heaves
            y[heave] = rng.uniform(300, 417, heave.sum())
            x[heave] = rng.uniform(-200, 200, heave.sum())
            d_ft = np.hypot(x, y) / 10
            three = is_three(x, y)
            corner = three & (np.abs(x) >= 220) & (y <= 140)
            logit = np.log(_base_prob(d_ft, three, corner) / (1 - _base_prob(d_ft, three, corner)))
            logit += np.where(d_ft < 5, p["skill_rim"], np.where(three, p["skill_three"], p["skill_mid"]))
            logit += rng.normal(0, 0.15, n)           # unmodeled context (defender, etc.)
            made = rng.random(n) < 1 / (1 + np.exp(-logit))
            made[heave] = rng.random(heave.sum()) < 0.03
            period = rng.choice([1, 2, 3, 4, 5], n, p=[0.245, 0.245, 0.245, 0.245, 0.02])
            mins = np.where(period == 5, rng.integers(0, 5, n), rng.integers(0, 12, n))
            start = pd.Timestamp(f"{s}-10-22")
            dates = start + pd.to_timedelta(rng.integers(0, 170, n), unit="D")
            rows.append(pd.DataFrame({
                "game_id": [f"{s}{p['player_id']}{g:03d}" for g in rng.integers(0, 82, n)],
                "game_date": dates, "season": f"{s}-{str(s + 1)[2:]}",
                "player_id": p["player_id"], "player_name": p["player_name"],
                "team_name": p["team_name"], "period": period,
                "minutes_remaining": mins, "seconds_remaining": rng.integers(0, 60, n),
                "shot_type": np.where(three, "3PT Field Goal", "2PT Field Goal"),
                "shot_distance": np.round(d_ft).astype(int),
                "loc_x": np.round(x).astype(int), "loc_y": np.round(y).astype(int),
                "shot_made_flag": made.astype(int),
            }))
    return pd.concat(rows, ignore_index=True).reindex(columns=COLUMNS)


# ---------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing rows, off-court artifacts and end-of-period heaves (> 40 ft)."""
    n0 = len(df)
    if "action_type" not in df:                     # sportradar / synthetic have no ACTION_TYPE
        df = df.assign(action_type="Unknown")
    df = df.dropna(subset=["loc_x", "loc_y", "shot_made_flag", "period"]).copy()
    df["shot_distance"] = np.hypot(df["loc_x"], df["loc_y"]) / C.UNITS_PER_FT
    df = df[(df["shot_distance"] <= C.HEAVE_DISTANCE_FT)
            & (df["loc_y"] <= C.HALF_COURT_Y) & (df["loc_y"] >= C.BASELINE_Y)
            & (df["loc_x"].abs() <= C.COURT_HALF_WIDTH)]
    print(f"clean: {n0:,} -> {len(df):,} rows ({n0 - len(df):,} dropped)")
    return df.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["nba_api", "sportradar", "synthetic", "csv"], default="nba_api")
    ap.add_argument("--seasons", nargs=2, default=["2014-15", "2025-26"], metavar=("FIRST", "LAST"))
    ap.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    ap.add_argument("--timeout", type=int, default=60, help="nba_api request timeout (s)")
    ap.add_argument("--proxy", help="optional proxy URL for nba_api")
    ap.add_argument("--flip-x", action="store_true",
                    help="negate LOC_X (use if the charts come out mirrored left/right)")
    ap.add_argument("--games-per-season", type=int, default=120, help="sportradar: games sampled per season (1 API call each)")
    ap.add_argument("--level", default="trial", choices=["trial", "production"], help="sportradar access level")
    ap.add_argument("--csv", help="existing normalized CSV (source=csv)")
    ap.add_argument("--out", default=f"{C.PATHS['data']}/shots.csv")
    a = ap.parse_args(argv)

    if a.source == "nba_api":
        raw = fetch_nba_api(_season_list(*a.seasons), a.season_type, timeout=a.timeout, proxy=a.proxy)
        df = normalize_nba_api(raw, a.flip_x)
    elif a.source == "sportradar":
        df = fetch_sportradar(a.seasons[0], a.seasons[1], a.games_per_season, a.level)
    elif a.source == "csv":
        df = pd.read_csv(a.csv, parse_dates=["game_date"])
    else:
        df = generate_synthetic(seasons=range(int(a.seasons[0][:4]), int(a.seasons[1][:4]) + 1))
    df = clean(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({a.source}): {len(df):,} shots, seasons {df['season'].min()} to {df['season'].max()}")
    return df


if __name__ == "__main__":
    main()
