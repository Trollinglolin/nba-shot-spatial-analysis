"""Sportradar NBA v8 client: schedule + play-by-play -> shot rows.

Auth: key in the `x-api-key` header, read from a `.env` file (variable `NBA_API` or
`SPORTRADAR_API_KEY`) or the environment. The key is never logged or cached.

Coordinates (verified against the API's own `shot_distance`, mean error < 0.01 ft):
  inches on a 1128 x 600 court; baskets at (63, 300) and (1065, 300).
We convert each shot to the NBA-style frame used by the rest of the project: tenths of a
foot, rim at the origin, +y toward half court.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests

BASE = "https://api.sportradar.com/nba/{level}/v8/en/"
LEFT_BASKET, RIGHT_BASKET, BASKET_Y, MID_X = 63.0, 1065.0, 300.0, 564.0
SHOT_TYPES = {"twopointmade", "twopointmiss", "threepointmade", "threepointmiss"}


class QuotaError(RuntimeError):
    """Raised when the API refuses further calls (rate limit / trial quota)."""


def load_key(env_names=("SPORTRADAR_API_KEY", "NBA_API")) -> str:
    for n in env_names:
        if os.environ.get(n):
            return os.environ[n].strip()
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() in env_names and v.strip():
                        return v.strip().strip("\"'")
    raise RuntimeError("No Sportradar key: put NBA_API=<key> in a .env file or set SPORTRADAR_API_KEY.")


class Client:
    def __init__(self, level="trial", sleep=1.1, key=None):
        self.base = BASE.format(level=level)
        self.key, self.sleep, self.calls = key or load_key(), sleep, 0
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.key, "accept": "application/json"})

    def get(self, path, retries=3):
        for attempt in range(retries):
            time.sleep(self.sleep)                       # trial keys allow ~1 request/second
            r = self.session.get(self.base + path, timeout=45)
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise QuotaError(f"HTTP {r.status_code} from Sportradar (key/quota/access): {r.text[:150]}")
            r.raise_for_status()
        raise QuotaError("HTTP 429 persisted: rate limit or quota exhausted.")


def to_nba_frame(x_in, y_in):
    """Sportradar inches -> (loc_x, loc_y) in tenths of feet, rim at origin, +y toward half court.

    Both baskets are mapped with a 180-degree rotation so every shot lands in one half court.
    """
    if x_in < MID_X:
        dx, dy = x_in - LEFT_BASKET, y_in - BASKET_Y
    else:
        dx, dy = RIGHT_BASKET - x_in, BASKET_Y - y_in
    return round(-dy / 12 * 10), round(dx / 12 * 10)


def parse_clock(clock):
    """'11:48' -> (11, 48). Malformed or out-of-range clocks fall back to mid-period (6, 0)."""
    try:
        m, _, sec = str(clock).partition(":")
        m, sec = int(m), int(float(sec or 0))
        if 0 <= m <= 12 and 0 <= sec < 60:
            return m, sec
    except (ValueError, TypeError):
        pass
    return 6, 0


def parse_game(pbp: dict):
    """Extract field-goal attempts from a play-by-play payload."""
    rows = []
    date = pbp["scheduled"][:10]
    for per in pbp.get("periods", []):
        for e in per.get("events", []):
            if e.get("event_type") not in SHOT_TYPES or "location" not in e:
                continue
            fg = next((s for s in e.get("statistics", []) if s.get("type") == "fieldgoal"), None)
            loc = e["location"]
            if not fg or "player" not in fg or loc.get("coord_x") is None or loc.get("coord_y") is None:
                continue
            lx, ly = to_nba_frame(loc["coord_x"], loc["coord_y"])
            m, sec = parse_clock(e.get("clock"))
            rows.append({
                "game_id": pbp["id"], "game_date": date, "player_id": fg["player"]["id"],
                "player_name": fg["player"]["full_name"],
                "team_name": " ".join(x for x in (fg.get("team", {}).get("market"), fg.get("team", {}).get("name")) if x),
                "period": per["number"], "minutes_remaining": m, "seconds_remaining": sec,
                "shot_type": "3PT Field Goal" if e["event_type"].startswith("three") else "2PT Field Goal",
                "shot_distance": fg.get("shot_distance"), "loc_x": lx, "loc_y": ly,
                "shot_made_flag": int(e["event_type"].endswith("made")),
            })
    return rows


def fetch_shots(start_years, games_per_season=60, level="trial", cache_dir="data/raw/sportradar",
                sleep=1.1, seed=42, season_type="REG"):
    """Fetch shots for a seeded random sample of completed games per season (cached, resumable).

    Raising `games_per_season` later reuses everything already cached.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    client, out, stopped = None, [], False
    for year in start_years:
        sf = cache / f"schedule_{year}_{season_type}.json"
        if sf.exists():
            games = json.loads(sf.read_text())
        else:
            client = client or Client(level, sleep)
            sched = client.get(f"games/{year}/{season_type}/schedule.json")
            games = sorted(({"id": g["id"], "scheduled": g["scheduled"]} for g in sched["games"]
                            if g["status"] in ("closed", "complete")), key=lambda g: g["id"])
            sf.write_text(json.dumps(games))
        order = random.Random(f"{seed}-{year}").sample(games, len(games))[:games_per_season]
        season = f"{year}-{str(year + 1)[2:]}"
        got = 0
        for g in order:
            gf = cache / f"shots_{g['id']}.json"
            if gf.exists():
                rows = json.loads(gf.read_text())
            elif stopped:
                continue
            else:
                client = client or Client(level, sleep)
                try:
                    rows = parse_game(client.get(f"games/{g['id']}/pbp.json"))
                except QuotaError as exc:
                    print(f"STOP: {exc}\n  continuing with {sum(1 for _ in cache.glob('shots_*.json'))} cached games")
                    stopped = True
                    continue
                gf.write_text(json.dumps(rows))
            for r in rows:
                r["season"] = season
            out.extend(rows)
            got += 1
        print(f"{season}: {got}/{len(order)} games loaded")
    if client:
        print(f"API calls made this run: {client.calls}")
    return out
