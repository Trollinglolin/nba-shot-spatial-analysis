"""Feature engineering: spatial features, zones and the binary target."""
from __future__ import annotations

import numpy as np
import pandas as pd

import constants as C

NUMERIC_FEATURES = ["distance_ft", "angle_deg", "angle_from_center", "seconds_left_in_period"]
LOCATION_CATEGORICAL = ["zone", "shot_type", "period_group", "clutch"]
STYLE_CATEGORICAL = ["base_type", "modifier"]          # from nba_api ACTION_TYPE (dunk, layup, pullup, ...)
CATEGORICAL_FEATURES = LOCATION_CATEGORICAL + STYLE_CATEGORICAL
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "shot_made_result"

_BASE = [("Dunk", "Dunk"), ("Layup", "Layup"), ("Finger Roll", "Layup"), ("Hook", "Hook"),
         ("Floating", "Floater"), ("Jump", "Jump"), ("Fadeaway", "Jump"), ("shot", "Jump")]
_MODS = [("Alley Oop", "Alley Oop"), ("Putback", "Putback"), ("Tip", "Tip"), ("Cutting", "Cutting"),
         ("Driving", "Driving"), ("Running", "Running"), ("Step Back", "Step Back"),
         ("Pull-Up", "Pullup"), ("Pullup", "Pullup"), ("Turnaround", "Turnaround"),
         ("Fadeaway", "Fadeaway"), ("Reverse", "Reverse")]


def shot_style(action_type: str):
    """'Driving Floating Jump Shot' -> ('Floater', 'Driving'). Unknown/missing -> ('Unknown', 'Unknown')."""
    if not isinstance(action_type, str) or not action_type or action_type == "Unknown":
        return "Unknown", "Unknown"
    base = next((b for k, b in _BASE if k.lower() in action_type.lower()), "Other")
    mod = next((m for k, m in _MODS if k.lower() in action_type.lower()), "Standard")
    return base, mod


def is_three(x, y):
    """Geometric 3PT test (works on arrays)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    return (r >= C.THREE_PT_ARC_RADIUS) | (np.abs(x) >= C.THREE_PT_CORNER_X)


def assign_zone(x, y, three):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    three = np.asarray(three, dtype=bool)
    corner = three & (np.abs(x) >= C.THREE_PT_CORNER_X) & (y <= C.THREE_PT_CORNER_Y)
    paint = (np.abs(x) <= C.PAINT_HALF_WIDTH) & (y <= C.FREE_THROW_Y)
    return np.select(
        [r <= C.RESTRICTED_RADIUS, corner, three, paint],
        ["Restricted Area", "Corner 3", "Above-Break 3", "Paint (Non-RA)"],
        default="Mid-Range",
    )


def assign_area(angle_deg):
    """Quadrant-style label from the angle (0 deg = right baseline, 90 = top of key)."""
    bins = [-1, 36, 72, 108, 144, 181]
    return pd.cut(angle_deg, bins=bins, labels=C.AREAS).astype(str)


def spatial_frame(x, y):
    """Spatial features for raw coordinates; used for training and for grid predictions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dist_ft = np.hypot(x, y) / C.UNITS_PER_FT                 # Pythagorean distance
    angle = np.degrees(np.arctan2(y, x))                      # atan2(y, x) per the plan
    three = is_three(x, y)
    return pd.DataFrame({
        "distance_ft": dist_ft,
        "angle_deg": angle,
        "angle_from_center": np.abs(angle - 90.0),            # 0 = straight on
        "zone": assign_zone(x, y, three),
        "area": assign_area(angle),
        "shot_type": np.where(three, "3PT Field Goal", "2PT Field Goal"),
    })


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sp = spatial_frame(df["loc_x"], df["loc_y"])
    for col in sp.columns:
        df[col] = sp[col].values
    # keep API-provided shot_type if present, otherwise geometric one is used above
    if "action_type" not in df:
        df["action_type"] = "Unknown"
    styles = {a: shot_style(a) for a in df["action_type"].unique()}
    df["base_type"] = df["action_type"].map(lambda a: styles[a][0])
    df["modifier"] = df["action_type"].map(lambda a: styles[a][1])
    df["period_group"] = np.where(df["period"] <= 4, "Q" + df["period"].astype(str), "OT")
    df["seconds_left_in_period"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
    df["clutch"] = np.where(df["seconds_left_in_period"] <= 120, "last2min", "other")
    df[TARGET] = df["shot_made_flag"].astype(int)
    df["season_start"] = df["season"].str[:4].astype(int)
    return df


def season_split(df: pd.DataFrame):
    """Chronological split: train on seasons starting before 2023, test on 2023+."""
    train = df[df["season_start"] < C.TRAIN_END_SEASON_START]
    test = df[df["season_start"] >= C.TRAIN_END_SEASON_START]
    return train, test
