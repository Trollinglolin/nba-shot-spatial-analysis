"""NBA court geometry and project-wide constants.

Coordinates follow the NBA stats API convention: units are TENTHS OF A FOOT,
origin (0, 0) is the center of the rim, +y points toward half court and
+x points to the right when the court is drawn with the baseline at the bottom.
"""
from __future__ import annotations

UNITS_PER_FT = 10.0

# --- Court dimensions (tenths of feet) ---
RIM_HEIGHT_FT = 10.0
RIM_RADIUS = 7.5                 # 9" inner radius = 0.75 ft
BACKBOARD_Y = -12.5              # backboard face, 1.25 ft behind rim center
BACKBOARD_HALF_WIDTH = 30.0      # 6 ft wide
BASELINE_Y = -52.5               # rim center is 5.25 ft from baseline
HALF_COURT_Y = 417.5             # 47 ft from baseline
COURT_HALF_WIDTH = 250.0         # 50 ft wide

RESTRICTED_RADIUS = 40.0         # 4 ft restricted-area arc
PAINT_HALF_WIDTH = 80.0          # 16 ft wide lane
LANE_INNER_HALF_WIDTH = 60.0     # 12 ft inner lane marking
FREE_THROW_Y = 137.5             # 19 ft from baseline
FREE_THROW_RADIUS = 60.0         # 6 ft circle

THREE_PT_ARC_RADIUS = 237.5      # 23.75 ft
THREE_PT_CORNER_X = 220.0        # 22 ft in the corners
THREE_PT_CORNER_Y = 140.0        # corner straightaway ends ~14 ft from baseline-line

# --- Data filtering ---
HEAVE_DISTANCE_FT = 40.0         # shots beyond this are treated as heaves
MIN_ATTEMPTS_PLAYER = 300        # min attempts to rank a player
TRAIN_END_SEASON_START = 2023    # train on seasons starting < 2023, test on >= 2023

# --- Heatmap grid ---
GRID_BIN = 40.0                  # 4 ft bins
GRID_X_RANGE = (-250.0, 250.0)
GRID_Y_RANGE = (-50.0, 430.0)

ZONES = ["Restricted Area", "Paint (Non-RA)", "Mid-Range", "Corner 3", "Above-Break 3"]
AREAS = ["Right Baseline", "Right Wing", "Top of Key", "Left Wing", "Left Baseline"]

PATHS = {
    "data": "data",
    "models": "models",
    "reports": "reports",
    "charts": "reports/charts",
}
