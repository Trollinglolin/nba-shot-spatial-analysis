# NBA Shot Chart Spatial Analysis & xFG% Model

**Live demo:** https://trollinglolin.github.io/nba-shot-spatial-analysis/

End-to-end pipeline: ingest NBA shot coordinates -> engineer spatial features -> logistic-regression
**Expected FG% (xFG%)** model -> interactive Plotly shot charts -> summary report.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py                              # nba_api, 2014-15 to 2025-26, all teams
.venv/bin/python run_pipeline.py --seasons 2018-19 2024-25    # different range
.venv/bin/python run_pipeline.py --player "Stephen Curry"     # choose dashboard player
.venv/bin/python run_pipeline.py --source synthetic           # offline demo data
.venv/bin/python run_pipeline.py --source sportradar          # alternative source (needs a key in .env)
.venv/bin/python tests_smoke.py
```

Outputs: `reports/summary.md`, `reports/*.csv`, `reports/charts/index.html` (links to every player profile),
`reports/charts/dashboard.html` (player vs league), `models/xfg_model.joblib`, `models/metrics.json`.
HTML files load Plotly from a CDN, so open them with an internet connection.

## Layout

| File | Role |
|---|---|
| `constants.py` | Court geometry (tenths of feet, rim at origin), split year, grid settings |
| `ingest.py` | `nba_api` fetch (cached per team-season), normalization, heave filter, source selection, synthetic generator |
| `sportradar.py` | Optional Sportradar v8 client: schedule + play-by-play to shot rows (cached per game) |
| `features.py` | Distance, `atan2(y, x)` angle, zone and court-area labels, period / clutch features, target `shot_made_result`, chronological split |
| `model.py` | Spline-on-distance + one-hot logistic regression; accuracy, precision, recall, ROC-AUC, Brier, confusion matrix; per-player xFG% |
| `visualize.py` | Plotly court, FG%-vs-league heatmap with league xFG% contours, per-player pages (top 150 by attempts, `--max-players`), comparative dashboard |
| `report.py` | League xFG% by zone, best/worst shot makers and shot-quality tables |

## Data sources

**Default: `nba_api` `ShotChartDetail`** (`stats.nba.com`). For each season it makes one call per team (30 per season, `player_id=0`),
which returns every field-goal attempt of that team with `LOC_X`/`LOC_Y` (tenths of a foot, rim at the origin), shot type, period,
clock and made/missed. 12 seasons (2014-15 to 2025-26) is 360 calls, roughly 2.5M shots.

- **Cached and resumable:** each team-season is saved to `data/raw/nba_api/`. A call that keeps failing (timeout, rate limit) is
  retried with backoff, then skipped and listed at the end; rerun the same command to fetch only what is missing.
- **Timeouts / blocking:** `stats.nba.com` is slow and rejects some networks (cloud/VPN IPs). Use `--timeout 120` or `--proxy <url>`.
- **Orientation:** `LOC_X` is used as returned. Verified against the API's own `SHOT_ZONE_AREA`: `Left Side(L)` shots have `LOC_X` of about -177, so negative x is the shooter's left, and the charts are labelled that way. `--flip-x` is only an escape hatch.
- `--season-type Playoffs` pulls playoff shots instead of the regular season.
- The raw `nba_api` data has an end-of-quarter heave filter applied here (> 40 ft dropped).

**Alternatives kept in the repo**
- `--source sportradar`: Sportradar NBA API v8 (`sportradar.py`), key read from `.env` (`NBA_API` or `SPORTRADAR_API_KEY`).
  A sampled-games source (one call per game); the trial key only reached 2024-25 onward. Nothing the model uses is Sportradar-only;
  its extras (on-court lineups, assist links) are not features here.
- `--source csv --csv file.csv` (columns listed in `ingest.COLUMNS`) and `--source synthetic` (fictional players, offline demos only).

## Model

Features: distance (cubic spline basis), angle, angle from the center line, seconds left in the period, zone, 2PT/3PT, period group
(Q1-Q4, OT), last-2-minutes flag, and the **shot style** parsed from NBA's `ACTION_TYPE` (dunk / layup / hook / floater / jump shot, plus
driving / pullup / step back / fadeaway / cutting / putback ...).

Two layers, both logistic regression:
- **Validation (honest):** fit on seasons before 2023, scored on 2023 onward. Reported in `reports/summary.md` for three models:
  `location`, `location+style`, and a gradient-boosting benchmark that shows how much a linear model leaves on the table.
- **Final (used for every table and chart):** the same logistic models refit on all seasons with a per-season effect, so xFG% is calibrated to
  each season's league scoring level. Without this the validation model under-predicted post-2022 FG% by about 2.5 points, which inflated every
  player's "FG% over expected" (especially rim finishers). `models/xfg_models.joblib` holds `full` (with style, for player xFG%) and `loc`
  (location-only, for the chart contours).

What the results say: on held-out seasons ROC-AUC is about 0.64-0.65 for all models and Brier is 0.226-0.232 vs 0.249 for a constant guess.
Adding shot style improves calibration and Brier but not AUC, and gradient boosting adds only about +0.01 AUC, so the linear model is close to the
ceiling for data without defender or tracking information.

Player rankings use **empirical-Bayes shrinkage**: the raw gap (FG% - xFG%) is pulled toward the league mean in proportion to its sampling noise,
and only players with at least 300 test-season attempts are ranked, so a hot 300-shot sample does not outrank a proven high-volume shooter.

## Interpretation guide (non-technical)

- **xFG%** = the chance an *average* shooter makes a shot of this type from this spot. It measures shot *quality*.
- **FG% - xFG%** = shot *making* relative to expectation. Positive means the player beats the average for the
  shots they take.
- Shot charts: orange cells = player shoots better than the league in that 4-ft cell, blue = worse,
  gray = about average. Dotted lines connect spots with equal league-average xFG%.
- Small samples are noisy: a cell needs a few attempts, and player rankings use at least 150 attempts.

## Limitations

- No defender distance, shot clock, shooter identity, or play type, so "contested" and "open" shots look the same.
- Linear logit with no interaction terms; the gradient-boosting benchmark in the report shows the (small) cost.
- Player xFG% is the expectation for an average shooter, so it measures shot quality; shooter skill shows up in FG% minus xFG%.
- `location+style` xFG% includes how the shot was taken (e.g. a dunk), so it is not purely a location model; the chart contours use the location-only model.
- Top-N-scorers sampling is biased toward high-volume shooters.
