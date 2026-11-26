# NBA Shot Chart Spatial Analysis - Summary

> **Data:** Real NBA shot data from `nba_api` ShotChartDetail (2014-15 to 2025-26, Regular Season, all teams).

Test set: 656,228 shots from seasons >= 2023. Validation models were trained on earlier seasons only.

## Model performance (chronological holdout)

| Model | ROC-AUC | Brier | Accuracy | Mean xFG% | Actual FG% |
|---|---|---|---|---|---|
| location | 0.644 | 0.2316 | 0.619 | 44.7% | 47.2% |
| location+style | 0.641 | 0.2277 | 0.621 | 46.9% | 47.2% |
| gbm_benchmark | 0.651 | 0.2259 | 0.628 | 46.7% | 47.2% |

Naive Brier baseline (always predict the training base rate): 0.2493.
`location` uses court position and game context only; `location+style` adds the shot style from NBA's action type (dunk, layup, pullup, step-back...);
`gbm_benchmark` is gradient boosting on the same features, shown only to measure how much a linear logit leaves unexplained.

**Reading the numbers.** The logistic `location+style` model is the deliverable. Its precision/recall at a 0.5 cutoff are
0.672 / 0.384 (confusion matrix TN=288,670, FP=58,077,
FN=190,665, TP=118,816); recall is low because most shots have make-probability below 50%,
so ROC-AUC and Brier score are the primary measures. Mean xFG% sits below actual FG% because league shooting efficiency rose after the training
years; the final models below absorb that with a per-season effect.

**Final models** (used for every table and chart below) are refit on all seasons with a per-season effect, so xFG% matches each season's league
scoring level (in-sample ROC-AUC 0.646, mean xFG% 47.2% vs actual 47.2%). These are league-level models with a few dozen parameters
fit on millions of shots, so in-sample optimism is negligible.

## League xFG% by zone (test seasons)

| Zone | FGA | FG% | xFG% | Share of FGA |
|---|---|---|---|---|
| Restricted Area | 188,195 | 66.6% | 64.9% | 28.7% |
| Paint (Non-RA) | 130,905 | 44.3% | 44.0% | 19.9% |
| Mid-Range | 68,499 | 41.8% | 43.1% | 10.4% |
| Corner 3 | 78,504 | 38.6% | 39.7% | 12.0% |
| Above-Break 3 | 190,125 | 35.4% | 36.4% | 29.0% |

## Best shot makers (min 300 FGA in the test seasons)

Ranked by the **shrunk** gap (FG% - xFG%, pulled toward the league mean in proportion to sample noise), so small samples do not dominate.

| Player | FGA | FG% | xFG% | FG% - xFG% (raw) | FG% - xFG% (shrunk) |
|---|---|---|---|---|---|
| Nikola Jokić | 3,874 | 58.1% | 47.0% | +11.0% | +10.3% |
| Kevin Durant | 3,935 | 52.3% | 44.7% | +7.6% | +7.1% |
| T.J. McConnell | 1,663 | 53.9% | 46.1% | +7.7% | +6.7% |
| Jakob Poeltl | 1,266 | 65.4% | 57.8% | +7.6% | +6.4% |
| Bol Bol | 334 | 56.6% | 45.7% | +10.9% | +6.1% |
| Maxime Raynaud | 671 | 57.1% | 48.6% | +8.4% | +6.1% |
| Luke Kennard | 1,188 | 49.1% | 41.8% | +7.2% | +6.0% |
| Shai Gilgeous-Alexander | 4,463 | 53.5% | 47.2% | +6.3% | +5.9% |
| Payton Pritchard | 2,578 | 47.0% | 40.6% | +6.4% | +5.8% |
| Ty Jerome | 833 | 50.8% | 43.9% | +6.9% | +5.3% |

## Weakest shot makers

| Player | FGA | FG% | xFG% | FG% - xFG% (raw) | FG% - xFG% (shrunk) |
|---|---|---|---|---|---|
| Jonathan Mogbo | 398 | 45.7% | 57.8% | -12.1% | -7.2% |
| AJ Johnson | 386 | 36.0% | 48.0% | -12.0% | -7.0% |
| Clint Capela | 1,234 | 55.8% | 64.2% | -8.4% | -6.9% |
| Yves Missi | 785 | 54.6% | 63.5% | -8.8% | -6.6% |
| Bilal Coulibaly | 1,597 | 42.6% | 49.9% | -7.3% | -6.2% |
| KJ Simpson | 400 | 34.5% | 44.2% | -9.7% | -5.7% |
| Mouhamed Gueye | 484 | 43.6% | 52.4% | -8.8% | -5.6% |
| Jeremy Sochan | 1,368 | 47.8% | 54.4% | -6.6% | -5.5% |
| Kris Murray | 920 | 42.5% | 49.6% | -7.1% | -5.4% |
| Nae'Qwan Tomlin | 346 | 47.1% | 56.6% | -9.5% | -5.3% |

## Best shot quality (highest xFG% - where and how they shoot)

| Player | FGA | FG% | xFG% | FG% - xFG% (raw) | FG% - xFG% (shrunk) |
|---|---|---|---|---|---|
| Jericho Sims | 311 | 73.3% | 75.3% | -2.0% | -1.1% |
| Mitchell Robinson | 389 | 66.6% | 72.0% | -5.4% | -3.4% |
| Dereck Lively II | 505 | 72.5% | 71.7% | +0.8% | +0.6% |
| Rudy Gobert | 1,614 | 67.0% | 71.3% | -4.3% | -3.7% |
| Jaxson Hayes | 659 | 73.6% | 71.1% | +2.5% | +1.9% |
| Adem Bona | 412 | 64.6% | 68.4% | -3.8% | -2.4% |
| Robert Williams III | 354 | 68.9% | 68.2% | +0.7% | +0.5% |
| Luke Kornet | 755 | 66.8% | 67.9% | -1.2% | -0.8% |
| Daniel Gafford | 1,199 | 69.9% | 66.9% | +3.0% | +2.5% |
| Trayce Jackson-Davis | 757 | 63.0% | 66.7% | -3.7% | -2.8% |

## Poorest shot quality (lowest xFG%)

| Player | FGA | FG% | xFG% | FG% - xFG% (raw) | FG% - xFG% (shrunk) |
|---|---|---|---|---|---|
| AJ Green | 1,235 | 42.6% | 38.5% | +4.1% | +3.5% |
| Doug McDermott | 539 | 42.1% | 39.1% | +3.0% | +2.1% |
| Caleb Houstan | 422 | 41.2% | 39.2% | +2.0% | +1.3% |
| Garrison Mathews | 535 | 42.2% | 39.3% | +3.0% | +2.1% |
| Sam Hauser | 1,629 | 43.9% | 39.6% | +4.3% | +3.8% |
| Gabe Vincent | 682 | 38.0% | 39.9% | -1.9% | -1.3% |
| Lindy Waters III | 441 | 39.5% | 39.9% | -0.5% | -0.2% |
| Royce O'Neale | 1,721 | 41.7% | 40.0% | +1.7% | +1.5% |
| Nicolas Batum | 731 | 43.2% | 40.3% | +2.9% | +2.2% |
| Baylor Scheierman | 443 | 42.9% | 40.4% | +2.5% | +1.6% |
