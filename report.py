"""Summary statistics and Markdown/CSV report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import constants as C
import features as F
import model as M
import visualize as V


def _md(df: pd.DataFrame, fmt: dict) -> str:
    d = df.copy()
    for c, f in fmt.items():
        d[c] = d[c].map(f.format)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def build_report(test_df, xfg, metrics, source_note, out_dir=C.PATHS["reports"], n_top=10):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zones = V.zone_summary(test_df, xfg).reset_index()
    players = M.player_table(test_df, xfg, C.MIN_ATTEMPTS_PLAYER)
    zones.to_csv(out / "zone_summary.csv", index=False)
    players.to_csv(out / "player_xfg.csv", index=False)

    pct = "{:.1%}"
    val = metrics["validation"]
    vrows = pd.DataFrame([{
        "Model": k, "ROC-AUC": v["test"]["roc_auc"], "Brier": v["test"]["brier"],
        "Accuracy": v["test"]["accuracy"], "Mean xFG%": v["test"]["mean_xfg"], "Actual FG%": v["test"]["base_rate"]}
        for k, v in val.items()])
    m = val["location+style"]["test"]
    mf = metrics["final_in_sample_test"]
    pc = ["player_name", "attempts", "fg_pct", "xfg_pct", "fg_minus_xfg", "fg_minus_xfg_shrunk"]
    pf = {"fg_pct": pct, "xfg_pct": pct, "fg_minus_xfg": "{:+.1%}", "fg_minus_xfg_shrunk": "{:+.1%}", "attempts": "{:,}"}
    pn = {"player_name": "Player", "attempts": "FGA", "fg_pct": "FG%", "xfg_pct": "xFG%",
          "fg_minus_xfg": "FG% - xFG% (raw)", "fg_minus_xfg_shrunk": "FG% - xFG% (shrunk)"}
    z = zones.rename(columns={"zone": "Zone", "fga": "FGA", "fg": "FG%", "xfg": "xFG%", "freq": "Share of FGA"})
    ranked = players[pc].rename(columns=pn)
    pfr = {pn[k]: v for k, v in pf.items()}
    by_q = players.sort_values("xfg_pct", ascending=False)[pc].rename(columns=pn)

    text = f"""# NBA Shot Chart Spatial Analysis - Summary

> **Data:** {source_note}

Test set: {m['n']:,} shots from seasons >= {C.TRAIN_END_SEASON_START}. Validation models were trained on earlier seasons only.

## Model performance (chronological holdout)

{_md(vrows, {"ROC-AUC": "{:.3f}", "Brier": "{:.4f}", "Accuracy": "{:.3f}", "Mean xFG%": pct, "Actual FG%": pct})}

Naive Brier baseline (always predict the training base rate): {metrics['baseline_brier_test']:.4f}.
`location` uses court position and game context only; `location+style` adds the shot style from NBA's action type (dunk, layup, pullup, step-back...);
`gbm_benchmark` is gradient boosting on the same features, shown only to measure how much a linear logit leaves unexplained.

**Reading the numbers.** The logistic `location+style` model is the deliverable. Its precision/recall at a 0.5 cutoff are
{m['precision']:.3f} / {m['recall']:.3f} (confusion matrix TN={m['confusion_matrix']['tn']:,}, FP={m['confusion_matrix']['fp']:,},
FN={m['confusion_matrix']['fn']:,}, TP={m['confusion_matrix']['tp']:,}); recall is low because most shots have make-probability below 50%,
so ROC-AUC and Brier score are the primary measures. Mean xFG% sits below actual FG% because league shooting efficiency rose after the training
years; the final models below absorb that with a per-season effect.

**Final models** (used for every table and chart below) are refit on all seasons with a per-season effect, so xFG% matches each season's league
scoring level (in-sample ROC-AUC {mf['roc_auc']:.3f}, mean xFG% {mf['mean_xfg']:.1%} vs actual {mf['base_rate']:.1%}). These are league-level models with a few dozen parameters
fit on millions of shots, so in-sample optimism is negligible.

## League xFG% by zone (test seasons)

{_md(z, {"FGA": "{:,}", "FG%": pct, "xFG%": pct, "Share of FGA": pct})}

## Best shot makers (min {C.MIN_ATTEMPTS_PLAYER} FGA in the test seasons)

Ranked by the **shrunk** gap (FG% - xFG%, pulled toward the league mean in proportion to sample noise), so small samples do not dominate.

{_md(ranked.head(n_top), pfr)}

## Weakest shot makers

{_md(ranked.tail(n_top).iloc[::-1], pfr)}

## Best shot quality (highest xFG% - where and how they shoot)

{_md(by_q.head(n_top), pfr)}

## Poorest shot quality (lowest xFG%)

{_md(by_q.tail(n_top).iloc[::-1], pfr)}
"""
    (out / "summary.md").write_text(text)
    return text
