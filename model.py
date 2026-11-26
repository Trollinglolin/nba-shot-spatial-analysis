"""xFG% models.

Validation (honest, chronological): fit on seasons < TRAIN_END_SEASON_START, score the later seasons.
  - location        : spatial + game-context features (the plan's baseline)
  - location+style  : adds shot style from ACTION_TYPE (dunk / layup / pullup / step-back / ...)
  - gbm (benchmark) : gradient boosting on the same features, to show how much a linear logit leaves on the table
Final (used for player tables and charts): the same logistic models refit on ALL seasons with a per-season
effect, so xFG% is calibrated to each season's league scoring level (shooting efficiency drifts upward over time).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

import constants as C
import features as F

LOC_CATS = F.LOCATION_CATEGORICAL
FULL_CATS = F.LOCATION_CATEGORICAL + F.STYLE_CATEGORICAL
EXTRA_NUM = ["angle_deg", "angle_from_center", "seconds_left_in_period"]


def build_logit(categorical) -> Pipeline:
    """Distance enters through a spline basis so the linear logit can bend with range."""
    pre = ColumnTransformer([
        ("dist", SplineTransformer(n_knots=7, degree=3, knots="quantile"), ["distance_ft"]),
        ("num", StandardScaler(), EXTRA_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    pipe.features_ = ["distance_ft"] + EXTRA_NUM + list(categorical)
    return pipe


def build_gbm(categorical) -> Pipeline:
    cats = list(categorical)
    nums = ["distance_ft"] + EXTRA_NUM
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
        ("num", "passthrough", nums)])
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=48,
                                         categorical_features=[True] * len(cats) + [False] * len(nums),
                                         random_state=0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.features_ = cats + nums
    return pipe


def predict_xfg(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[model.features_])[:, 1]


def evaluate(model, df: pd.DataFrame) -> dict:
    y = df[F.TARGET].values
    p = predict_xfg(model, df)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "n": int(len(y)), "base_rate": float(y.mean()), "mean_xfg": float(p.mean()),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate(df: pd.DataFrame, out_dir=C.PATHS["models"]):
    train, test = F.season_split(df)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    y = train[F.TARGET]

    validation = {}
    for name, builder, cats in (("location", build_logit, LOC_CATS),
                                ("location+style", build_logit, FULL_CATS),
                                ("gbm_benchmark", build_gbm, FULL_CATS)):
        print(f"fitting validation model: {name}", flush=True)
        m = builder(cats)
        m.fit(train[m.features_], y)
        validation[name] = {"train": evaluate(m, train), "test": evaluate(m, test)}
    metrics = {
        "split": f"train seasons < {C.TRAIN_END_SEASON_START}, test >= {C.TRAIN_END_SEASON_START}",
        "validation": validation,
        "baseline_brier_test": brier_score_loss(test[F.TARGET], np.full(len(test), y.mean())),
    }

    # Final models: all seasons, with a per-season effect (season-calibrated xFG%).
    final = {}
    for key, cats in (("full", FULL_CATS + ["season"]), ("loc", LOC_CATS + ["season"])):
        print(f"fitting final model: {key}", flush=True)
        final[key] = build_logit(cats)
        final[key].fit(df[final[key].features_], df[F.TARGET])
    final["latest_season"] = str(df["season"].max())
    metrics["final_in_sample_test"] = evaluate(final["full"], test)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(final, Path(out_dir) / "xfg_models.joblib")
    json.dump(metrics, open(Path(out_dir) / "metrics.json", "w"), indent=2)
    return final, metrics, train, test


def shrink(table: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of FG% - xFG% toward the league mean gap.

    Small samples produce extreme gaps by luck; this pulls them toward the average in proportion to
    their sampling noise (binomial variance), so rankings are not dominated by 150-shot players.
    """
    t = table.copy()
    p = t["xfg_pct"].clip(0.05, 0.95)
    s2 = p * (1 - p) / t["attempts"]                      # sampling variance of the gap
    mean_gap = np.average(t["fg_minus_xfg"], weights=1 / (s2 + 1e-9))
    tau2 = max(float(np.var(t["fg_minus_xfg"]) - s2.mean()), 1e-5)   # true between-player variance
    w = tau2 / (tau2 + s2)
    t["fg_minus_xfg_shrunk"] = mean_gap + w * (t["fg_minus_xfg"] - mean_gap)
    t["se"] = np.sqrt(s2)
    return t


def player_table(df: pd.DataFrame, xfg: np.ndarray, min_attempts: int = 1) -> pd.DataFrame:
    d = df.assign(xfg=xfg, is3=(df["shot_type"] == "3PT Field Goal").astype(float),
                  isrim=(df["zone"] == "Restricted Area").astype(float))
    t = d.groupby(["player_id", "player_name"]).agg(
        attempts=(F.TARGET, "size"), fg_pct=(F.TARGET, "mean"), xfg_pct=("xfg", "mean"),
        avg_distance_ft=("distance_ft", "mean"), three_rate=("is3", "mean"), rim_rate=("isrim", "mean"),
    ).reset_index()
    t["fg_minus_xfg"] = t["fg_pct"] - t["xfg_pct"]
    t = t[t["attempts"] >= min_attempts].reset_index(drop=True)
    return shrink(t).sort_values("fg_minus_xfg_shrunk", ascending=False)


def predict_grid(loc_model, xs: np.ndarray, ys: np.ndarray, season: str) -> np.ndarray:
    """League-average xFG% at each (x, y) for `season`: mid-game, period 2, no clutch. Shape (len(ys), len(xs))."""
    gx, gy = np.meshgrid(xs, ys)
    sp = F.spatial_frame(gx.ravel(), gy.ravel())
    sp["period_group"], sp["clutch"], sp["seconds_left_in_period"], sp["season"] = "Q2", "other", 360.0, season
    return predict_xfg(loc_model, sp).reshape(gx.shape)
