"""Random Forest model: predict whether a departure will be delayed > 15 min.

Only information available before push-back is used (no leakage from the actual
departure time): origin, destination, aircraft type, scheduled hour/day/month,
reported weather, planned turnaround, turnaround slack and the inbound
(previous-leg) delay.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT_FEATURES = ["Airport", "Destination", "Aircraft_Type", "Weather"]
NUM_FEATURES = ["Dep_Hour", "Day_Num", "Month", "Turnaround_Time", "Turn_Slack", "Prev_Delay_Min"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "Is_Delayed"
SEED = 22


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    v = df[df[TARGET].notna()].copy()
    v["Day_Num"] = v["Scheduled_Departure"].dt.dayofweek
    X = v[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X, v[TARGET].astype(int)


def build_pipeline(**rf_params) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
                                random_state=SEED, **rf_params)
    return Pipeline([("prep", pre), ("rf", rf)])


def buffer_scenario(model: Pipeline, df: pd.DataFrame, min_slack: int) -> dict:
    """Predicted delayed departures if every same-day turnaround had at least `min_slack` minutes of slack.

    Inbound delays are held fixed, so second-order knock-on benefits are not counted (conservative).
    """
    X, y = model_frame(df)
    base = model.predict_proba(X)[:, 1]
    Xs = X.copy()
    same_day = Xs["Turn_Slack"] < 240
    add = (min_slack - Xs["Turn_Slack"]).clip(lower=0).where(same_day, 0)
    Xs["Turn_Slack"] += add
    Xs["Turnaround_Time"] += add
    new = model.predict_proba(Xs)[:, 1]
    # class_weight="balanced" inflates raw probabilities, so apply the model's *relative* reduction
    # to the actual number of delayed departures
    rel = 1 - new.sum() / base.sum()
    actual = int(y.sum())
    return {"actual": actual, "new": actual * (1 - rel), "avoided": actual * rel, "rel": rel,
            "flights_changed": int((add > 0).sum()), "minutes_added": float(add.sum())}


def train(df: pd.DataFrame, out_dir: Path) -> dict:
    X, y = model_frame(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"rf__max_depth": [12, 20], "rf__min_samples_leaf": [5, 20],
                    "rf__max_features": ["sqrt"]},
        scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1,
        refit=True,
    )
    grid.fit(X_tr, y_tr)
    model = grid.best_estimator_

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    pidx = np.linspace(0, len(prec) - 1, min(200, len(prec))).astype(int)

    # Impurity importance from the forest, aggregated per original feature (1:1 with ordinal encoding)
    imp = model.named_steps["rf"].feature_importances_
    sample = X_te.sample(min(8000, len(X_te)), random_state=SEED)
    perm = permutation_importance(model, sample, y_te.loc[sample.index], scoring="roc_auc",
                                  n_repeats=5, random_state=SEED, n_jobs=1)

    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "positive_rate": float(y.mean()),
        "best_params": {k.replace("rf__", ""): v for k, v in grid.best_params_.items()},
        "cv_auc": float(grid.best_score_),
        "cv_results": pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        .assign(params=lambda d: d["params"].astype(str)).to_dict("records"),
        "accuracy": float(accuracy_score(y_te, pred)),
        "baseline_accuracy": float(max(y_te.mean(), 1 - y_te.mean())),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
        "pr": {"precision": prec[pidx].tolist(), "recall": rec[pidx].tolist()},
        "importance": {f: float(i) for f, i in zip(FEATURES, imp)},
        "perm_importance": {f: float(m) for f, m in zip(FEATURES, perm.importances_mean)},
        "perm_importance_std": {f: float(s) for f, s in zip(FEATURES, perm.importances_std)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "rf_model.joblib", compress=3)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
