"""Tabular models: logistic regression (baseline) and random forest.

    python -m forest_fire_risk.tabular            # tune, fit, evaluate -> reports/tabular/
    python -m forest_fire_risk.tabular_figures    # figures from those results

Data: the 27 fire-day features (STATIC + DYNAMIC on day 29) for the 2017-2022 samples that
also have a Sentinel-2 tile from a scene with at most MAX_CLOUD % cloud, so the tabular,
CNN and fusion models all see exactly the same samples.

How each year is used:
    2017-2019  train. Leave-one-year-out cross-validation (k = 3 folds, one per year) picks
               the hyperparameters and the decision threshold; the final models are then
               refit on all three years.
    2020       never used to fit or tune anything here. The final models only predict it,
               and the fusion model is trained on those predictions.
    2021-2022  test, scored once at the end.

Outputs (reports/tabular/): metrics.csv, cv_results.csv, best_params.json and
predictions.csv (2020 + 2021-22 probabilities per sample_id, for fusion). Fitted models go
to models/tabular/ (git-ignored).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forest_fire_risk.data import DYNAMIC, STATIC, load_track_a, split_by_year, to_tabular, with_images

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "tabular"
MODEL_DIR = ROOT / "models" / "tabular"
FEATURES = STATIC + DYNAMIC
# Tiles from scenes cloudier than this are dropped, which is what we expect the CNN to do
# (not confirmed yet). None keeps every tile with status=ok.
MAX_CLOUD = 20
SEED = 42

LR_GRID = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}        # smaller C = stronger L2
RF_GRID = {"randomforestclassifier__max_depth": [None, 10, 20],
           "randomforestclassifier__min_samples_leaf": [1, 5, 20],
           "randomforestclassifier__max_features": ["sqrt", 0.5],
           "randomforestclassifier__class_weight": [None, "balanced"]}
NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "month": "Month only (baseline)"}


def make_lr():
    """Median imputation -> standardisation -> L2 logistic regression."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000))


def make_rf():
    """Median imputation -> random forest (trees need no scaling)."""
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED))


def load_data(window=1, max_cloud=MAX_CLOUD, intersection=True):
    """One row per sample with `split`, `year` and `month`. With `intersection`, only samples
    that have a usable Sentinel-2 tile (2017-2022); otherwise every Track A sample."""
    tab = to_tabular(load_track_a(), window=window)
    if intersection:
        tab = with_images(tab)
        if max_cloud is not None:
            tab = tab[tab.cloud <= max_cloud]
    tab = tab.reset_index(drop=True)
    tab["split"] = split_by_year(tab)
    tab["year"], tab["month"] = tab.time.dt.year, tab.time.dt.month
    return tab


def best_threshold(y, p):
    """Probability cut-off that maximises F1 on (y, p)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])])          # the last (prec, rec) pair has no threshold


def season_weights(y, month):
    """Sample weights that give the negatives the same month distribution as the positives.

    Positives keep weight 1; a negative from month m gets P(m | fire) / P(m | no fire), rescaled
    so the negatives' total weight is unchanged. With these weights the month of a sample says
    nothing about its label, so a model cannot score by recognising fire season. Negatives
    from months without any fire get weight 0."""
    y, month = np.asarray(y), np.asarray(month)
    pos = pd.Series(month[y == 1]).value_counts(normalize=True)
    neg = pd.Series(month[y == 0]).value_counts(normalize=True)
    w = np.ones(len(y))
    w_neg = (pos / neg).reindex(month[y == 0]).fillna(0).to_numpy()
    w[y == 0] = w_neg * (y == 0).sum() / w_neg.sum()
    return w


def month_rate(train, rows):
    """'Month only' baseline: the fire rate of each calendar month in `train`."""
    rate = train.groupby("month").label.mean()
    return rows.month.map(rate).fillna(train.label.mean()).to_numpy()


def scores(y, p, threshold, weights=None):
    pred = (p >= threshold).astype(int)
    kw = {"sample_weight": weights}
    return {"auprc": average_precision_score(y, p, **kw), "roc_auc": roc_auc_score(y, p, **kw),
            "f1": f1_score(y, pred, **kw), "precision": precision_score(y, pred, **kw),
            "recall": recall_score(y, pred, **kw), "threshold": threshold}


def fit(model, grid, train, features=FEATURES):
    """Grid search with leave-one-year-out CV on `train` (scored by AUPRC), refit the best
    setting on all of `train`, and pick the threshold on out-of-fold predictions."""
    X, y, years = train[features], train.label, train.year
    cv = LeaveOneGroupOut()
    search = GridSearchCV(model, grid, scoring="average_precision", cv=cv, refit=True)
    search.fit(X, y, groups=years)
    oof = cross_val_predict(clone(search.best_estimator_), X, y, groups=years, cv=cv,
                            method="predict_proba")[:, 1]
    return {"model": search.best_estimator_, "params": search.best_params_,
            "cv_auprc": search.best_score_, "cv_results": pd.DataFrame(search.cv_results_),
            "oof": oof, "threshold": best_threshold(y, oof)}


def evaluate(fitted, train, test, features=FEATURES):
    """Test metrics for each fitted model and the month-only baseline, both on the plain test
    set and season-matched (negatives reweighted to the fires' month distribution)."""
    probs = {name: f["model"].predict_proba(test[features])[:, 1] for name, f in fitted.items()}
    probs["month"] = month_rate(train, test)
    thresholds = {name: f["threshold"] for name, f in fitted.items()}
    years = sorted(train.year.unique())
    oof_month = np.concatenate([month_rate(train[train.year != yr], train[train.year == yr]) for yr in years])
    thresholds["month"] = best_threshold(np.concatenate([train.label[train.year == yr] for yr in years]), oof_month)
    rows = []
    for evaluation, w in [("standard", None), ("season_matched", season_weights(test.label, test.month))]:
        for name, p in probs.items():
            rows.append({"model": name, "evaluation": evaluation,
                         **scores(test.label, p, thresholds[name], w),
                         "cv_auprc": fitted[name]["cv_auprc"] if name in fitted else np.nan})
    return pd.DataFrame(rows)


def main():
    tab = load_data()
    train, val, test = (tab[tab.split == s] for s in ("train", "val", "test"))
    print(f"{len(tab)} samples with a tile (cloud <= {MAX_CLOUD}%)")
    print(pd.crosstab(tab.split, tab.label).rename(columns={0: "no fire", 1: "fire"}))

    fitted = {}
    for name, model, grid in [("logreg", make_lr(), LR_GRID), ("rf", make_rf(), RF_GRID)]:
        print(f"\ntuning {NAMES[name]} ({len(ParameterGrid(grid))} settings x 3 folds) ...")
        fitted[name] = fit(model, grid, train)
        print(f"  best {fitted[name]['params']}  CV AUPRC {fitted[name]['cv_auprc']:.3f}"
              f"  threshold {fitted[name]['threshold']:.3f}")

    metrics = evaluate(fitted, train, test)
    print("\n", metrics.round(3).to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    pd.concat([f["cv_results"].assign(model=name) for name, f in fitted.items()]).to_csv(
        REPORT_DIR / "cv_results.csv", index=False)
    (REPORT_DIR / "best_params.json").write_text(json.dumps(
        {name: {"params": f["params"], "cv_auprc": f["cv_auprc"], "threshold": f["threshold"]}
         for name, f in fitted.items()}, indent=2, default=str))
    preds = pd.concat([val, test])[["sample_id", "label", "time", "split"]].copy()
    for name, f in fitted.items():
        preds[f"p_{name}"] = f["model"].predict_proba(pd.concat([val, test])[FEATURES])[:, 1]
    preds.to_csv(REPORT_DIR / "predictions.csv", index=False)
    pd.DataFrame({"sample_id": train.sample_id, **{f"oof_{n}": f["oof"] for n, f in fitted.items()}}).to_csv(
        MODEL_DIR / "train_oof.csv", index=False)
    for name, f in fitted.items():
        joblib.dump(f["model"], MODEL_DIR / f"{name}.joblib")
    print(f"\nwrote {REPORT_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
