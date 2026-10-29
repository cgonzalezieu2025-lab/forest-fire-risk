"""Fusion: a logistic-regression meta-learner on the random forest and CNN probabilities.

    python -m forest_fire_risk.fusion             # fit on 2020, evaluate on 2021-22 -> reports/fusion/
    python -m forest_fire_risk.fusion_figures     # figures from those results

Inputs are the two base models' predictions, joined on `sample_id`:
    reports/tabular/predictions.csv   p_rf   (random forest on the 27 tabular drivers)
    reports/cnn/predictions.csv       p_cnn  (ResNet-18 on the Sentinel-2 tile)
Neither base model was fitted or tuned on 2020 or 2021-22, so their predictions on those
years are honest out-of-sample predictions (stacking without leakage).

How each year is used:
    2017-2019  not used here (the base models were trained on it).
    2020       fit the meta-learner. 5-fold stratified cross-validation inside 2020 picks
               the input scale (probability or logit) and the L2 strength C, and its
               out-of-fold predictions pick the decision threshold. The final meta-learner
               is then refit on all of 2020.
    2021-2022  test, scored once at the end, plain and season-matched.

The meta-learner is  p_final = sigmoid(w0 + w_rf * z_rf + w_cnn * z_cnn)  where z is the
(standardised) probability or logit of each base model, so w_rf and w_cnn are directly
comparable: change in log-odds of fire per standard deviation of that model's output.

Outputs (reports/fusion/): metrics.csv, cv_results.csv, meta_learner.json, bootstrap.csv and
predictions.csv (p_fusion for 2020 out-of-fold and 2021-22).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from forest_fire_risk.tabular import SEED, best_threshold, scores, season_weights

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "fusion"
TABULAR_PREDS = ROOT / "reports" / "tabular" / "predictions.csv"
CNN_PREDS = ROOT / "reports" / "cnn" / "predictions.csv"
TABULAR_METRICS = ROOT / "reports" / "tabular" / "metrics.csv"
CNN_METRICS = ROOT / "reports" / "cnn" / "metrics.csv"
INPUTS = ["p_rf", "p_cnn"]
EPS = 1e-6                     # the CNN outputs probabilities as small as 2e-5; clip before the logit
N_BOOT = 2000


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


TRANSFORMS = {"prob": None, "logit": logit}
GRID = {"functiontransformer__func": list(TRANSFORMS.values()),
        "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]}
NAMES = {"fusion": "Fusion (RF + CNN)", "average": "Average of RF and CNN",
         "rf": "Random forest (tabular)", "cnn": "CNN (image)"}


def make_meta():
    """Optional logit -> standardise -> L2 logistic regression on the two base probabilities."""
    return make_pipeline(FunctionTransformer(), StandardScaler(), LogisticRegression(max_iter=1000))


def load_predictions(tabular_path=TABULAR_PREDS, cnn_path=CNN_PREDS):
    """One row per 2020 / 2021-22 sample with label, split, month, p_rf and p_cnn.
    Fails loudly if the two files do not hold exactly the same samples, labels and splits."""
    tab = pd.read_csv(tabular_path, parse_dates=["time"])
    cnn = pd.read_csv(cnn_path, parse_dates=["time"])
    df = tab[["sample_id", "label", "time", "split", "p_rf"]].merge(
        cnn[["sample_id", "label", "split", "p_cnn"]], on="sample_id", how="outer",
        suffixes=("", "_cnn"), indicator=True)
    if (df._merge != "both").any():
        raise ValueError(f"{(df._merge != 'both').sum()} samples are in only one predictions file")
    for col in ("label", "split"):
        if (df[col] != df[f"{col}_cnn"]).any():
            raise ValueError(f"tabular and CNN predictions disagree on `{col}`")
    df["month"] = df.time.dt.month
    return df.drop(columns=["_merge", "label_cnn", "split_cnn"]).reset_index(drop=True)


def fit_meta(fit_rows, features=INPUTS):
    """Grid search over input scale and C with 5-fold stratified CV on `fit_rows` (AUPRC),
    refit on all of them, and pick the threshold on the out-of-fold predictions."""
    X, y = fit_rows[features], fit_rows.label
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(make_meta(), GRID, scoring="average_precision", cv=cv, refit=True)
    search.fit(X, y)
    oof = cross_val_predict(search.best_estimator_, X, y, cv=cv, method="predict_proba")[:, 1]
    lr = search.best_estimator_.named_steps["logisticregression"]
    func = search.best_params_["functiontransformer__func"]
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["input"] = ["logit" if f is logit else "prob" for f in cv_results.param_functiontransformer__func]
    return {"model": search.best_estimator_, "input": "logit" if func is logit else "prob",
            "C": search.best_params_["logisticregression__C"], "cv_auprc": search.best_score_,
            "cv_results": cv_results.drop(columns=["params", "param_functiontransformer__func"]),
            "oof": oof, "threshold": best_threshold(y, oof),
            "intercept": float(lr.intercept_[0]), "weights": dict(zip(features, lr.coef_[0].tolist()))}


def paired_bootstrap(y, month, p_a, p_b, n_boot=N_BOOT, seed=SEED):
    """Resample test samples with replacement; return AUPRC(a) - AUPRC(b) per resample,
    plain and season-matched (season weights recomputed on each resample)."""
    rng = np.random.default_rng(seed)
    y, month, p_a, p_b = map(np.asarray, (y, month, p_a, p_b))
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        w = season_weights(y[i], month[i])
        rows.append({"standard": average_precision_score(y[i], p_a[i]) - average_precision_score(y[i], p_b[i]),
                     "season_matched": average_precision_score(y[i], p_a[i], sample_weight=w)
                     - average_precision_score(y[i], p_b[i], sample_weight=w)})
    return pd.DataFrame(rows)


def main():
    df = load_predictions()
    fit_rows, test = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    print(f"meta-learner fit on 2020: {len(fit_rows)} samples ({fit_rows.label.sum()} fires); "
          f"test 2021-22: {len(test)} samples ({test.label.sum()} fires)")
    print("correlation of p_rf and p_cnn:", {s: round(g.p_rf.corr(g.p_cnn), 3) for s, g in df.groupby("split")})

    meta = fit_meta(fit_rows)
    print(f"\nbest input: {meta['input']}, C = {meta['C']}, CV AUPRC on 2020 {meta['cv_auprc']:.3f}, "
          f"threshold {meta['threshold']:.3f}")
    print(f"p_final = sigmoid({meta['intercept']:.3f} + {meta['weights']['p_rf']:.3f} * z_rf "
          f"+ {meta['weights']['p_cnn']:.3f} * z_cnn)   (z = standardised {meta['input']})")

    # Test probabilities. The base models keep the thresholds from their own reports (picked on
    # 2017-19); the simple average is a no-learning fusion whose threshold is picked on 2020.
    probs = {"fusion": meta["model"].predict_proba(test[INPUTS])[:, 1],
             "average": test[INPUTS].mean(axis=1).to_numpy(),
             "rf": test.p_rf.to_numpy(), "cnn": test.p_cnn.to_numpy()}
    base = pd.concat([pd.read_csv(TABULAR_METRICS), pd.read_csv(CNN_METRICS)])
    base = base[base.evaluation == "standard"].set_index("model").threshold
    thresholds = {"fusion": meta["threshold"], "average": best_threshold(fit_rows.label, fit_rows[INPUTS].mean(axis=1)),
                  "rf": float(base["rf"]), "cnn": float(base["cnn"])}
    rows = []
    for evaluation, w in [("standard", None), ("season_matched", season_weights(test.label, test.month))]:
        for name, p in probs.items():
            rows.append({"model": name, "evaluation": evaluation, **scores(test.label, p, thresholds[name], w)})
    metrics = pd.DataFrame(rows)
    print("\n", metrics.round(3).to_string(index=False))

    boot = pd.concat([paired_bootstrap(test.label, test.month, probs["fusion"], probs[b]).assign(vs=b)
                      for b in ("rf", "cnn")])
    summary = boot.groupby("vs").agg(["mean", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975),
                                      lambda s: (s > 0).mean()])
    summary.columns = [f"{e}_{s}" for e, s in zip(summary.columns.get_level_values(0),
                                                    ["mean", "ci_low", "ci_high", "share_positive"] * 2)]
    print("\nfusion AUPRC minus base model AUPRC on test (paired bootstrap, 95 % interval):")
    print(summary.round(3).to_string())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    meta["cv_results"].to_csv(REPORT_DIR / "cv_results.csv", index=False)
    summary.reset_index().to_csv(REPORT_DIR / "bootstrap.csv", index=False)
    (REPORT_DIR / "meta_learner.json").write_text(json.dumps(
        {k: meta[k] for k in ("input", "C", "cv_auprc", "threshold", "intercept", "weights")}
        | {"thresholds_used": thresholds,
           "input_correlation": {s: g.p_rf.corr(g.p_cnn) for s, g in df.groupby("split")}}, indent=2))
    preds = pd.concat([fit_rows.assign(p_fusion=meta["oof"]), test.assign(p_fusion=probs["fusion"])])
    preds[["sample_id", "label", "time", "split", "p_rf", "p_cnn", "p_fusion"]].to_csv(
        REPORT_DIR / "predictions.csv", index=False)
    print(f"\nwrote {REPORT_DIR}")


if __name__ == "__main__":
    main()
