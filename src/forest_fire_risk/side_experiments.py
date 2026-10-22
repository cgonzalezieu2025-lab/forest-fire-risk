"""Side experiments for the tabular branch. NOT part of the main pipeline.

    python -m forest_fire_risk.tabular              # run the main pipeline first
    python -m forest_fire_risk.side_experiments     # -> reports/tabular/side_experiments.csv

They answer two "what if" questions for the report. Their models are never saved and they
never touch predictions.csv, so the fusion model only ever sees the main LR / RF.

  A. all_years   Train on every Track A year up to 2019 (2006-2019, tabular only) instead of
                 the 2017-2019 intersection. Would more years of tabular data help? Same
                 hyperparameters as the main models; leave-one-year-out CV (14 folds) picks
                 the threshold.
  B. mean7       Add the mean of each dynamic driver over the last 7 days to the fire-day
                 features. Does the build-up before the fire day carry extra signal? Tuned
                 from scratch with the same grids and CV as the main models.

Both are scored on exactly the same 2021-22 test samples as the main models.
"""
import json

import pandas as pd

from forest_fire_risk.data import DYNAMIC
from forest_fire_risk.tabular import (FEATURES, LR_GRID, REPORT_DIR, RF_GRID, evaluate, fit,
                                      load_data, make_lr, make_rf)

MEAN7 = FEATURES + [f"{v}_mean7" for v in DYNAMIC]


def all_years(test):
    best = json.loads((REPORT_DIR / "best_params.json").read_text())
    full = load_data(intersection=False)
    train = full[full.split == "train"]
    print(f"A. all_years: {len(train)} train samples ({train.year.min()}-{train.year.max()})")
    fitted = {name: fit(make(), {k: [v] for k, v in best[name]["params"].items()}, train)
              for name, make in [("logreg", make_lr), ("rf", make_rf)]}
    return evaluate(fitted, train, test).assign(experiment="all_years", train_years="2006-2019")


def mean7():
    tab = load_data(window=7)
    train, test = tab[tab.split == "train"], tab[tab.split == "test"]
    print(f"B. mean7: {len(MEAN7)} features")
    fitted = {name: fit(make(), grid, train, MEAN7)
              for name, make, grid in [("logreg", make_lr, LR_GRID), ("rf", make_rf, RF_GRID)]}
    return evaluate(fitted, train, test, MEAN7).assign(experiment="mean7", train_years="2017-2019"), test


def main():
    b, test = mean7()
    a = all_years(test)
    out = pd.concat([a, b])
    out = out[out.model != "month"]                    # the baseline is in metrics.csv already
    main_rows = pd.read_csv(REPORT_DIR / "metrics.csv").query("model != 'month'")
    out = pd.concat([main_rows.assign(experiment="main", train_years="2017-2019"), out])
    cols = ["experiment", "train_years", "model", "evaluation", "auprc", "roc_auc", "f1",
            "precision", "recall", "threshold", "cv_auprc"]
    out[cols].to_csv(REPORT_DIR / "side_experiments.csv", index=False)
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
