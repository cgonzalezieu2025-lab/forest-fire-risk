# %% [markdown]
# # Tabular models: logistic regression and random forest
#
# Interactive companion to `reports/tabular/README.md` (the written report with all figures).
# The models are trained by the pipeline, not here:
#
#     python -m forest_fire_risk.tabular            # tune + train + evaluate -> reports/tabular/
#     python -m forest_fire_risk.side_experiments   # optional
#     python -m forest_fire_risk.tabular_figures    # figures
#
# This notebook loads those results and shows how to work with them.
#
# | Years | Role | Used to |
# |---|---|---|
# | 2017-2019 | train | fit; leave-one-year-out CV (k = 3) picks hyperparameters and threshold |
# | 2020 | fusion only | the final LR / RF only predict it; never used to fit or tune them |
# | 2021-2022 | test | scored once, at the end |

# %%
import json
from pathlib import Path

import joblib
import pandas as pd
from IPython.display import Image, display

from forest_fire_risk.tabular import FEATURES, MAX_CLOUD, MODEL_DIR, REPORT_DIR, load_data, season_weights, scores

FIG = REPORT_DIR / "figures"
pd.set_option("display.precision", 3)

# %% [markdown]
# ## 1. The data: tabular rows that also have a Sentinel-2 tile

# %%
tab = load_data()
print(f"{len(tab):,} samples with a tile (scene cloud <= {MAX_CLOUD}%), {len(FEATURES)} fire-day features")
pd.crosstab(tab.split, tab.label, margins=True).rename(columns={0: "no fire", 1: "fire"})

# %%
display(Image(FIG / "01_data_split.png"))

# %% [markdown]
# ## 2. Chosen settings (from cross-validation on 2017-19)

# %%
best = json.loads((REPORT_DIR / "best_params.json").read_text())
pd.DataFrame({m: {"settings": v["params"], "CV AUPRC": round(v["cv_auprc"], 3),
                  "threshold": round(v["threshold"], 3)} for m, v in best.items()}).T

# %% [markdown]
# ## 3. Test results (2021-22), standard and season-matched
#
# Season-matched = test non-fires re-weighted so their months follow the fires' months, which
# removes the "fire season" shortcut. The month-only baseline drops to chance (ROC-AUC 0.5).

# %%
metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
metrics.pivot(index="model", columns="evaluation", values=["auprc", "roc_auc", "f1"])

# %%
for f in ["02_roc_pr_curves.png", "03_season_matched.png", "04_confusion_matrices.png"]:
    display(Image(FIG / f))

# %% [markdown]
# ## 4. Using the saved models
#
# The fitted pipelines (imputer [+ scaler] + model) are in `models/tabular/`. They take the
# 27 raw features, so any row from `to_tabular` can be scored directly. Here: recompute the
# random forest's test scores from scratch, plain and season-matched.

# %%
rf = joblib.load(MODEL_DIR / "rf.joblib")
test = tab[tab.split == "test"]
p = rf.predict_proba(test[FEATURES])[:, 1]
pd.DataFrame({"standard": scores(test.label, p, best["rf"]["threshold"]),
              "season_matched": scores(test.label, p, best["rf"]["threshold"],
                                       season_weights(test.label, test.month))})

# %% [markdown]
# ## 5. What drives the predictions

# %%
for f in ["05_lr_coefficients.png", "06_rf_feature_importance.png"]:
    display(Image(FIG / f))
pd.read_csv(REPORT_DIR / "rf_feature_importance.csv", index_col=0).head(10)

# %% [markdown]
# ## 6. Predictions for the fusion model
#
# `predictions.csv`: one row per 2020 (`val`, to train fusion) and 2021-22 (`test`) sample.
# Join the CNN's predictions on `sample_id`.

# %%
preds = pd.read_csv(REPORT_DIR / "predictions.csv")
print(preds.split.value_counts().to_dict())
preds.head()

# %% [markdown]
# ## 7. Side experiments (not part of the main pipeline, not used by fusion)

# %%
side = REPORT_DIR / "side_experiments.csv"
if side.exists():
    display(pd.read_csv(side).pivot_table(index=["experiment", "model"], columns="evaluation", values="auprc"))
    display(Image(FIG / "10_side_experiments.png"))
