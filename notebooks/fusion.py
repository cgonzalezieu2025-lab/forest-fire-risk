# %% [markdown]
# # Fusion: random forest + CNN
#
# Interactive companion to `reports/fusion/README.md` (the written report with all figures).
# The meta-learner is fitted by the pipeline, not here:
#
#     python -m forest_fire_risk.fusion             # fit on 2020, evaluate on 2021-22 -> reports/fusion/
#     python -m forest_fire_risk.fusion_figures     # figures
#
# This notebook loads those results and shows how to work with them. Unlike the tabular and CNN
# notebooks, **every cell runs without the raw data, the tiles or a GPU**: fusion only needs the
# two committed prediction files, `reports/tabular/predictions.csv` and `reports/cnn/predictions.csv`.
#
# | Years | Role here |
# |---|---|
# | 2017-2019 | not used (the base models were trained on it) |
# | 2020 | fit the meta-learner; 5-fold CV inside 2020 picks its settings and threshold |
# | 2021-2022 | test, scored once, at the end |

# %%
import json

import numpy as np
import pandas as pd
from IPython.display import Image, display

from forest_fire_risk.fusion import INPUTS, REPORT_DIR, fit_meta, load_predictions, logit
from forest_fire_risk.tabular import scores, season_weights

FIG = REPORT_DIR / "figures"
pd.set_option("display.precision", 3)

# %% [markdown]
# ## 1. The inputs: one RF and one CNN probability per sample
#
# `load_predictions` joins the two files on `sample_id` and fails if they do not hold exactly
# the same samples, labels and splits.

# %%
df = load_predictions()
fit_rows = df[df.split == "val"].reset_index(drop=True)      # 2020
test = df[df.split == "test"].reset_index(drop=True)         # 2021-22
pd.crosstab(df.split.map({"val": "2020 (fit fusion)", "test": "2021-22 (test)"}), df.label,
            margins=True).rename(columns={0: "no fire", 1: "fire"})

# %% [markdown]
# How similar are the two base models? A correlation near 1 would mean they say the same thing
# and fusion has nothing to add. Around 0.6 they often disagree, which is what makes combining
# them worthwhile. On the logit scale the CNN's extreme probabilities (as low as 2e-5) are spread
# out instead of squashed against 0.

# %%
pd.DataFrame({"probabilities": df.groupby("split").apply(lambda g: g.p_rf.corr(g.p_cnn)),
              "logits": df.groupby("split").apply(lambda g: pd.Series(logit(g.p_rf)).corr(pd.Series(logit(g.p_cnn))))})

# %%
df.groupby(["split", "label"])[INPUTS].describe().T.loc[(slice(None), ["min", "50%", "max"]), :]

# %% [markdown]
# ## 2. Chosen settings (from cross-validation inside 2020)
#
# $p_{final} = \sigma(w_0 + w_{rf}\, z_{rf} + w_{cnn}\, z_{cnn})$, with $z$ the standardised
# logit of each base model, so the two weights are "change in log-odds of fire per standard
# deviation" and can be compared directly.

# %%
meta = json.loads((REPORT_DIR / "meta_learner.json").read_text())
pd.Series({"input scale": meta["input"], "C": meta["C"], "CV AUPRC (2020)": round(meta["cv_auprc"], 3),
           "threshold": round(meta["threshold"], 3), "w0 (intercept)": round(meta["intercept"], 3),
           "w_rf": round(meta["weights"]["p_rf"], 3), "w_cnn": round(meta["weights"]["p_cnn"], 3)})

# %% [markdown]
# Every setting tried: the logit scale beats raw probabilities for every C, and C hardly matters
# from 0.1 upwards (only two inputs, so there is little to regularise).

# %%
cv = pd.read_csv(REPORT_DIR / "cv_results.csv")
cv.pivot(index="param_logisticregression__C", columns="input", values="mean_test_score").rename_axis("C")

# %% [markdown]
# ## 3. Test results (2021-22), standard and season-matched
#
# Season-matched = test non-fires re-weighted so their months follow the fires' months, which
# removes the "fire season" shortcut (random guessing: AUPRC 0.351).
# "Average" is a fusion that learns nothing: $(p_{rf} + p_{cnn})/2$.

# %%
metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
metrics.pivot(index="model", columns="evaluation", values=["auprc", "roc_auc", "f1"]).loc[
    ["fusion", "average", "rf", "cnn"]]

# %%
for f in ["01_scatter_boundary.png", "02_roc_pr_curves.png", "03_auprc_bars.png"]:
    display(Image(FIG / f))

# %% [markdown]
# ## 4. Refit the meta-learner live
#
# The meta-learner has three parameters and 2,098 training rows, so refitting takes seconds.
# The test scores below are recomputed from scratch and should match the table above.

# %%
fit = fit_meta(fit_rows)
p_fusion = fit["model"].predict_proba(test[INPUTS])[:, 1]
w = season_weights(test.label, test.month)
pd.DataFrame({"standard": scores(test.label, p_fusion, fit["threshold"]),
              "season_matched": scores(test.label, p_fusion, fit["threshold"], w)})

# %% [markdown]
# Scoring a single new sample needs only its two base-model probabilities:

# %%
example = pd.DataFrame({"p_rf": [0.45, 0.45, 0.20, 0.20], "p_cnn": [0.05, 0.90, 0.95, 0.9999]})
example.assign(p_fusion=fit["model"].predict_proba(example)[:, 1],
               predicted_fire=lambda d: d.p_fusion >= fit["threshold"])

# %% [markdown]
# Rows 1-2: same lukewarm random forest (0.45), opposite image evidence, and the CNN decides.
# Rows 3-4: the random forest leans towards "no fire" (0.20). A CNN at 0.95 is not enough to
# overrule it: the CNN is overconfident and says 0.95 or more for 17 % of the 2020 samples (42 % of
# the fires), so 0.95 is ordinary evidence for this model. Only an extreme value (0.9999, under 1 %
# of samples) flips the answer. This is what the logit scale buys:
# 0.95 and 0.9999 look almost equal as probabilities but are far apart as logits (2.9 vs 9.2).

# %% [markdown]
# ## 5. Is the gain real?
#
# Paired bootstrap: resample the 3,758 test samples with replacement 2,000 times and recompute
# "fusion AUPRC minus base-model AUPRC" each time. If the 95 % interval stays above zero, the
# gain is not luck in which test samples we happen to have. (It does not cover the CNN's own
# run-to-run variance of about +-0.02-0.03; see the report.)

# %%
boot = pd.read_csv(REPORT_DIR / "bootstrap.csv").set_index("vs")
pd.DataFrame({ev: boot.apply(lambda r: f"{r[ev + '_mean']:+.3f}  [{r[ev + '_ci_low']:+.3f}, {r[ev + '_ci_high']:+.3f}]",
                             axis=1) for ev in ["standard", "season_matched"]}).rename(
    index={"rf": "fusion minus RF", "cnn": "fusion minus CNN"})

# %% [markdown]
# ## 6. Where fusion changes the decision
#
# Compare the random forest alone (its own threshold, 0.369) with fusion (threshold from 2020) on
# each test sample. Most samples get the same answer; the interesting ones are the switches.

# %%
rf_thr = json.loads((REPORT_DIR / "meta_learner.json").read_text())["thresholds_used"]["rf"]
cmp = test.assign(rf_says=test.p_rf >= rf_thr, fusion_says=p_fusion >= fit["threshold"])
cmp["change"] = np.select(
    [cmp.rf_says == cmp.fusion_says, ~cmp.rf_says & cmp.fusion_says],
    ["same answer", "RF: no fire -> fusion: fire"], "RF: fire -> fusion: no fire")
cmp["fusion_right"] = cmp.fusion_says == cmp.label.astype(bool)
summary = cmp.groupby("change").agg(samples=("label", "size"), actual_fires=("label", "sum"),
                                    fusion_right=("fusion_right", "sum"))
summary["share_fusion_right"] = summary.fusion_right / summary.samples
summary

# %% [markdown]
# Net effect of the switches: fires caught and false alarms, RF alone vs fusion.

# %%
pd.DataFrame({
    "RF alone": {"fires caught": int((cmp.rf_says & (cmp.label == 1)).sum()),
                 "false alarms": int((cmp.rf_says & (cmp.label == 0)).sum())},
    "fusion": {"fires caught": int((cmp.fusion_says & (cmp.label == 1)).sum()),
               "false alarms": int((cmp.fusion_says & (cmp.label == 0)).sum())}})

# %% [markdown]
# **Reading it.** Fusion catches about the same number of fires as the random forest alone (+5)
# but raises **75 fewer false alarms** (419 instead of 494). Most of the gain comes from the
# 215 samples the random forest flagged and fusion cleared: 70 % of those clearings are correct.
# The opposite switches (145 samples, "no fire" -> "fire") are right only about half the time,
# so they roughly cancel out. In short: the image is most useful as a **second opinion that
# vetoes random-forest false alarms**, e.g. hot, dry days over cells whose tile does not look
# like fuel that burns.
#
# The samples where the image overrules the random forest most strongly:

# %%
cmp.assign(p_fusion=p_fusion)[cmp.change != "same answer"].assign(
    shift=lambda d: (d.p_fusion - d.p_rf).abs()).nlargest(8, "shift")[
    ["sample_id", "time", "label", "p_rf", "p_cnn", "p_fusion", "change"]]
