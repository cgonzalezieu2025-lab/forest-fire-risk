"""Figures for the tabular report, from the outputs of `python -m forest_fire_risk.tabular`
(and `side_experiments`, if it was run).

    python -m forest_fire_risk.tabular_figures      # -> reports/tabular/figures/*.png

All test-set figures use 2021-22. 2020 is not shown anywhere: it is kept for the fusion model.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from forest_fire_risk.tabular import (FEATURES, MAX_CLOUD, MODEL_DIR, NAMES, REPORT_DIR, SEED,  # noqa: E402
                                      load_data, month_rate)

FIG_DIR = REPORT_DIR / "figures"
FIRE, NOFIRE = "#eb6834", "#2a78d6"           # same as the EDA figures
COLOR = {"logreg": "#2a78d6", "rf": "#eb6834", "month": "#8a8984"}
STYLE = {"logreg": "-", "rf": "-", "month": "--"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titleweight": "bold", "font.size": 10})


def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("  ", name)


def fig_data_split(tab):
    counts = pd.crosstab(tab.year, tab.label)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(counts))
    ax.bar(x - 0.2, counts[0], 0.38, color=NOFIRE, label="no fire")
    ax.bar(x + 0.2, counts[1], 0.38, color=FIRE, label="fire")
    for lo, hi, text in [(-0.5, 2.5, "TRAIN\n3-fold CV (one fold per year)"), (2.5, 3.5, "FUSION\nonly predicted"),
                         (3.5, 5.5, "TEST\nscored once")]:
        ax.axvspan(lo, hi, color="#8a8984", alpha=0.08 if text.startswith("FUSION") else 0)
        ax.axvline(hi, color="#8a8984", lw=1, ls=":") if hi < 5.5 else None
        ax.text((lo + hi) / 2, counts.values.max() * 1.12, text, ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, counts.index)
    ax.set_ylim(0, counts.values.max() * 1.35)
    ax.set_xlabel("Year (end of the 30-day window)")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Samples used by the tabular models: {len(tab):,} with a Sentinel-2 tile (cloud ≤ {MAX_CLOUD}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    save(fig, "01_data_split.png")


def fig_roc_pr(test, probs, metrics):
    std = metrics[metrics.evaluation == "standard"].set_index("model")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(test.label, p)
        a.plot(fpr, tpr, STYLE[name], color=COLOR[name], lw=2, label=f"{NAMES[name]}  (ROC-AUC {std.roc_auc[name]:.3f})")
        prec, rec, _ = precision_recall_curve(test.label, p)
        b.plot(rec, prec, STYLE[name], color=COLOR[name], lw=2, label=f"{NAMES[name]}  (AUPRC {std.auprc[name]:.3f})")
    a.plot([0, 1], [0, 1], ":", color="black", lw=1, label="Random guessing (0.5)")
    b.axhline(test.label.mean(), ls=":", color="black", lw=1, label=f"Random guessing ({test.label.mean():.3f})")
    a.set(xlabel="False positive rate (share of no-fire samples flagged as fire)",
          ylabel="True positive rate (share of fires caught)", title="ROC curve, test 2021-22")
    b.set(xlabel="Recall (share of fires caught)", ylabel="Precision (share of fire alarms that are real)",
          title="Precision-recall curve, test 2021-22", ylim=(0, 1.02))
    for ax in (a, b):
        ax.legend(loc="lower right" if ax is a else "lower left", fontsize=8.5)
    save(fig, "02_roc_pr_curves.png")


def fig_season(metrics, prevalence):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    models = ["logreg", "rf", "month"]
    for ax, metric, chance in [(axes[0], "auprc", prevalence), (axes[1], "roc_auc", 0.5)]:
        x = np.arange(len(models))
        for k, (ev, label, alpha) in enumerate([("standard", "Standard test set", 1.0),
                                                 ("season_matched", "Season-matched", 0.45)]):
            vals = metrics[metrics.evaluation == ev].set_index("model").loc[models, metric]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=[COLOR[m] for m in models], alpha=alpha,
                          hatch="" if k == 0 else "//", edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
        ax.axhline(chance, ls=":", color="black", lw=1)
        ax.text(2.6, chance, f" chance\n ({chance:.2f})", va="center", fontsize=8.5)
        ax.set_xticks(x, [NAMES[m].replace(" (baseline)", "\n(baseline)") for m in models])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel({"auprc": "AUPRC", "roc_auc": "ROC-AUC"}[metric])
        ax.set_title({"auprc": "AUPRC (main metric)", "roc_auc": "ROC-AUC"}[metric])
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#8a8984", label="Standard test set"),
                            Patch(facecolor="#8a8984", alpha=0.45, hatch="//", edgecolor="white",
                                  label="Season-matched (month carries no information)")],
                   fontsize=8.5, loc="upper right")
    fig.suptitle("How much of the score is just 'fire season'? Test 2021-22, standard vs season-matched",
                 fontweight="bold", y=1.02)
    save(fig, "03_season_matched.png")


def fig_confusion(test, probs, best):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, name in zip(axes, ["logreg", "rf"]):
        thr = best[name]["threshold"]
        cm = confusion_matrix(test.label, probs[name] >= thr)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j]:.0%} of row)", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["no fire", "fire"])
        ax.set_yticks([0, 1], ["no fire", "fire"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{NAMES[name]} (threshold {thr:.2f})")
        ax.grid(False)
    fig.suptitle("Confusion matrices on test 2021-22 (threshold chosen by cross-validation on 2017-19)",
                 fontweight="bold", y=1.02)
    save(fig, "04_confusion_matrices.png")


def fig_lr_coefficients(lr):
    coef = pd.Series(lr.named_steps["logisticregression"].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(coef.index, coef, color=np.where(coef > 0, FIRE, NOFIRE))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient θ: change in log-odds of fire per +1 standard deviation of the feature")
    ax.set_title(f"Logistic regression coefficients (standardised features, C = "
                 f"{lr.named_steps['logisticregression'].C})")
    ax.text(0.98, 0.04, "orange: higher value → more likely fire\nblue: higher value → less likely fire",
            transform=ax.transAxes, ha="right", fontsize=8.5)
    save(fig, "05_lr_coefficients.png")


def fig_rf_importance(rf, test):
    perm = permutation_importance(rf, test[FEATURES], test.label, scoring="average_precision",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"perm": perm.importances_mean, "perm_sd": perm.importances_std,
                        "gini": rf.named_steps["randomforestclassifier"].feature_importances_},
                       index=FEATURES).sort_values("perm")
    imp.sort_values("perm", ascending=False).to_csv(REPORT_DIR / "rf_feature_importance.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    a.barh(imp.index, imp.perm, xerr=imp.perm_sd, color=FIRE, error_kw={"lw": 1, "ecolor": "#555"})
    a.set(xlabel="Drop in test AUPRC when the feature is shuffled", title="Permutation importance (test 2021-22)")
    b.barh(imp.index, imp.gini, color="#8a8984")
    b.set(xlabel="Mean decrease in impurity (sums to 1)", title="Impurity importance (from training)")
    fig.suptitle("Random forest: which features matter?", fontweight="bold", y=1.0)
    save(fig, "06_rf_feature_importance.png")


def fig_tuning(cv):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    folds = ["split0_test_score", "split1_test_score", "split2_test_score"]
    lr = cv[cv.model == "logreg"].copy()
    lr["C"] = lr["param_logisticregression__C"].astype(float)
    lr = lr.sort_values("C")
    for f, yr, mk in zip(folds, [2017, 2018, 2019], ["o", "s", "^"]):
        a.plot(lr.C, lr[f], mk, color=COLOR["logreg"], alpha=0.45, ms=6, label=f"fold: {yr} held out")
    a.plot(lr.C, lr.mean_test_score, "-", color=COLOR["logreg"], lw=2, label="mean of 3 folds")
    a.set_xscale("log")
    a.set(xlabel="C (log scale; smaller = stronger L2 penalty)", ylabel="Cross-validated AUPRC",
          title="Logistic regression: choosing C")
    a.legend(fontsize=8)
    rf = cv[cv.model == "rf"].copy()
    rf["depth"] = rf["param_randomforestclassifier__max_depth"].fillna("none").astype(str).str.replace(".0", "")
    best = (rf.groupby(["param_randomforestclassifier__min_samples_leaf", "depth"]).mean_test_score.max()
            .unstack(0).reindex(["10", "20", "none"]))
    for (leaf, col), shade in zip(best.items(), ["#f4a582", "#eb6834", "#a8401a"]):
        b.plot(best.index, col, "o-", color=shade, lw=2, ms=7, label=f"min. samples per leaf = {int(leaf)}")
    b.set(xlabel="Maximum tree depth", ylabel="Cross-validated AUPRC (best over the other settings)",
          title="Random forest: choosing tree size")
    b.legend(fontsize=8)
    fig.suptitle("Hyperparameter tuning with leave-one-year-out cross-validation on 2017-19",
                 fontweight="bold", y=1.02)
    save(fig, "07_tuning.png")


def fig_calibration(test, probs):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in ["logreg", "rf"]:
        frac, mean = calibration_curve(test.label, probs[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", color=COLOR[name], lw=2, ms=5, label=NAMES[name])
    ax.plot([0, 1], [0, 1], ":", color="black", lw=1, label="Perfectly calibrated")
    ax.set(xlabel="Predicted fire probability (mean per bin)", ylabel="Actual share of fires in the bin",
           title="Calibration on test 2021-22\n(10 bins with equal numbers of samples)")
    ax.legend(fontsize=8.5)
    save(fig, "08_calibration.png")


def fig_example_tree(train, rf):
    X = rf.named_steps["simpleimputer"].transform(train[FEATURES])
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X, train.label)
    fig, ax = plt.subplots(figsize=(19, 7))
    plot_tree(tree, feature_names=FEATURES, class_names=["no fire", "fire"], impurity=False,
              proportion=True, rounded=True, precision=2, fontsize=9, ax=ax)
    ax.set_title("One small decision tree (depth 3) trained on 2017-19, for illustration only.\n"
                 "Each box: the question asked, share of training samples reaching it, "
                 "[share no fire, share fire], majority class.", fontsize=10)
    save(fig, "09_example_decision_tree.png")


def fig_side_experiments():
    path = REPORT_DIR / "side_experiments.csv"
    if not path.exists():
        print("   (side_experiments.csv missing: run python -m forest_fire_risk.side_experiments)")
        return
    side = pd.read_csv(path)
    labels = {"main": "Main\n(2017-19, fire day)", "all_years": "A: all years\n(2006-19, fire day)",
              "mean7": "B: + 7-day means\n(2017-19)"}
    exps = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, ev in zip(axes, ["standard", "season_matched"]):
        d = side[side.evaluation == ev]
        x = np.arange(len(exps))
        for k, name in enumerate(["logreg", "rf"]):
            vals = d[d.model == name].set_index("experiment").loc[exps, "auprc"]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=COLOR[name], label=NAMES[name])
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x, [labels[e] for e in exps], fontsize=8.5)
        ax.set_ylim(0, 1.1)
        ax.set_title({"standard": "Standard test set", "season_matched": "Season-matched"}[ev])
    axes[0].set_ylabel("Test AUPRC (2021-22)")
    axes[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(1.1, -0.18), ncol=2, frameon=False)
    fig.suptitle("Side experiments (not used by fusion): more years, or 7-day means", fontweight="bold", y=1.02)
    save(fig, "10_side_experiments.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tab = load_data()
    train, test = tab[tab.split == "train"], tab[tab.split == "test"]
    lr, rf = (joblib.load(MODEL_DIR / f"{n}.joblib") for n in ("logreg", "rf"))
    probs = {"logreg": lr.predict_proba(test[FEATURES])[:, 1], "rf": rf.predict_proba(test[FEATURES])[:, 1],
             "month": month_rate(train, test)}
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    best = json.loads((REPORT_DIR / "best_params.json").read_text())

    print("figures ->", FIG_DIR)
    fig_data_split(tab)
    fig_roc_pr(test, probs, metrics)
    fig_season(metrics, test.label.mean())
    fig_confusion(test, probs, best)
    fig_lr_coefficients(lr)
    fig_rf_importance(rf, test)
    fig_tuning(pd.read_csv(REPORT_DIR / "cv_results.csv"))
    fig_calibration(test, probs)
    fig_example_tree(train, rf)
    fig_side_experiments()


if __name__ == "__main__":
    main()
