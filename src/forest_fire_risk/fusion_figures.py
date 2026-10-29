"""Figures for reports/fusion/, from the outputs of `python -m forest_fire_risk.fusion`.

    python -m forest_fire_risk.fusion_figures     # -> reports/fusion/figures/

Reuses the tabular ROC / PR plot so fusion is drawn next to the random forest and the CNN on
the same 2021-22 test samples.
"""
import json

import numpy as np
import pandas as pd

from forest_fire_risk import tabular_figures as tf
from forest_fire_risk.fusion import NAMES, REPORT_DIR, logit

plt = tf.plt
FIG_DIR = REPORT_DIR / "figures"
FUSION = "#7b3fa0"
for name, color, style in [("fusion", FUSION, "-"), ("average", "#8a8984", "--"),
                           ("rf", tf.COLOR["rf"], "-"), ("cnn", "#2e9e5b", "-")]:
    tf.NAMES[name], tf.COLOR[name], tf.STYLE[name] = NAMES[name], color, style
tf.FIG_DIR = FIG_DIR                                # tf.save writes here


def fig_scatter(test, meta):
    """Each test sample at (logit p_rf, logit p_cnn), coloured by label, with the line where the
    meta-learner's probability equals its threshold. Above / right of the line = predicted fire."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    x, y = logit(test.p_rf), logit(test.p_cnn)
    for lab, color, name in [(0, tf.NOFIRE, "no fire"), (1, tf.FIRE, "fire")]:
        m = test.label == lab
        ax.scatter(x[m], y[m], s=5, alpha=0.35, color=color, label=f"{name} (n = {m.sum()})", rasterized=True)
    if meta["input"] == "logit":
        # standardised logit: z = (v - mean) / sd, so the boundary is linear in the raw logits
        fit = pd.read_csv(REPORT_DIR / "predictions.csv").query("split == 'val'")
        mu = {c: logit(fit[c]).mean() for c in ("p_rf", "p_cnn")}
        sd = {c: logit(fit[c]).std(ddof=0) for c in ("p_rf", "p_cnn")}
        w0, wr, wc = meta["intercept"], meta["weights"]["p_rf"], meta["weights"]["p_cnn"]
        t = np.log(meta["threshold"] / (1 - meta["threshold"]))
        xs = np.linspace(x.min(), x.max(), 100)
        ys = mu["p_cnn"] + sd["p_cnn"] / wc * (t - w0 - wr * (xs - mu["p_rf"]) / sd["p_rf"])
        ax.plot(xs, ys, color=FUSION, lw=2.5, label=f"fusion decision boundary (p = {meta['threshold']:.2f})")
    ax.set(xlim=(np.quantile(x, 0.002) - 0.5, x.max() + 0.3), ylim=(y.min() - 0.3, y.max() + 0.3),
           xlabel="Random forest: logit of fire probability", ylabel="CNN: logit of fire probability",
           title="Test 2021-22: the two base models and the fusion boundary")
    ax.legend(loc="upper left", fontsize=8.5, markerscale=3)
    tf.save(fig, "01_scatter_boundary.png")


def fig_bars(metrics, boot):
    """Test AUPRC of every model, plain and season-matched."""
    models = ["cnn", "rf", "average", "fusion"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(models))
    for k, (evaluation, alpha) in enumerate([("standard", 1.0), ("season_matched", 0.55)]):
        vals = metrics[metrics.evaluation == evaluation].set_index("model").auprc[models]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=[tf.COLOR[m] for m in models], alpha=alpha,
                      label="standard" if k == 0 else "season-matched (lighter)")
        ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
    ax.set_xticks(x, [NAMES[m].replace(" (", "\n(") for m in models])
    ax.set(ylabel="AUPRC on test 2021-22", ylim=(0, 1.12), yticks=np.arange(0, 1.01, 0.2),
           title="Fusion vs the base models")
    rf = boot.set_index("vs").loc["rf"]
    fig.text(0.5, -0.04, f"Fusion minus random forest, 95 % paired-bootstrap interval:  "
             f"standard {rf.standard_mean:+.3f} [{rf.standard_ci_low:+.3f}, {rf.standard_ci_high:+.3f}],  "
             f"season-matched {rf.season_matched_mean:+.3f} [{rf.season_matched_ci_low:+.3f}, "
             f"{rf.season_matched_ci_high:+.3f}]", ha="center", fontsize=8.5)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    tf.save(fig, "03_auprc_bars.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(REPORT_DIR / "predictions.csv")
    test = preds[preds.split == "test"].reset_index(drop=True)
    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    meta = json.loads((REPORT_DIR / "meta_learner.json").read_text())
    probs = {"fusion": test.p_fusion.to_numpy(), "average": test[["p_rf", "p_cnn"]].mean(axis=1).to_numpy(),
             "rf": test.p_rf.to_numpy(), "cnn": test.p_cnn.to_numpy()}
    print("figures ->", FIG_DIR)
    fig_scatter(test, meta)
    tf.fig_roc_pr(test, probs, metrics)                 # -> 02_roc_pr_curves.png
    fig_bars(metrics, pd.read_csv(REPORT_DIR / "bootstrap.csv"))


if __name__ == "__main__":
    main()
