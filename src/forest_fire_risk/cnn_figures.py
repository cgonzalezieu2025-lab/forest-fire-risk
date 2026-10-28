"""Figures and checks for reports/cnn/, from the outputs of `python -m forest_fire_risk.cnn`.

    python -m forest_fire_risk.cnn_figures        # -> reports/cnn/figures/, reports/cnn/by_cloud.csv

Reuses the tabular ROC / PR plot so the CNN is drawn next to the random forest and the
month-only baseline on the same test samples.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from forest_fire_risk import tabular_figures as tf
from forest_fire_risk.cnn import REPORT_DIR, tiles
from forest_fire_risk.tabular import REPORT_DIR as TABULAR_DIR
from forest_fire_risk.tabular import load_data, month_rate

plt = tf.plt
FIG_DIR = REPORT_DIR / "figures"
tf.NAMES["cnn"], tf.COLOR["cnn"], tf.STYLE["cnn"] = "CNN (ResNet-18)", "#2e9e5b", "-"
tf.FIG_DIR = FIG_DIR                                # tf.save writes here


def fig_history(history):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(history.epoch, history.train_loss, color=tf.COLOR["cnn"], lw=2, label="training loss")
    ax2 = ax.twinx()
    ax2.plot(history.epoch, history.holdout_auprc, color=tf.FIRE, lw=2, label="hold-out AUPRC")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    best = history.holdout_auprc.idxmax()
    ax2.scatter(history.epoch[best], history.holdout_auprc[best], color=tf.FIRE, zorder=3, s=60,
                label=f"kept: epoch {history.epoch[best]} (AUPRC {history.holdout_auprc[best]:.3f})")
    ax.set(xlabel="Epoch", ylabel="Training loss (binary cross-entropy)", title="CNN training")
    ax2.set_ylabel("AUPRC on the 15 % hold-out of 2017-19")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5)
    tf.save(fig, "01_training.png")


def fig_examples(test, p, n=6):
    """The n most confident correct and wrong predictions of each kind on the test set."""
    df = test.assign(p=p)
    panels = [("Fires: most confident", df[df.label == 1].nlargest(n, "p")),
              ("Fires: missed", df[df.label == 1].nsmallest(n, "p")),
              ("No fire: most confident", df[df.label == 0].nsmallest(n, "p")),
              ("No fire: false alarms", df[df.label == 0].nlargest(n, "p"))]
    fig, axes = plt.subplots(len(panels), n, figsize=(1.9 * n, 2.1 * len(panels)))
    for row, (title, rows) in zip(axes, panels):
        imgs = tiles(rows.sample_id).permute(0, 2, 3, 1).numpy()
        for ax, img, (_, r) in zip(row, imgs, rows.iterrows()):
            ax.imshow(img)
            ax.set_title(f"p={r.p:.2f}  {r.time:%b %Y}\ncloud {r.cloud:.0f}%", fontsize=7.5)
            ax.axis("off")
        row[0].text(-0.1, 0.5, title, transform=row[0].transAxes, rotation=90, va="center",
                    ha="right", fontsize=9, fontweight="bold")
    fig.suptitle("Test tiles (2.56 km, 2021-22) and the CNN's fire probability", fontweight="bold")
    tf.save(fig, "03_example_tiles.png")


def by_cloud(test, probs):
    """Scores per scene-cloud bucket: does the CNN lean on cloud cover (a hint for 'no fire')?"""
    bucket = pd.cut(test.cloud, [-0.01, 1, 5, 20], labels=["0-1 %", "1-5 %", "5-20 %"])
    rows = []
    for b, g in test.groupby(bucket, observed=True):
        for name, p in probs.items():
            pb = p[g.index.map(test.index.get_loc)]
            rows.append({"cloud": b, "n": len(g), "fire_share": g.label.mean(), "model": name,
                         "auprc": average_precision_score(g.label, pb), "roc_auc": roc_auc_score(g.label, pb),
                         "mean_p": pb.mean()})
    return pd.DataFrame(rows)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tab = load_data()
    train, test = tab[tab.split == "train"], tab[tab.split == "test"]
    preds = pd.read_csv(REPORT_DIR / "predictions.csv").merge(
        pd.read_csv(TABULAR_DIR / "predictions.csv")[["sample_id", "p_rf"]], on="sample_id")
    preds = test[["sample_id"]].merge(preds, on="sample_id")       # test rows, in test order
    assert (preds.sample_id.to_numpy() == test.sample_id.to_numpy()).all()
    probs = {"cnn": preds.p_cnn.to_numpy(), "rf": preds.p_rf.to_numpy(), "month": month_rate(train, test)}
    metrics = pd.concat([pd.read_csv(REPORT_DIR / "metrics.csv"), pd.read_csv(TABULAR_DIR / "metrics.csv")])

    print("figures ->", FIG_DIR)
    fig_history(pd.read_csv(REPORT_DIR / "history.csv"))
    tf.fig_roc_pr(test, probs, metrics)
    fig_examples(test, probs["cnn"])
    cloud = by_cloud(test, {"cnn": probs["cnn"], "rf": probs["rf"]})
    cloud.to_csv(REPORT_DIR / "by_cloud.csv", index=False)
    print(cloud.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
