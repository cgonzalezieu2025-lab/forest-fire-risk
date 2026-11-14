"""Where fusion is wrong, and where it changes the random forest's decision, on the map.

    python -m forest_fire_risk.fusion_errors     # -> reports/fusion/errors.csv, errors_summary.csv,
                                                 #    figures/04_error_map.png, 05_switch_map.png

Reads reports/fusion/predictions.csv (run `python -m forest_fire_risk.fusion` first) and the
cell coordinates of every sample: from the raw Track A files (data/raw/track_a/) when they are
there, otherwise from the map viewer's samples.json (same samples, no download of the raw data).

Each 2021-22 test sample gets an outcome at fusion's threshold and at the random forest's
threshold (both from reports/fusion/meta_learner.json, picked before the test years):
TP = fire flagged, FN = fire missed, FP = false alarm, TN = no fire, not flagged.
A "switch" is a sample where fusion and the random forest disagree:
    fixed  = fusion right, RF wrong  (a missed fire caught, or a false alarm cleared)
    broken = fusion wrong, RF right
"""
import json

import numpy as np
import pandas as pd

from forest_fire_risk import tabular_figures as tf
from forest_fire_risk.data import DATA_DIR, MAP_URL
from forest_fire_risk.fusion import REPORT_DIR

plt = tf.plt
FIG_DIR = REPORT_DIR / "figures"
OUTCOME_STYLE = {"TN": ("#c9c8c2", "no fire, not flagged"), "TP": ("#2e9e5b", "fire flagged"),
                 "FP": ("#3b6fb6", "false alarm"), "FN": ("#d1495b", "missed fire")}
WEST_OF = 3.5          # longitude splitting Iberia / north-west Africa from the rest of the basin


def coordinates(data_dir=DATA_DIR):
    """sample_id, x (lon), y (lat) for every sample: raw Track A if present, else the map viewer."""
    if (data_dir / "positives.csv").exists():
        from forest_fire_risk.data import load_track_a
        df = load_track_a(data_dir)
        return df.groupby("sample_id", as_index=False)[["x", "y"]].first()
    return pd.read_json(MAP_URL)[["sample_id", "x", "y"]]


def outcome(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return np.select([(y == 1) & pred, (y == 1) & ~pred, (y == 0) & pred], ["TP", "FN", "FP"], "TN")


def label_errors(preds, coords, thresholds):
    """Test rows with coordinates, the fusion and RF outcomes, and the switch type."""
    test = preds[preds.split == "test"].merge(coords, on="sample_id", how="left", validate="one_to_one")
    if test[["x", "y"]].isna().any().any():
        raise ValueError(f"{test.x.isna().sum()} test samples have no coordinates")
    test["outcome_fusion"] = outcome(test.label, test.p_fusion >= thresholds["fusion"])
    test["outcome_rf"] = outcome(test.label, test.p_rf >= thresholds["rf"])
    right_f = test.outcome_fusion.isin(["TP", "TN"])
    right_rf = test.outcome_rf.isin(["TP", "TN"])
    test["switch"] = np.select([right_f & ~right_rf, ~right_f & right_rf], ["fixed", "broken"], "")
    return test


def summarise(test):
    """Counts by outcome for both models, switches by type, and fusion errors by month."""
    rows = []
    for model in ("fusion", "rf"):
        for k, n in test[f"outcome_{model}"].value_counts().items():
            rows.append({"table": "outcome", "model": model, "group": k, "n": n})
    for (sw, of), n in test[test.switch != ""].groupby(["switch", "outcome_fusion"]).size().items():
        rows.append({"table": "switch", "model": "fusion_vs_rf", "group": f"{sw}:{of}", "n": n})
    month = test.assign(month=pd.to_datetime(test.time).dt.month)
    for (m, k), n in month.groupby(["month", "outcome_fusion"]).size().items():
        rows.append({"table": "fusion_by_month", "model": "fusion", "group": f"{m:02d}:{k}", "n": n})
    # Share of fires missed, by fire season (Jun-Sep vs the rest) and by side of the basin
    # (west of 3.5 E = Iberia and north-west Africa, east = everything else).
    fires = month[month.label == 1]
    fires = fires.assign(season=np.where(fires.month.between(6, 9), "jun-sep", "other_months"),
                         side=np.where(fires.x < WEST_OF, "west_of_3.5E", "east_of_3.5E"))
    for by in ("season", "side"):
        for g, d in fires.groupby(by):
            for model in ("fusion", "rf"):
                rows.append({"table": f"miss_rate_by_{by}", "model": model, "group": g,
                             "n": len(d), "rate": (d[f"outcome_{model}"] == "FN").mean()})
    return pd.DataFrame(rows)


def _base(ax, test):
    ax.scatter(test.x, test.y, s=2, color="#e6e5e0", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def fig_error_map(test):
    """Fusion's false alarms and missed fires next to its correct fire calls."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, sharey=True, layout="constrained")
    for ax, kinds, title in [(axes[0], ["TP", "FN"], "Fires: caught vs missed"),
                             (axes[1], ["TN", "FP"], "No-fire samples: false alarms")]:
        _base(ax, test)
        for k in kinds:
            m = test.outcome_fusion == k
            color, name = OUTCOME_STYLE[k]
            ax.scatter(test.x[m], test.y[m], s=6 if k != "TN" else 3, color=color, alpha=0.8,
                       label=f"{name} (n = {m.sum()})", rasterized=True)
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axes[0].set_xlabel("")
    fig.suptitle("Fusion on the 2021-22 test samples, at its threshold")
    tf.save(fig, "04_error_map.png")


def fig_switch_map(test):
    """Where fusion changes the random forest's decision, and whether that helped."""
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    _base(ax, test)
    for sw, color, name in [("fixed", "#2e9e5b", "fusion right, RF wrong"),
                            ("broken", "#d1495b", "fusion wrong, RF right")]:
        m = test.switch == sw
        ax.scatter(test.x[m], test.y[m], s=8, color=color, alpha=0.8, label=f"{name} (n = {m.sum()})")
    ax.set_title("Where fusion and the random forest disagree (2021-22)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    tf.save(fig, "05_switch_map.png")


def main():
    tf.FIG_DIR = FIG_DIR
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((REPORT_DIR / "meta_learner.json").read_text())["thresholds_used"]
    preds = pd.read_csv(REPORT_DIR / "predictions.csv")
    test = label_errors(preds, coordinates(), thresholds)
    summary = summarise(test)
    print(summary[summary.table != "fusion_by_month"].round(3).to_string(index=False))
    cols = ["sample_id", "label", "time", "x", "y", "p_rf", "p_cnn", "p_fusion",
            "outcome_fusion", "outcome_rf", "switch"]
    test[cols].to_csv(REPORT_DIR / "errors.csv", index=False)
    summary.to_csv(REPORT_DIR / "errors_summary.csv", index=False)
    fig_error_map(test)
    fig_switch_map(test)
    print(f"wrote errors.csv, errors_summary.csv and figures/04_error_map.png, 05_switch_map.png in {REPORT_DIR}")


if __name__ == "__main__":
    main()
