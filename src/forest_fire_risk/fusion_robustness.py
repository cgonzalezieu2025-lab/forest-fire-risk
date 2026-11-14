"""Robustness checks for fusion: one refit per CNN training, and scores per test year.

    python -m forest_fire_risk.fusion_robustness     # ~1 min -> reports/fusion/

Needs only committed files: reports/tabular/predictions.csv, reports/cnn/predictions.csv,
reports/cnn/predictions_runs.csv and reports/cnn/runs.csv. Does not change any existing
output of `python -m forest_fire_risk.fusion`.

1. Per CNN run (`per_run.csv`, `per_run_summary.csv`). The CNN input of the main fusion is the
   average of five trainings (seeds 42-46). Here the meta-learner is refitted on 2020 once per
   single training, exactly as in `fusion.main` (same grid, same CV, threshold from 2020), and
   scored on 2021-22. The spread over the five rows is the training noise the test-sample
   bootstrap in `bootstrap.csv` does not cover.
2. Per year (`per_year.csv`, `per_year_bootstrap.csv`). The main fusion model (averaged CNN),
   the simple average, the random forest and the CNN, scored on 2021 and on 2022 separately,
   plain and season-matched (season weights recomputed inside each year), with the paired
   bootstrap of fusion minus RF inside each year. Thresholds are the ones of `fusion.main`;
   nothing is chosen on the test years.
"""
import json

import pandas as pd

from forest_fire_risk import fusion
from forest_fire_risk.fusion import INPUTS, REPORT_DIR, fit_meta, load_predictions, paired_bootstrap
from forest_fire_risk.tabular import best_threshold, scores, season_weights

ROOT = fusion.ROOT
CNN_RUNS_PREDS = ROOT / "reports" / "cnn" / "predictions_runs.csv"
CNN_RUNS = ROOT / "reports" / "cnn" / "runs.csv"
EVALUATIONS = ("standard", "season_matched")


def load_runs(df, runs_path=CNN_RUNS_PREDS):
    """Add one column `p_cnn_<seed>` per CNN training to `df` (from `load_predictions`).
    Fails loudly if the runs file does not hold the same samples, labels and splits."""
    runs = pd.read_csv(runs_path)
    cols = [c for c in runs.columns if c.startswith("p_cnn_")]
    out = df.merge(runs[["sample_id", "label", "split"] + cols], on="sample_id", how="left",
                   suffixes=("", "_runs"), validate="one_to_one")
    if len(runs) != len(df) or out[cols].isna().any().any():
        raise ValueError("CNN runs file does not hold exactly the fusion samples")
    for col in ("label", "split"):
        if (out[col] != out[f"{col}_runs"]).any():
            raise ValueError(f"CNN runs and fusion inputs disagree on `{col}`")
    return out.drop(columns=["label_runs", "split_runs"]), [int(c.removeprefix("p_cnn_")) for c in cols]


def score_all(test, probs, thresholds):
    """Rows (model, evaluation, auprc, roc_auc, f1, precision, recall, threshold) on `test`."""
    weights = {"standard": None, "season_matched": season_weights(test.label, test.month)}
    return [{"model": name, "evaluation": ev, **scores(test.label, p, thresholds[name], weights[ev])}
            for ev in EVALUATIONS for name, p in probs.items()]


def refit(fit_rows, test, cnn_threshold, rf_threshold):
    """Fit fusion on 2020 with the current `p_cnn` column; scores on test plus the fitted meta."""
    meta = fit_meta(fit_rows)
    probs = {"fusion": meta["model"].predict_proba(test[INPUTS])[:, 1],
             "average": test[INPUTS].mean(axis=1).to_numpy(),
             "rf": test.p_rf.to_numpy(), "cnn": test.p_cnn.to_numpy()}
    thresholds = {"fusion": meta["threshold"],
                  "average": best_threshold(fit_rows.label, fit_rows[INPUTS].mean(axis=1)),
                  "rf": rf_threshold, "cnn": cnn_threshold}
    return pd.DataFrame(score_all(test, probs, thresholds)), meta, probs, thresholds


def per_run(df, seeds, rf_threshold):
    """One fusion refit per CNN training, plus the averaged CNN of the main fusion for reference."""
    run_thr = pd.read_csv(CNN_RUNS).set_index("seed").threshold
    cnn_thr = json.loads((REPORT_DIR / "meta_learner.json").read_text())["thresholds_used"]["cnn"]
    variants = [(str(s), f"p_cnn_{s}", float(run_thr[s])) for s in seeds] + [("average_of_5", "p_cnn", cnn_thr)]
    rows = []
    for name, col, thr in variants:
        d = df.assign(p_cnn=df[col])
        fit_rows, test = d[d.split == "val"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
        m, meta, _, _ = refit(fit_rows, test, thr, rf_threshold)
        auprc = m.pivot(index="model", columns="evaluation", values="auprc")
        row = {"cnn_run": name, "input": meta["input"], "C": meta["C"], "cv_auprc_2020": meta["cv_auprc"],
               "threshold": meta["threshold"], "w_rf": meta["weights"]["p_rf"], "w_cnn": meta["weights"]["p_cnn"]}
        for ev in EVALUATIONS:
            fus = m[(m.model == "fusion") & (m.evaluation == ev)].iloc[0]
            row |= {f"fusion_auprc_{ev}": fus.auprc, f"fusion_roc_auc_{ev}": fus.roc_auc, f"fusion_f1_{ev}": fus.f1,
                    f"average_auprc_{ev}": auprc.loc["average", ev], f"cnn_auprc_{ev}": auprc.loc["cnn", ev],
                    f"rf_auprc_{ev}": auprc.loc["rf", ev],
                    f"gain_vs_rf_{ev}": fus.auprc - auprc.loc["rf", ev],
                    f"fusion_minus_average_{ev}": fus.auprc - auprc.loc["average", ev]}
        rows.append(row)
        print(f"  CNN run {name:>12}: fusion AUPRC {row['fusion_auprc_standard']:.3f} "
              f"(season-matched {row['fusion_auprc_season_matched']:.3f}), gain vs RF "
              f"{row['gain_vs_rf_standard']:+.3f} / {row['gain_vs_rf_season_matched']:+.3f}, "
              f"w_rf {row['w_rf']:.2f}, w_cnn {row['w_cnn']:.2f}")
    table = pd.DataFrame(rows)
    singles = table[table.cnn_run != "average_of_5"].drop(columns=["cnn_run", "input", "C"])
    summary = singles.agg(["mean", "std", "min", "max"]).T.rename_axis("quantity").reset_index()
    return table, summary


def per_year(df, rf_threshold):
    """Main fusion (averaged CNN) scored inside each test year, with the paired bootstrap vs RF."""
    cnn_thr = json.loads((REPORT_DIR / "meta_learner.json").read_text())["thresholds_used"]["cnn"]
    fit_rows, test = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    _, meta, probs, thresholds = refit(fit_rows, test, cnn_thr, rf_threshold)
    test = test.assign(year=test.time.dt.year, **{f"p_{k}": v for k, v in probs.items() if k in ("fusion", "average")})
    rows, boots = [], []
    for year, g in test.groupby("year"):
        g = g.reset_index(drop=True)
        p = {"fusion": g.p_fusion.to_numpy(), "average": g.p_average.to_numpy(),
             "rf": g.p_rf.to_numpy(), "cnn": g.p_cnn.to_numpy()}
        rows += [{"year": year, "n": len(g), "fires": int(g.label.sum()), **r} for r in score_all(g, p, thresholds)]
        b = paired_bootstrap(g.label, g.month, p["fusion"], p["rf"])
        boots.append({"year": year, **{f"{ev}_{s}": v for ev in EVALUATIONS for s, v in
                                       [("mean", b[ev].mean()), ("ci_low", b[ev].quantile(0.025)),
                                        ("ci_high", b[ev].quantile(0.975)), ("share_positive", (b[ev] > 0).mean())]}})
    return pd.DataFrame(rows), pd.DataFrame(boots)


def main():
    df, seeds = load_runs(load_predictions())
    rf_threshold = json.loads((REPORT_DIR / "meta_learner.json").read_text())["thresholds_used"]["rf"]

    print(f"1. Fusion refitted on each of the {len(seeds)} CNN trainings (seeds {seeds}):")
    table, summary = per_run(df, seeds, rf_threshold)
    print("\nover the single trainings (mean, sd, min, max):")
    print(summary.set_index("quantity").round(3).to_string())

    print("\n2. Scores per test year (main fusion, averaged CNN):")
    years, boots = per_year(df, rf_threshold)
    print(years.pivot_table(index=["year", "model"], columns="evaluation", values="auprc").round(3).to_string())
    print("\nfusion AUPRC minus RF AUPRC per year (paired bootstrap, 95 % interval):")
    print(boots.round(3).to_string(index=False))

    table.to_csv(REPORT_DIR / "per_run.csv", index=False)
    summary.to_csv(REPORT_DIR / "per_run_summary.csv", index=False)
    years.to_csv(REPORT_DIR / "per_year.csv", index=False)
    boots.to_csv(REPORT_DIR / "per_year_bootstrap.csv", index=False)
    print(f"\nwrote per_run.csv, per_run_summary.csv, per_year.csv, per_year_bootstrap.csv in {REPORT_DIR}")


if __name__ == "__main__":
    main()
