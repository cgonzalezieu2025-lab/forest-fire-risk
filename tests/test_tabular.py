import numpy as np
import pandas as pd
import pytest

from forest_fire_risk import tabular


def _fake_tab(n=300, seed=0):
    """Synthetic one-row-per-sample table: 3 train years, fire driven by t2m and by month."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(tabular.FEATURES))), columns=tabular.FEATURES)
    df["year"] = rng.choice([2017, 2018, 2019], n)
    df["month"] = rng.choice([4, 7, 8], n)
    logit = 2 * df.t2m + (df.month > 6) - 0.7
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[::17, "ndvi"] = np.nan                                  # holes for the imputer
    return df


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    assert tabular.best_threshold(y, p) == pytest.approx(0.35)    # catches all fires, one false alarm


def test_season_weights_make_month_uninformative():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    month = np.array([8, 8, 7, 8, 7, 4, 4, 5])
    w = tabular.season_weights(y, month)
    assert (w[y == 1] == 1).all()
    assert w[y == 0].sum() == pytest.approx((y == 0).sum())       # total negative weight kept
    assert (w[np.isin(month, [4, 5]) & (y == 0)] == 0).all()       # months with no fire drop out
    neg = pd.Series(w[y == 0], index=month[y == 0]).groupby(level=0).sum()
    assert (neg / neg.sum()).to_dict() == pytest.approx({4: 0, 5: 0, 7: 1 / 3, 8: 2 / 3})


def test_month_baseline_is_chance_when_season_matched():
    df = _fake_tab(600)
    p = tabular.month_rate(df, df)
    s = tabular.scores(df.label, p, 0.5, tabular.season_weights(df.label, df.month))
    assert s["roc_auc"] == pytest.approx(0.5)


@pytest.mark.parametrize("make, grid", [(tabular.make_lr, {"logisticregression__C": [0.1, 1]}),
                                        (tabular.make_rf, {"randomforestclassifier__max_depth": [3]})])
def test_fit_uses_leave_one_year_out(make, grid):
    df = _fake_tab()
    out = tabular.fit(make(), grid, df)
    assert len(out["cv_results"].filter(like="split").filter(like="test_score").columns) == 3   # one fold per year
    assert len(out["oof"]) == len(df) and 0 < out["threshold"] < 1
    assert out["cv_auprc"] > df.label.mean()                       # learns the t2m signal
