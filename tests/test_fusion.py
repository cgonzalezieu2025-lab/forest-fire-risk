import numpy as np
import pandas as pd
import pytest

from forest_fire_risk import fusion


def _fake_preds(tmp_path, n=400, seed=0):
    """Two prediction files like reports/tabular and reports/cnn: each model sees the label with
    its own independent noise, so combining them should beat either one."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    split = np.where(np.arange(n) < n // 2, "val", "test")
    time = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    sig = lambda z: 1 / (1 + np.exp(-z))                           # noqa: E731
    base = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "label": label, "time": time, "split": split})
    tab, cnn = tmp_path / "tab.csv", tmp_path / "cnn.csv"
    base.assign(p_rf=sig(2 * (label - 0.5) * 2 + rng.normal(0, 1.5, n))).to_csv(tab, index=False)
    base.assign(p_cnn=sig(2 * (label - 0.5) * 2 + rng.normal(0, 1.5, n))).sample(frac=1, random_state=1).to_csv(cnn, index=False)
    return tab, cnn


def test_logit_is_finite_at_the_edges():
    z = fusion.logit([0.0, 0.5, 1.0])
    assert np.isfinite(z).all() and z[1] == pytest.approx(0) and z[0] == pytest.approx(-z[2])


def test_load_predictions_joins_by_sample_id(tmp_path):
    tab, cnn = _fake_preds(tmp_path)
    df = fusion.load_predictions(tab, cnn)
    assert len(df) == 400 and {"p_rf", "p_cnn", "month"} <= set(df.columns)
    raw = pd.read_csv(cnn).set_index("sample_id").p_cnn                # cnn file was shuffled
    assert np.allclose(df.set_index("sample_id").p_cnn, raw.loc[df.sample_id])


def test_load_predictions_rejects_mismatched_files(tmp_path):
    tab, cnn = _fake_preds(tmp_path)
    pd.read_csv(cnn).iloc[:-1].to_csv(cnn, index=False)                # one sample missing
    with pytest.raises(ValueError):
        fusion.load_predictions(tab, cnn)


def test_meta_learner_beats_each_base_model(tmp_path):
    df = fusion.load_predictions(*_fake_preds(tmp_path, n=2000))
    fit_rows, test = df[df.split == "val"], df[df.split == "test"]
    meta = fusion.fit_meta(fit_rows)
    assert meta["weights"]["p_rf"] > 0 and meta["weights"]["p_cnn"] > 0
    assert 0 < meta["threshold"] < 1 and len(meta["oof"]) == len(fit_rows)
    from sklearn.metrics import average_precision_score as ap
    p = meta["model"].predict_proba(test[fusion.INPUTS])[:, 1]
    assert ap(test.label, p) > max(ap(test.label, test.p_rf), ap(test.label, test.p_cnn))


def test_paired_bootstrap_is_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y, month, p = rng.integers(0, 2, 200), rng.choice([6, 7, 8], 200), rng.random(200)
    out = fusion.paired_bootstrap(y, month, p, p, n_boot=20)
    assert (out.abs() < 1e-12).all().all()
