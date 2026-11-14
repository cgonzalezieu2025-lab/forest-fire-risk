import numpy as np
import pandas as pd
import pytest

from forest_fire_risk import fusion_errors, fusion_robustness


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "label": rng.integers(0, 2, n),
                         "split": np.where(np.arange(n) < n // 2, "val", "test"),
                         "p_rf": rng.random(n), "p_cnn": rng.random(n)})


def test_load_runs_adds_one_column_per_seed(tmp_path):
    df = _df()
    runs = df[["sample_id", "label", "split"]].assign(p_cnn_1=0.1, p_cnn_2=0.2).sample(frac=1, random_state=0)
    runs.to_csv(tmp_path / "runs.csv", index=False)
    out, seeds = fusion_robustness.load_runs(df, tmp_path / "runs.csv")
    assert seeds == [1, 2] and len(out) == len(df) and (out.p_cnn_2 == 0.2).all()


def test_load_runs_rejects_missing_samples(tmp_path):
    df = _df()
    df[["sample_id", "label", "split"]].iloc[:-1].assign(p_cnn_1=0.1).to_csv(tmp_path / "runs.csv", index=False)
    with pytest.raises(ValueError):
        fusion_robustness.load_runs(df, tmp_path / "runs.csv")


def test_load_runs_rejects_label_mismatch(tmp_path):
    df = _df()
    runs = df[["sample_id", "label", "split"]].assign(p_cnn_1=0.1)
    runs.loc[0, "label"] = 1 - runs.loc[0, "label"]
    runs.to_csv(tmp_path / "runs.csv", index=False)
    with pytest.raises(ValueError):
        fusion_robustness.load_runs(df, tmp_path / "runs.csv")


def test_outcome_and_switches():
    preds = pd.DataFrame({"sample_id": list("abcd"), "label": [1, 1, 0, 0], "split": "test",
                          "time": "2021-07-01", "p_rf": [0.9, 0.1, 0.9, 0.1], "p_fusion": [0.1, 0.9, 0.1, 0.9],
                          "p_cnn": 0.5})
    coords = pd.DataFrame({"sample_id": list("abcd"), "x": 0.0, "y": 0.0})
    t = fusion_errors.label_errors(preds, coords, {"fusion": 0.5, "rf": 0.5})
    assert list(t.outcome_fusion) == ["FN", "TP", "TN", "FP"]
    assert list(t.outcome_rf) == ["TP", "FN", "FP", "TN"]
    assert list(t.switch) == ["broken", "fixed", "fixed", "broken"]


def test_label_errors_needs_every_coordinate():
    preds = pd.DataFrame({"sample_id": ["a", "b"], "label": [1, 0], "split": "test", "time": "2021-07-01",
                          "p_rf": 0.5, "p_cnn": 0.5, "p_fusion": 0.5})
    with pytest.raises(ValueError):
        fusion_errors.label_errors(preds, pd.DataFrame({"sample_id": ["a"], "x": [0.0], "y": [0.0]}),
                                   {"fusion": 0.5, "rf": 0.5})
