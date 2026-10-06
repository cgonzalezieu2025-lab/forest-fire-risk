import numpy as np
import pandas as pd
import pytest

from forest_fire_risk import data


def _fake(tmp_path):
    """Two tiny CSVs in the real layout: 2 positive + 1 negative samples x 30 days."""
    def block(n_samples, label):
        rows = []
        for s in range(n_samples):
            for t in range(30):
                r = {c: float(t) for c in data.DYNAMIC}
                r.update({c: 1.0 for c in data.STATIC}); r.update({c: 0.0 for c in data.LEAK})
                r.update(time=f"2021-07-{t+1:02d}", time_idx=t, sample=s, x=10.0, y=40.0)
                if label and t == 3: r["ndvi"] = np.nan          # a hole to fill
                rows.append(r)
        return pd.DataFrame(rows)
    block(2, 1).to_csv(tmp_path / "positives.csv", index=False)
    block(1, 0).to_csv(tmp_path / "negatives.csv", index=False)
    return tmp_path


def test_shapes_and_nan_fill(tmp_path):
    df = data.load_track_a(_fake(tmp_path))
    assert df.sample_id.nunique() == 3 and len(df) == 90

    tab = data.to_tabular(df, window=7)
    assert len(tab) == 3 and tab["t2m"].tolist() == [29.0] * 3
    assert tab["t2m_mean7"].iloc[0] == pytest.approx(np.mean(range(23, 30)))

    X_dyn, X_static, y, meta = data.to_sequences(df)
    assert X_dyn.shape == (3, 30, len(data.DYNAMIC)) and X_static.shape == (3, len(data.STATIC))
    assert not np.isnan(X_dyn).any() and sorted(y) == [0, 1, 1]
    assert set(data.split_by_year(meta)) == {"test"}
