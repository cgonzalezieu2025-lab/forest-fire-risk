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


def test_gee_window_excludes_fire_day():
    from forest_fire_risk import gee_images
    start, stop = gee_images.date_window("2021-07-30 00:00:00")
    assert (start, stop) == ("2021-07-15", "2021-07-30")       # stop is exclusive -> day 29 never included
    meta = pd.DataFrame({"time": ["2016-08-01", "2017-03-01", "2022-10-31", "2023-01-01"]})
    assert gee_images.select_samples(meta)["time"].tolist() == ["2017-03-01", "2022-10-31"]


def test_with_images(tmp_path):
    pd.DataFrame({"sample_id": ["pos_0", "pos_1"], "status": ["ok", "missing"],
                  "image": ["a", ""], "cloud": [1.0, None], "date": ["2021-07-01", ""]}).to_csv(tmp_path / "index.csv", index=False)
    meta = pd.DataFrame({"sample_id": ["pos_0", "pos_1", "neg_0"], "label": [1, 1, 0]})
    out = data.with_images(meta, tmp_path)
    assert out.sample_id.tolist() == ["pos_0"] and out.image_path[0].endswith("pos_0.png")


def test_load_tiles(tmp_path):
    from PIL import Image
    Image.new("RGB", (256, 256), (10, 20, 30)).save(tmp_path / "pos_0.png")
    tiles = data.load_tiles(pd.DataFrame({"sample_id": ["pos_0", "pos_0"]}), tmp_path)
    assert tiles.shape == (2, 256, 256, 3) and tiles.dtype == np.uint8 and tiles[0, 0, 0].tolist() == [10, 20, 30]


def test_tile_index_falls_back_to_map_copy(tmp_path, monkeypatch):
    fallback = tmp_path / "tile_index_from_map.csv"
    pd.DataFrame({"sample_id": ["pos_0", "neg_0"], "status": ["ok", "missing"],
                  "cloud": [3.0, None], "date": ["2021-07-01", None]}).to_csv(fallback, index=False)
    monkeypatch.setattr(data, "MAP_INDEX", fallback)
    images = tmp_path / "images"                                  # no index.csv here -> fallback
    images.mkdir()
    assert data.tile_index(images).sample_id.tolist() == ["pos_0", "neg_0"]
    meta = pd.DataFrame({"sample_id": ["pos_0", "neg_0"], "label": [1, 0]})
    assert data.with_images(meta, images).sample_id.tolist() == ["pos_0"]
