"""Mesogeos Track A loaders.

Raw format: positives.csv / negatives.csv, one row per (sample, day). Every sample
is a 30-day window (`time_idx` 0..29) for one 1 km cell; day 29 is the fire day for
positives. `load_track_a` returns that long table; `to_tabular` / `to_sequences`
turn it into model-ready arrays; `split_by_year` gives the Mesogeos train/val/test
split so the tabular and image branches stay aligned.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# repo root / data / raw / track_a
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "track_a"

# these three *are* the label -> never features
LEAK = ["burned_areas", "ignition_points", "burned_area_has"]
DYNAMIC = ["d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi", "sp", "ssrd",
           "t2m", "tp", "wind_direction", "wind_speed"]
STATIC = ["aspect", "curvature", "dem", "roads_distance", "slope", "lc_agriculture",
          "lc_forest", "lc_grassland", "lc_settlement", "lc_shrubland",
          "lc_sparse_vegetation", "lc_water_bodies", "lc_wetland", "population"]
META = ["sample_id", "label", "time", "x", "y"]   # x = lon, y = lat, time = window end


def load_track_a(data_dir=DATA_DIR):
    """Return one row per (sample, day) with a `label` (1 = fire, 0 = no fire)
    and a unique `sample_id`."""
    data_dir = Path(data_dir)
    pos = pd.read_csv(data_dir / "positives.csv", parse_dates=["time"])
    neg = pd.read_csv(data_dir / "negatives.csv", parse_dates=["time"])

    pos["label"] = 1
    neg["label"] = 0

    # `sample` restarts at 0 in each file, so prefix it to make ids unique
    pos["sample_id"] = "pos_" + pos["sample"].astype(int).astype(str)
    neg["sample_id"] = "neg_" + neg["sample"].astype(int).astype(str)

    df = pd.concat([pos, neg], ignore_index=True)
    return df.drop(columns="sample")


def to_tabular(df, window=1):
    """One row per sample for sklearn-style models.

    Columns: META + STATIC + DYNAMIC on the last day (+ `<var>_mean<window>`,
    the mean over the last `window` days, when window > 1).
    """
    df = df.sort_values(["sample_id", "time_idx"])
    last = df[df.time_idx == 29].set_index("sample_id")
    out = last[META[1:] + STATIC + DYNAMIC].copy()
    if window > 1:
        agg = (df[df.time_idx >= 30 - window].groupby("sample_id")[DYNAMIC].mean()
               .add_suffix(f"_mean{window}"))
        out = out.join(agg)
    return out.reset_index()


def to_sequences(df, nan_fill=0.0):
    """Arrays for sequence models (LSTM / transformer).

    Returns (X_dyn [N, 30, 13] float32, X_static [N, 14] float32, y [N] int,
    meta DataFrame with META columns in the same order). NaNs in a sequence are
    filled with that sequence's per-feature mean, then `nan_fill` (as Mesogeos does).
    """
    df = df.sort_values(["sample_id", "time_idx"])
    n = df.sample_id.nunique()
    assert len(df) == n * 30, "every sample must have exactly 30 rows"
    X_dyn = df[DYNAMIC].to_numpy(np.float32).reshape(n, 30, len(DYNAMIC))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN feature in a sequence
        col_mean = np.nanmean(X_dyn, axis=1, keepdims=True)   # [N, 1, 13]
    X_dyn = np.where(np.isnan(X_dyn), col_mean, X_dyn)
    X_dyn = np.nan_to_num(X_dyn, nan=nan_fill)
    last = df[df.time_idx == 29]
    X_static = np.nan_to_num(last[STATIC].to_numpy(np.float32), nan=nan_fill)
    y = last.label.to_numpy()
    return X_dyn, X_static, y, last[META].reset_index(drop=True)


def split_by_year(meta, val=(2020,), test=(2021, 2022)):
    """'train' / 'val' / 'test' per row, by the window's end year (Mesogeos split).
    Pass the frame from `to_tabular` or the meta from `to_sequences`."""
    year = pd.to_datetime(meta["time"]).dt.year
    return pd.Series(np.select([year.isin(test), year.isin(val)], ["test", "val"], "train"),
                     index=meta.index, name="split")


def main():
    df = load_track_a()
    print("long:", df.shape, "| samples:", df.sample_id.nunique())
    tab = to_tabular(df, window=7)
    print("tabular:", tab.shape)
    X_dyn, X_static, y, meta = to_sequences(df)
    print("sequences:", X_dyn.shape, X_static.shape, y.shape, "| NaNs left:", int(np.isnan(X_dyn).sum()))
    print(pd.crosstab(split_by_year(meta), y))


if __name__ == "__main__":
    main()
