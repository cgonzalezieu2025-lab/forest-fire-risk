# Forest fire risk classification (multi-modal ML)

## Setup
    conda env create -f environment.yml
    conda activate forest-fire-risk
    pip install -e .

## Run tests
    pytest

## Data (Mesogeos Track A)
Pre-extracted tabular dataset from the Mesogeos Google Drive (`ml_tracks/a.danger_forecasting`).
Each sample is a 30-day sequence for one 1 km cell; `positives.csv` ends on a fire day, `negatives.csv` on a random non-fire day.

    python -m forest_fire_risk.download_data      # -> data/raw/track_a/ (~220 MB)
    python -m forest_fire_risk.data               # sanity-print shapes and the year split

Loading in your own code:

    from forest_fire_risk.data import load_track_a, to_tabular, to_sequences, split_by_year, DYNAMIC, STATIC
    df = load_track_a()                       # long: one row per (sample, day), 777k rows
    tab = to_tabular(df, window=7)            # one row per sample, for sklearn / boosting
    X_dyn, X_static, y, meta = to_sequences(df)   # [N,30,13], [N,14], [N] for LSTM / torch
    split = split_by_year(tab)                # 'train' <=2019, 'val' 2020, 'test' 2021-22

Never use `burned_areas`, `ignition_points`, `burned_area_has` as features: they encode the label.

Full datacube (`mesogeos_cube.zarr`), burned-area and ecoregion shapefiles, and Track B samples live in the same Drive folder:
https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9

## EDA
`notebooks/eda_track_a.ipynb` (generated from `eda_track_a.py` with jupytext; figures in `reports/eda/`).
