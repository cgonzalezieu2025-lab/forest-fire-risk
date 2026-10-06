# Forest fire risk classification (multi-modal ML)

Can we tell whether a 1 km patch of the Mediterranean will burn on a given day?
We predict a binary fire / no-fire label from two kinds of input and compare them:

1. **Tabular drivers** (weather, vegetation, soil, terrain, land cover, human activity) from the
   [Mesogeos](https://github.com/Orion-AI-Lab/mesogeos) datacube (NeurIPS 2023). Classic ML and a sequence model.
2. **Satellite images** of the same patch from Sentinel-2, pulled via Google Earth Engine. A CNN.

Then a fused model that uses both. Mesogeos covers 2006–2022 and Sentinel-2 starts in 2017,
so the image branch and the fusion work on the 2017–2022 overlap.

## Setup
    conda env create -f environment.yml
    conda activate forest-fire-risk
    pip install -e .
    pytest                                        # quick sanity tests

## Data

### Track A tabular data (Mesogeos)
The Mesogeos authors ship a pre-extracted dataset for exactly this task ("Track A: fire danger
forecasting"), so we do not slice the raw datacube ourselves.

    python -m forest_fire_risk.download_data      # -> data/raw/track_a/ (~220 MB, git-ignored)
    python -m forest_fire_risk.data               # prints shapes and the year split

What is in it:

| | |
|---|---|
| Samples | 25,916 (8,574 fire, 17,342 no-fire, ratio 1:2) |
| Period | 2006–2022, fire season only (Mar–Oct) |
| Region | whole Mediterranean basin, 1 km cells (`x` = lon, `y` = lat) |
| Shape | **each sample is a 30-day sequence**, one row per day (`time_idx` 0–29); day 29 is the fire day for positives |
| Dynamic features (13) | `t2m`, `d2m`, `rh`, `tp`, `sp`, `ssrd`, `wind_speed`, `wind_direction` (ERA5-Land weather); `lst_day`, `lst_night`, `ndvi`, `lai` (MODIS); `smi` (soil moisture) |
| Static features (14) | `dem`, `slope`, `aspect`, `curvature`, 8 land-cover fractions `lc_*`, `population`, `roads_distance` |
| Label | fire (1) vs no-fire (0), from EFFIS burned-area polygons |

**Leakage warning:** `burned_areas`, `ignition_points` and `burned_area_has` are the label in
disguise. The loader exposes `DYNAMIC` / `STATIC` lists that exclude them. Never add them back.

### Loading it in your own code

    from forest_fire_risk.data import load_track_a, to_tabular, to_sequences, split_by_year

    df = load_track_a()                            # long: one row per (sample, day), 777k rows
    tab = to_tabular(df, window=7)                 # one row per sample -> sklearn / boosting
    X_dyn, X_static, y, meta = to_sequences(df)    # [N,30,13], [N,14], [N] -> LSTM / torch
    split = split_by_year(tab)                     # 'train' <=2019, 'val' 2020, 'test' 2021-22

`split_by_year` is the one split everybody uses, so the tabular and image branches are evaluated
on the same samples. Never split randomly by row: the 30 rows of a sample are near-duplicates.

### Sentinel-2 images (to do)
For every sample in 2017–2022 (3,968 fires, 8,025 non-fires) we will query Google Earth Engine
for the least-cloudy `COPERNICUS/S2_SR_HARMONIZED` image within ±5 days of the sample's end date
at its lon/lat, and save an RGB tile named by `sample_id`. The `meta` frame from `to_sequences`
has everything the query needs (`sample_id`, `time`, `x`, `y`).

### Other files in the Mesogeos Drive
The full datacube (`mesogeos_cube.zarr`, large), burned-area and ecoregion shapefiles, and the
Track B burned-area samples are in the same folder; `explore_mesogeos.py` prints the cube's
structure if you download it.
https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9

## EDA
`notebooks/eda_track_a.ipynb` (source: `notebooks/eda_track_a.py`, converted with jupytext;
figures in `reports/eda/`). Main findings so far:

- Fire days are hotter and drier (`t2m`, `lst_*`, `ssrd` up; `rh`, `smi`, `tp` down) and the gap
  widens over the last ~10 days of the window, so the sequence carries signal beyond the last day.
  Wind speed barely differs.
- Negatives are spread evenly over Mar–Oct while fires peak in Jul–Aug. The sampling is not
  season-matched, so a model can score well just by learning "hot month". Keep this in mind
  when reading metrics; resampling negatives to the fire-month distribution is an option.
- MODIS land-surface temperature is missing far more often on no-fire days (fire days are
  cloud-free by construction). Impute before training so the model does not learn the gap itself.
- Strong correlations inside the temperature family (`t2m`, `lst_*`, `d2m`), `sp` with `dem`,
  `lai` with `ndvi`. Expect multicollinearity in linear models.
- Fire size is heavy-tailed (median ~150 ha, max ~108k ha). Italy has the most fire samples.
- `slope` values sit in a narrow 1.44–1.57 band for nearly all cells and look mis-scaled.

Mesogeos baselines on Track A for reference: LSTM F1 0.79, Transformer F1 0.79 / AUPRC 0.87.

## Repo layout
    src/forest_fire_risk/
        download_data.py    fetch Track A from Google Drive
        data.py             loaders: long table, tabular view, sequence arrays, year split
        explore_mesogeos.py inspect the raw zarr datacube (optional)
    notebooks/              EDA (jupytext .py + executed .ipynb)
    reports/eda/            EDA figures
    tests/                  pytest sanity checks (synthetic data, no download needed)
    data/                   raw downloads, git-ignored

## Next steps
1. Tabular baseline (logistic regression, gradient boosting) on `to_tabular` with the year split.
2. LSTM on `to_sequences`, compare with the Mesogeos numbers above.
3. GEE script to pull Sentinel-2 tiles for the 2017–2022 samples; CNN on the tiles.
4. Fusion model and final comparison.
