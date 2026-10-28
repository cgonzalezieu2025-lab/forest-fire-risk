# Forest fire risk classification (multi-modal ML)

Can we tell whether a 1 km patch of the Mediterranean will burn on a given day?
We predict a binary fire / no-fire label from two kinds of input and compare them:

1. **Tabular drivers** (weather, vegetation, soil, terrain, land cover, human activity) from the
   [Mesogeos](https://github.com/Orion-AI-Lab/mesogeos) datacube (NeurIPS 2023). Classic ML and a sequence model.
2. **Satellite images** of the same patch from Sentinel-2, pulled via Google Earth Engine. A CNN.

Then a fused model that uses both. Mesogeos covers 2006–2022 and Sentinel-2 starts in 2017,
so **every model, the tabular ones included, is trained and tested on the 2017–2022 samples that
have a Sentinel-2 tile**. That way all models are compared on exactly the same samples, and
fusion has a tabular and an image prediction for each of them.

## How it fits together

    Mesogeos Track A (CSV)  ──►  data.py ──► tabular rows / 30-day sequences ──► tabular + LSTM models
            │ sample_id, lon/lat, date                                                  │
            ▼                                                                           ▼
    gee_images.py ──► Google Earth Engine ──► data/images/<sample_id>.png ──► CNN ──► fusion model
            │
            ▼
    app/build_data.py ──► app/ (map viewer, https://forest-fire-risk-map.vercel.app)

Everything is keyed by `sample_id`, so a tabular row, its 30-day sequence, its satellite tile and
its dot on the map are the same sample. One year-based split (`split_by_year`) is used by every
branch, so the models are compared on exactly the same test samples.

Status (Oct 2026): data loaders, EDA, satellite tiles for all 2017–2022 samples, the map viewer and
the tabular models (logistic regression, random forest; see `reports/tabular/README.md`) and a first CNN
(`reports/cnn/README.md`) are done. Fusion is next (see `docs/model_plan.md`).

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

### Sentinel-2 images (Google Earth Engine)
One RGB tile per sample in 2017-2022 (3,968 fires, 8,025 non-fires). Needs a Google account with a
Cloud project that has the Earth Engine API enabled (https://developers.google.com/earth-engine/guides/access).

    earthengine authenticate                      # once, opens a browser
    export EE_PROJECT=<your-cloud-project-id>
    python -m forest_fire_risk.gee_images --limit 20   # try a few, look at data/images/
    python -m forest_fire_risk.gee_images              # all ~12k; resumable, ~1 h with 8 workers

Each tile is the least-cloudy `COPERNICUS/S2_SR_HARMONIZED` scene in the 15 days **before** the
window's end date, 256x256 px at 10 m (2.56 km around the cell centre), saved as
`data/images/<sample_id>.png`. `data/images/index.csv` records the scene id, cloud % and date per
sample, or `status=missing` when nothing usable was found.

Result of the full run (Oct 2026): 11,579 tiles, 414 missing (3.5%). Most gaps are 2017
negatives from before L2A coverage started (361 of 2,365); fires miss only 1-9% per year.
Scene cloud % is low (median 0.2%, 95th percentile 23%); 644 tiles come from scenes >20% cloudy.
`with_images(meta)` in `data.py` joins the index onto the tabular / sequence meta and adds
`image_path`, so the image and tabular branches use the same rows and the same year split.

**Without the images:** to know *which* samples have a tile you don't need GEE or the 1.7 GB of
tiles. If `data/images/index.csv` is missing, `tile_index()` / `with_images()` fall back to
`data/tile_index_from_map.csv`, downloaded once from the map viewer's data (built from the same
`index.csv`, same 11,579 ok / 414 missing). The fallback lives outside `data/images/` on purpose:
`gee_images.py` treats rows in `data/images/index.csv` as already downloaded.

**Leakage warning:** never use images from the fire day or later. Smoke and burn scars are the
label. The lookback window stops the day before day 29.

### Other files in the Mesogeos Drive
The full datacube (`mesogeos_cube.zarr`, large), burned-area and ecoregion shapefiles, and the
Track B burned-area samples are in the same folder; `explore_mesogeos.py` prints the cube's
structure if you download it.
https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9

## Map viewer
Live at **https://forest-fire-risk-map.vercel.app** (static, 128 px thumbnails). Every 2017–2022
sample on a map, coloured by label or by a feature, with filters for year, label and scene cloud.
Click a point for its Sentinel-2 tile, the features on the window's last day, and a Python snippet
that loads that sample. The sidebar gives a gallery of the selected tiles, a CSV download, and the
pandas code that reproduces the selection. A walkthrough runs on first visit (`?` button replays it).

Locally the page shows full-res tiles if `data/images/` is there. Source is `app/`:

    python app/build_data.py                      # -> app/samples.json + app/sprites/ (git-ignored)
    python -m http.server 8000                    # from the repo root, open http://localhost:8000/app/
    cd app && vercel deploy --prod                # redeploy after rebuilding the data

### Loading images with the data

    from forest_fire_risk.data import load_track_a, to_tabular, with_images, split_by_year, load_tile, load_tiles
    from forest_fire_risk.images import TileDataset

    tab = with_images(to_tabular(load_track_a(), window=7))   # rows with a tile + image_path / cloud / date
    split = split_by_year(tab)
    tile = load_tile("pos_1611")                              # [256, 256, 3] uint8
    tiles = load_tiles(tab[split == "val"])                   # [N, 256, 256, 3], ~200 KB per tile
    ds = TileDataset(tab[split == "train"])                   # torch-style: (img [3,256,256], static [14], label)

## Tabular models (logistic regression, random forest)
Full write-up with all figures: **`reports/tabular/README.md`**. Code: `src/forest_fire_risk/tabular.py`.

    python -m forest_fire_risk.tabular            # tune + train + evaluate, ~2 min -> reports/tabular/
    python -m forest_fire_risk.side_experiments   # optional extras, not used by fusion
    python -m forest_fire_risk.tabular_figures    # figures -> reports/tabular/figures/

The 27 fire-day features of the 10,922 samples that have a tile from a scene with ≤ 20 % cloud
(the cloud rule is expected to match the CNN; not confirmed yet, it's `MAX_CLOUD` in `tabular.py`).

| Years | Role | Used to |
|---|---|---|
| 2017–2019 | train | fit; leave-one-year-out cross-validation (k = 3) picks hyperparameters and the threshold |
| 2020 | fusion only | the final LR / RF only *predict* it; never used to fit or tune them |
| 2021–2022 | test | scored once, at the end |

Test 2021–22: random forest AUPRC 0.829 (ROC-AUC 0.899), logistic regression 0.710 (0.819), month-only
baseline 0.554. Season-matched (the "fire season" shortcut removed): RF 0.731, LR 0.604, chance 0.351.
`reports/tabular/predictions.csv` holds the 2020 and 2021–22 probabilities per `sample_id` for fusion.

## Image model (CNN)
Full write-up with figures: **`reports/cnn/README.md`**. Code: `src/forest_fire_risk/cnn.py`.

    python -m forest_fire_risk.cnn                # cache tiles, train on the GPU (~7 min), evaluate -> reports/cnn/
    python -m forest_fire_risk.cnn_figures        # figures -> reports/cnn/figures/

ResNet-18 (ImageNet weights) fine-tuned on the RGB tiles of the same 10,922 samples and the same
years as the tabular models (2017–2019 train with a 15 % hold-out for early stopping, 2020 untouched,
2021–22 test). Test 2021–22: AUPRC 0.749 (ROC-AUC 0.843); season-matched 0.705. Between the logistic
regression and the random forest. `reports/cnn/predictions.csv` holds `p_cnn` for 2020 and 2021–22.

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
        gee_images.py       fetch Sentinel-2 tiles from Google Earth Engine
        images.py           TileDataset for the CNN branch
        tabular.py          logistic regression + random forest (main tabular pipeline)
        side_experiments.py tabular "what if" experiments, not used by fusion
        tabular_figures.py  figures for reports/tabular/
        cnn.py              ResNet-18 on the Sentinel-2 tiles (image branch)
        cnn_figures.py      figures and cloud-cover check for reports/cnn/
    app/                    map viewer (index.html + build_data.py), deployed on Vercel
    docs/model_plan.md      modelling strategy: LR baseline -> RF -> CNN -> LR meta-learner fusion
    notebooks/              EDA (jupytext .py + executed .ipynb)
    reports/eda/            EDA figures
    reports/tabular/        tabular models: written report, metrics, predictions for fusion, figures
    reports/cnn/            CNN: written report, metrics, predictions for fusion, figures
    models/                 fitted models, git-ignored (re-created by the pipelines)
    tests/                  pytest sanity checks (synthetic data, no download needed)
    data/                   raw downloads, git-ignored

## Done so far
- [x] Track A download + loaders (`data.py`), year split, leakage-safe feature lists, tests
- [x] EDA notebook and findings (above)
- [x] Sentinel-2 tiles for all 2017–2022 samples via Earth Engine (`gee_images.py`), 11,579 tiles
- [x] Image loaders (`with_images`, `load_tile`, `load_tiles`, `TileDataset`)
- [x] Map viewer with walkthrough, gallery, CSV export and code snippets, deployed on Vercel
- [x] Tabular models: logistic regression baseline and random forest on the tabular + Sentinel-2
      intersection, leave-one-year-out CV, season-matched evaluation, report (`reports/tabular/`)
- [x] CNN (ResNet-18) on the tiles, same samples and years, cloudy-tile rule confirmed, report (`reports/cnn/`)

## Next steps
The plan and reasoning are in `docs/model_plan.md`; build order:
1. ~~Tabular baseline and random forest~~ done, see `reports/tabular/README.md`.
2. LSTM on `to_sequences`, compare with the Mesogeos numbers above.
3. ~~CNN on the tiles~~ first version done, see `reports/cnn/README.md`. Cloudy-tile rule confirmed
   (tiles above 20 % cloud are dropped by every model). Still to try: less overfitting, year-wise CV.
4. Fusion model on `reports/tabular/predictions.csv` (`p_rf`) + `reports/cnn/predictions.csv` (`p_cnn`): train on 2020,
   evaluate on 2021–22, standard and season-matched.
