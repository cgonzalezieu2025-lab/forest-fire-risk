# Forest fire risk classification (multi-modal ML)

Can we tell whether a 1 km patch of the Mediterranean will burn on a given day?
We predict a binary fire / no-fire label from two kinds of input and compare them:

1. **Tabular drivers** (weather, vegetation, soil, terrain, land cover, human activity) from the
   [Mesogeos](https://github.com/Orion-AI-Lab/mesogeos) datacube (NeurIPS 2023). Logistic regression and a random forest.
2. **Satellite images** of the same patch from Sentinel-2, pulled via Google Earth Engine. A CNN.

Then a fused model that uses both. Mesogeos covers 2006–2022 and Sentinel-2 starts in 2017,
so **every model, the tabular ones included, is trained and tested on the 2017–2022 samples that
have a Sentinel-2 tile**. That way all models are compared on exactly the same samples, and
fusion has a tabular and an image prediction for each of them.

## How it fits together

    Mesogeos Track A (CSV)  ──►  data.py ──► tabular rows ──► logistic regression + random forest
            │ sample_id, lon/lat, date                                                  │
            ▼                                                                           ▼
    gee_images.py ──► Google Earth Engine ──► data/images/<sample_id>.png ──► CNN ──► fusion model
            │
            ▼
    app/build_data.py ──► app/ (map viewer, https://forest-fire-risk-map.vercel.app)

Everything is keyed by `sample_id`, so a tabular row, its 30-day sequence, its satellite tile and
its dot on the map are the same sample. One year-based split (`split_by_year`) is used by every
branch, so the models are compared on exactly the same test samples.

Status (Nov 2026): data loaders, EDA, satellite tiles for all 2017–2022 samples, the map viewer and
the tabular models (logistic regression, random forest; see `reports/tabular/README.md`), a first CNN
(`reports/cnn/README.md`) and the fusion model (`reports/fusion/README.md`) are done (see `docs/model_plan.md`).
Since the progress report the tabular models use the weather of the day before the fire (t-1)
instead of the fire day itself; all tabular and fusion numbers below are the t-1 ones.

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
| Period | 2006–2022, every month (most samples Mar–Oct) |
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
    X_dyn, X_static, y, meta = to_sequences(df)    # [N,30,13], [N,14], [N]; includes the fire day, no model uses it
    split = split_by_year(tab)                     # 'train' <=2019, 'val' 2020, 'test' 2021-22

`split_by_year` is the one split everybody uses, so the tabular and image branches are evaluated
on the same samples. Never split randomly by row: the 30 rows of a sample are near-duplicates.

### Sentinel-2 images (Google Earth Engine)
One RGB tile per sample in 2017-2022 (3,968 fires, 8,025 non-fires).

**Quick way, no Earth Engine account:** the exact tiles we used are a file on the repo's
`tiles-v1` release.

    python -m forest_fire_risk.download_images    # -> data/images/ (1.7 GB: 11,579 PNGs + index.csv)

**From scratch:** needs a Google account with a Cloud project that has the Earth Engine API
enabled (https://developers.google.com/earth-engine/guides/access).

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
Scene cloud % is low (median 0.2%, 95th percentile 23%); 657 tiles come from scenes >20% cloudy.
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

27 features (13 dynamic ones from day t-1, the day before the fire, never the fire day; 14 static)
of the 10,922 samples that have a tile from a scene with ≤ 20 % cloud
(the same cloud rule as the CNN, agreed with the image branch; it's `MAX_CLOUD` in `tabular.py`).

| Years | Role | Used to |
|---|---|---|
| 2017–2019 | train | fit; leave-one-year-out cross-validation (k = 3) picks hyperparameters and the threshold |
| 2020 | fusion only | the final LR / RF only *predict* it; never used to fit or tune them |
| 2021–2022 | test | scored once, at the end |

Test 2021–22: random forest AUPRC 0.811 (ROC-AUC 0.888), logistic regression 0.695 (0.807), month-only
baseline 0.554. Season-matched (the "fire season" shortcut removed): RF 0.707, LR 0.583, chance 0.351.
`reports/tabular/predictions.csv` holds the 2020 and 2021–22 probabilities per `sample_id` for fusion.

## Image model (CNN)
Full write-up with figures: **`reports/cnn/README.md`**. Code: `src/forest_fire_risk/cnn.py`.

    python -m forest_fire_risk.cnn                # cache tiles, train 5 seeds on the GPU (~20 min), evaluate -> reports/cnn/
    python -m forest_fire_risk.cnn_figures        # figures -> reports/cnn/figures/

ResNet-18 (ImageNet weights) fine-tuned on the RGB tiles of the same 10,922 samples and the same
years as the tabular models (2017–2019 train: 2019 is held out to pick the stopping epoch and the
threshold, then the model is refit on all three years; 2020 untouched; 2021–22 test). GPU training is
noisy, so it is trained five times and the reported CNN is the average of the five: test AUPRC 0.760
(ROC-AUC 0.850); season-matched 0.723. A single training gets 0.714 ± 0.019. Between the logistic
regression and the random forest. `reports/cnn/predictions.csv` holds `p_cnn` for 2020 and 2021–22.

## Fusion (random forest + CNN)
Full write-up with figures: **`reports/fusion/README.md`**. Code: `src/forest_fire_risk/fusion.py`.

    python -m forest_fire_risk.fusion             # ~10 s, reads the two predictions.csv files -> reports/fusion/
    python -m forest_fire_risk.fusion_figures     # figures -> reports/fusion/figures/

Logistic-regression meta-learner on the logits of `p_rf` and `p_cnn`, fitted on 2020 (where neither base
model trained), settings and threshold from 5-fold CV inside 2020. Test 2021–22: AUPRC 0.853 (ROC-AUC 0.911),
against 0.811 for the random forest and 0.760 for the CNN; season-matched 0.782 against 0.707. Both gains over
the random forest are clearly above zero in a paired bootstrap (+0.042 and +0.075).

## EDA
`notebooks/eda_track_a.ipynb` (source: `notebooks/eda_track_a.py`, converted with jupytext;
figures in `reports/eda/`). Main findings so far:

- Fire days are hotter and drier (`t2m`, `lst_*`, `ssrd` up; `rh`, `smi`, `tp` down) and the gap
  widens over the last ~10 days of the window, so the sequence carries signal beyond the last day.
  Wind speed barely differs.
- Negatives are spread fairly evenly over Mar–Oct (with a few in every month) while fires peak in
  Jul–Aug. The sampling is not
  season-matched, so a model can score well just by learning "hot month". Keep this in mind
  when reading metrics; every model is also scored season-matched (test negatives re-weighted to
  the fires' month mix, see `reports/tabular/README.md` section 7).
- MODIS land-surface temperature is missing far more often on no-fire days (fire days are
  cloud-free by construction). Impute before training so the model does not learn the gap itself.
- Strong correlations inside the temperature family (`t2m`, `lst_*`, `d2m`), `sp` with `dem`,
  `lai` with `ndvi`. Expect multicollinearity in linear models.
- Fire size is heavy-tailed (median ~150 ha, max ~108k ha). Italy has the most fire samples.
- `slope` values sit in a narrow 1.44–1.57 band for nearly all cells and look mis-scaled.

Mesogeos baselines on Track A for reference (Kondylatos et al. 2023, Table 1): LSTM F1 0.79 / AUPRC 0.853,
Transformer F1 0.78 / AUPRC 0.856, GTN F1 0.79 / AUPRC 0.858. They use the 30 days before the fire,
train on 2006–2019 and test on all 2021–22 samples, so they are not directly comparable to ours.

## Repo layout
    src/forest_fire_risk/
        download_data.py    fetch Track A from Google Drive
        data.py             loaders: long table, tabular view, sequence arrays, year split
        explore_mesogeos.py inspect the raw zarr datacube (optional)
        gee_images.py       fetch Sentinel-2 tiles from Google Earth Engine
        download_images.py  or download the same tiles from the GitHub release
        images.py           TileDataset for the CNN branch
        tabular.py          logistic regression + random forest (main tabular pipeline)
        side_experiments.py tabular "what if" experiments, not used by fusion
        tabular_figures.py  figures for reports/tabular/
        cnn.py              ResNet-18 on the Sentinel-2 tiles (image branch)
        cnn_figures.py      figures and cloud-cover check for reports/cnn/
        fusion.py           logistic-regression meta-learner on p_rf + p_cnn (fusion)
        fusion_figures.py   figures for reports/fusion/
    app/                    map viewer (index.html + build_data.py), deployed on Vercel
    docs/model_plan.md      modelling strategy: LR baseline -> RF -> CNN -> LR meta-learner fusion
    notebooks/              EDA, tabular and fusion notebooks (jupytext .py + executed .ipynb)
    reports/eda/            EDA figures
    reports/tabular/        tabular models: written report, metrics, predictions for fusion, figures
    reports/cnn/            CNN: written report, metrics, predictions for fusion, figures
    reports/fusion/         fusion: written report, metrics, meta-learner weights, figures
    reports/progress_report/ progress report as submitted (PDF + LaTeX source)
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
- [x] CNN: 2019 hold-out instead of a random one, five seeds and their average, freezing tested (no gain)
- [x] Tiles downloadable from the `tiles-v1` release (`download_images.py`), no Earth Engine needed
- [x] Fusion: LR meta-learner on RF + CNN, trained on 2020, test AUPRC 0.853, report (`reports/fusion/`)
- [x] Tabular inputs switched to day t-1 (no fire-day weather), tabular + fusion re-run

## Next steps
The plan and reasoning are in `docs/model_plan.md`. What is left before the sentence outline (15 Nov)
and the final report (25 Nov), as planned in the progress report:
1. Tabular models: done, including the switch to day t-1 inputs (`reports/tabular/README.md`).
2. CNN: done (2019 hold-out, five seeds, freezing tested; `reports/cnn/README.md`). Optional:
   near-infrared bands.
3. Fusion: refit on each single CNN training (`reports/cnn/predictions_runs.csv`), add per-year
   scores, and look at its errors on the map (`reports/fusion/README.md`). Whenever the CNN or the tabular models change,
   re-run `python -m forest_fire_risk.fusion` (10 s).
4. Final report, written together.
