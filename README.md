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

    mkdir -p data/track_a && cd data/track_a
    python -c "
    import gdown
    for id, name in [('1IYONBanlerMi84wedto-Vck8vmfnyVUR','positives.csv'), ('1qB6TjMCgpVvM04ysCZSgE-5sjJ2C9gNJ','negatives.csv'),
                     ('1PdwYyMX-51-zJ1dr36g8Hq-_7K7KTVAx','vars_dict.json'), ('1uXsRpCJa7JNlPXmHvzCCIxs3E0eC6NUE','norms.json')]:
        gdown.download(id=id, output=name)"

Full datacube (`mesogeos_cube.zarr`), burned-area and ecoregion shapefiles, and Track B samples live in the same Drive folder:
https://drive.google.com/drive/folders/1aRXQXVvw6hz0eYgtJDoixjPQO-_bRKz9

## EDA
`notebooks/eda_track_a.ipynb` (generated from `eda_track_a.py` with jupytext; figures in `reports/eda/`).
