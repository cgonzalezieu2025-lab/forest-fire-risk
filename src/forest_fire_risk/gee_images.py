"""Pull one Sentinel-2 RGB tile per Track A sample (2017-2022) from Google Earth Engine.

One-time setup (needs a Google account with a Cloud project that has the
Earth Engine API enabled, see https://developers.google.com/earth-engine/guides/access):

    earthengine authenticate
    export EE_PROJECT=<your-cloud-project-id>
    python -m forest_fire_risk.gee_images --limit 20     # try a few first
    python -m forest_fire_risk.gee_images                # all ~12k, resumable

Output: data/images/<sample_id>.png (256 x 256 px, 10 m/px = 2.56 km around the
cell centre) and data/images/index.csv (sample_id, image id, cloud %, date, or
status=missing when no usable image was found).

The image is the least-cloudy scene in the `LOOKBACK` days *before* the window's
end date (day 29 = fire day for positives). Never include the fire day or later:
smoke and burn scars would leak the label into the picture.
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from forest_fire_risk.data import load_track_a, to_sequences

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "images"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"   # L2A; swap for COPERNICUS/S2_HARMONIZED (TOA) if coverage is poor
LOOKBACK = 15          # days before the window end to search
MAX_CLOUD = 60         # scene-level CLOUDY_PIXEL_PERCENTAGE cutoff
SIZE_PX, SCALE_M = 256, 10
VIS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}


def select_samples(meta, years=(2017, 2022)):
    """Rows of `meta` whose window ends in [years[0], years[1]] (Sentinel-2 era)."""
    yr = pd.to_datetime(meta["time"]).dt.year
    return meta[yr.between(*years)].reset_index(drop=True)


def date_window(end):
    """(start, stop) ISO dates for the EE filter: LOOKBACK days ending the day before `end`."""
    end = pd.Timestamp(end).normalize()
    return (end - pd.Timedelta(days=LOOKBACK)).date().isoformat(), end.date().isoformat()  # stop is exclusive


def fetch_one(row, out_dir):
    import ee  # imported here so tests do not need it
    region = ee.Geometry.Point([row.x, row.y]).buffer(SIZE_PX * SCALE_M / 2).bounds()
    start, stop = date_window(row.time)
    coll = (ee.ImageCollection(COLLECTION).filterBounds(region).filterDate(start, stop)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
            .sort("CLOUDY_PIXEL_PERCENTAGE"))
    feats = coll.limit(1).getInfo()["features"]      # one round trip; empty list when nothing matches
    if not feats:
        return {"sample_id": row.sample_id, "status": "missing"}
    first = ee.Image(feats[0]["id"])
    info = {"id": feats[0]["id"].rsplit("/", 1)[-1], "cloud": feats[0]["properties"]["CLOUDY_PIXEL_PERCENTAGE"],
            "date": pd.Timestamp(feats[0]["properties"]["system:time_start"], unit="ms").date().isoformat()}
    url = first.getThumbURL({**VIS, "region": region, "dimensions": f"{SIZE_PX}x{SIZE_PX}", "format": "png"})
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    (out_dir / f"{row.sample_id}.png").write_bytes(r.content)
    return {"sample_id": row.sample_id, "status": "ok", "image": info["id"],
            "cloud": info["cloud"], "date": info["date"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"), help="GCP project id for EE")
    ap.add_argument("--limit", type=int, help="only the first N samples (for testing)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)

    _, _, _, meta = to_sequences(load_track_a())
    samples = select_samples(meta)
    args.out.mkdir(parents=True, exist_ok=True)
    index = args.out / "index.csv"
    done = set(pd.read_csv(index).sample_id) if index.exists() else set()
    todo = samples[~samples.sample_id.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(samples)} samples in 2017-2022, {len(done)} done, fetching {len(todo)}")

    fields = ["sample_id", "status", "image", "cloud", "date"]
    with open(index, "a", newline="") as f, ThreadPoolExecutor(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        futs = {pool.submit(fetch_one, row, args.out): row.sample_id for row in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                w.writerow(fut.result()); f.flush()
            except Exception as e:  # ponytail: log and continue; rerun picks up anything not in index.csv
                print(f"  {futs[fut]}: {type(e).__name__}: {e}")
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}")


if __name__ == "__main__":
    main()
