"""Build the map viewer's data: app/samples.json and app/sprites/*.jpg (128 px thumbnails,
16x16 per sheet) so the page works without the 1.7 GB of full-res tiles.
    python app/build_data.py && python -m http.server 8000   # then open http://localhost:8000/app/
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from forest_fire_risk.data import DYNAMIC, STATIC, load_track_a, to_tabular

ROOT = Path(__file__).resolve().parents[1]
IMG, SPRITES = ROOT / "data/images", ROOT / "app/sprites"
PX, GRID = 128, 16                                   # 256 thumbnails per sheet

tab = to_tabular(load_track_a())
tab = tab[tab.time.dt.year >= 2017]
idx = pd.read_csv(IMG / "index.csv")
idx = idx[idx.status == "ok"].reset_index(drop=True)
idx["sprite"] = [[i // GRID**2, i % GRID, i // GRID % GRID] for i in range(len(idx))]   # [sheet, col, row]
tab = tab.merge(idx[["sample_id", "cloud", "date", "sprite"]], on="sample_id", how="left")
tab["time"] = tab.time.dt.strftime("%Y-%m-%d")
cols = ["sample_id", "label", "time", "x", "y", "cloud", "date", "sprite"] + DYNAMIC + STATIC
recs = [{k: (None if not isinstance(v, list) and pd.isna(v) else v) for k, v in r.items()}
        for r in tab[cols].round(4).to_dict("records")]
(ROOT / "app/samples.json").write_text(json.dumps(recs, separators=(",", ":")))
print(len(recs), "samples ->", ROOT / "app/samples.json")

SPRITES.mkdir(exist_ok=True)
for k in range(0, len(idx), GRID**2):
    out = SPRITES / f"{k // GRID**2}.jpg"
    if out.exists():
        continue
    sheet = Image.new("RGB", (GRID * PX, GRID * PX))
    for i, sid in enumerate(idx.sample_id[k:k + GRID**2]):
        sheet.paste(Image.open(IMG / f"{sid}.png").convert("RGB").resize((PX, PX), Image.LANCZOS), (i % GRID * PX, i // GRID * PX))
    sheet.save(out, "JPEG", quality=80)
print(len(list(SPRITES.glob("*.jpg"))), "sprite sheets in", SPRITES)
