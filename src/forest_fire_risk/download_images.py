"""Download the Sentinel-2 tiles from the GitHub release instead of Google Earth Engine.

    python -m forest_fire_risk.download_images    # -> data/images/ (1.7 GB: 11,579 PNGs + index.csv)

The archive is exactly what `gee_images.py` fetched in October 2026, so the CNN can be
reproduced without an Earth Engine account.
"""
import hashlib
import tarfile
import urllib.request

from forest_fire_risk.data import DATA_DIR

URL = "https://github.com/cgonzalezieu2025-lab/forest-fire-risk/releases/download/tiles-v1/sentinel2_tiles.tar"
SHA256 = "6432da53b0f4df25621429d0f71678c18199b10eb26fe3a60be8d849c387eb27"
OUT = DATA_DIR.parents[1] / "images"


def main():
    if (OUT / "index.csv").exists() and any(OUT.glob("*.png")):
        print(f"Already downloaded in {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "sentinel2_tiles.tar"
    print(f"downloading {URL} (1.7 GB) ...")
    urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise SystemExit(f"checksum mismatch, delete {archive} and try again")
    with tarfile.open(archive) as tar:
        tar.extractall(OUT, filter="data")
    archive.unlink()
    print(f"Done. {len(list(OUT.glob('*.png')))} tiles in {OUT}")


if __name__ == "__main__":
    main()
