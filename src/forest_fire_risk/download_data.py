from pathlib import Path

import gdown

# Drive folder "a.danger_forecasting" (inside ml_tracks of the Mesogeos Drive)
TRACK_A_URL = "https://drive.google.com/drive/folders/1dRyn7EAwG88f0QMKGz74rWY8krKH8VSV"

# repo root / data / raw / track_a
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "track_a"


def main():
    if (OUT / "positives.csv").exists() and (OUT / "negatives.csv").exists():
        print(f"Already downloaded in {OUT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(TRACK_A_URL, output=str(OUT), quiet=False)
    print(f"Done. Files in {OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()







