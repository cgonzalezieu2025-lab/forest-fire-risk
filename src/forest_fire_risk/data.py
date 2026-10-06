from pathlib import Path

import pandas as pd

# repo root / data / raw / track_a
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "track_a"


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


def main():
    df = load_track_a()
    print("shape:", df.shape)
    print("date range:", df["time"].min().date(), "->", df["time"].max().date())

    per_sample = df.groupby("sample_id").size()
    print("\nrows per sample (should be one constant number, e.g. 30):")
    print(per_sample.value_counts().head())

    samples = df.drop_duplicates("sample_id")
    print("\nsamples per label:")
    print(samples["label"].value_counts())

    print("\nfraction of missing values (top 10 columns):")
    print(df.isna().mean().sort_values(ascending=False).head(10).round(3))


if __name__ == "__main__":
    main()