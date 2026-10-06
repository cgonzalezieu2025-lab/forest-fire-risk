import argparse

import pandas as pd
import xarray as xr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube", help="path to mesogeos_cube.zarr")
    ap.add_argument("--date", default="2020-07-15", help="single day used for the sample statistics")
    args = ap.parse_args()

    # lazy: only metadata is read here, no data
    ds = xr.open_zarr(args.cube)

    print("=== DIMENSIONS ===")
    print(dict(ds.sizes))

    print("\n=== COORDINATES ===")
    t = ds["time"].values
    print(f"time: {pd.Timestamp(t[0]).date()} -> {pd.Timestamp(t[-1]).date()} ({len(t)} days)")
    for c in ("x", "y"):
        v = ds[c].values
        step = abs(float(v[1] - v[0]))
        print(f"{c}: {v.min():.3f} -> {v.max():.3f} (step {step:.4f} deg, {len(v)} cells)")

    print("\n=== VARIABLES ===")
    rows = []
    for name, da in ds.data_vars.items():
        kind = "static" if "time" not in da.dims else "daily"
        rows.append((name, kind, str(da.dims), str(da.dtype), da.attrs.get("units", "")))
    print(pd.DataFrame(rows, columns=["name", "kind", "dims", "dtype", "units"]).to_string(index=False))
    print(f"\n{len(rows)} data variables")

    print(f"\n=== SAMPLE STATISTICS FOR {args.date} (loads one day, can take a minute) ===")
    day = ds.sel(time=args.date)
    stats = []
    for name, da in day.data_vars.items():
        a = da.values  # one 2D map
        s = pd.Series(a.ravel())
        stats.append((name, float(s.min()), float(s.max()), float(s.mean()), float(pd.isna(a).mean())))
    out = pd.DataFrame(stats, columns=["name", "min", "max", "mean", "frac_nan"])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()































