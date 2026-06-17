#!/usr/bin/env python3
"""Combine 25 lag member outputs into per-year JJA daily Tmax files.

For each year 1980-2016:
  - 25 members, each initialized around May 1 (center ± 12×6h, i.e. Apr 28 – May 4)
  - Computes daily Tmax from 6-hourly TMP2m
  - Keeps only June, July, August days
  - Member 12 (init May 1 00:00) is used as the reference time axis;
    other members are reindexed onto it (NaN where out of range)

Output: outputs/lag_may/combined_jja/tmax_jja_{year}.nc
        dims: (member=25, time=~70days, lat, lon)   units: K
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT    = PROJECT_ROOT / "outputs/lag_may/runs"
OUT_DIR      = PROJECT_ROOT / "outputs/lag_may/combined_jja"

YEARS     = list(range(1980, 2017))
N_MEMBERS = 25


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 5, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def assign_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    """Convert the raw integer time coord (microseconds from init) to datetime64."""
    vals = ds["time"].values.astype(np.int64)
    base = pd.Timestamp(init_time)
    return ds.assign_coords(time=base + pd.to_timedelta(vals, unit="us"))


def extract_member_jja_tmax(pred_path: Path, init_time: datetime) -> xr.DataArray:
    ds    = xr.open_dataset(pred_path, decode_times=False)
    ds    = assign_times(ds, init_time)
    tmp2m = ds["TMP2m"].isel(sample=0)          # (time, lat, lon)
    dmax  = tmp2m.resample(time="1D").max()      # daily max
    ds.close()
    jja   = dmax.time.dt.month.isin([6, 7, 8])
    return dmax.sel(time=jja).astype(np.float32)


def combine_year(year: int):
    out_path = OUT_DIR / f"tmax_jja_{year}.nc"
    if out_path.exists():
        print(f"  SKIP {year}: already exists")
        return

    times   = lag_times(year)
    members = []

    for idx in range(N_MEMBERS):
        pred_path = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing: {pred_path}")
        print(f"  [{year} m{idx:02d}]", flush=True)
        members.append(extract_member_jja_tmax(pred_path, times[idx]))

    # Use member 12 (init May 1 00:00) as the canonical time axis
    ref_times = members[12].time.values

    tmax_arr = np.stack(
        [m.reindex(time=ref_times, fill_value=np.nan).values for m in members],
        axis=0,
    )  # (25, n_jja_days, lat, lon)

    lat = members[12]["lat"].values
    lon = members[12]["lon"].values

    da = xr.DataArray(
        tmax_arr,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(N_MEMBERS),
            "time":   ref_times,
            "lat":    lat,
            "lon":    lon,
        },
        attrs={"units": "K", "long_name": "JJA daily max TMP2m", "init_year": year},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name="TMP2m").to_netcdf(out_path)
    print(f"  wrote {out_path.name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all", help="'all' or comma-separated years")
    args  = p.parse_args()
    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        print(f"\n=== {year} ===", flush=True)
        combine_year(year)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
