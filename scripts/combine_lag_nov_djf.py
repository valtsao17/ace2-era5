#!/usr/bin/env python3
"""Combine Nov-1 lag member outputs into per-year DJF daily Tmax/Tmin files.

For each init year (2002-2015):
  - 25 members, each initialized around Nov 1 (center ± 12×6h, Oct 29 – Nov 4)
  - 500 steps × 6h = 125-day runs → covers Dec, Jan, Feb
  - Computes daily Tmax and Tmin from 6-hourly TMP2m
  - Keeps only December, January, February days
  - Member 12 (init Nov 1 00:00) used as the canonical time axis

Input:  /home/vt55/ace2/runs/{year}/member_{i:02d}/autoregressive_predictions.nc
Output: outputs/lag_nov/combined_djf/tmax_djf_{year}.nc
        outputs/lag_nov/combined_djf/tmin_djf_{year}.nc
        dims: (member=25, time=~90 days, lat=180, lon=360)  units: K
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT    = PROJECT_ROOT / "runs"
OUT_DIR      = PROJECT_ROOT / "outputs/lag_nov/combined_djf"

YEARS     = list(range(1981, 2016))   # 35 years (1981-2015)
N_MEMBERS = 25


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def assign_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    vals = ds["time"].values.astype(np.int64)   # microseconds from init
    base = pd.Timestamp(init_time)
    return ds.assign_coords(time=base + pd.to_timedelta(vals, unit="us"))


def extract_member_djf(pred_path: Path, init_time: datetime
                       ) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (tmax_djf, tmin_djf) for this member, DJF days only, in K."""
    ds    = xr.open_dataset(pred_path, decode_times=False)
    ds    = assign_times(ds, init_time)
    tmp2m = ds["TMP2m"].isel(sample=0)          # (time, lat, lon)
    dmax  = tmp2m.resample(time="1D").max()
    dmin  = tmp2m.resample(time="1D").min()
    ds.close()
    djf   = dmax.time.dt.month.isin([12, 1, 2])
    return (dmax.sel(time=djf).astype(np.float32),
            dmin.sel(time=djf).astype(np.float32))


def combine_year(year: int):
    out_tmax = OUT_DIR / f"tmax_djf_{year}.nc"
    out_tmin = OUT_DIR / f"tmin_djf_{year}.nc"
    if out_tmax.exists() and out_tmin.exists():
        print(f"  SKIP {year}: already exists")
        return

    times   = lag_times(year)
    tmax_members, tmin_members = [], []

    for idx in range(N_MEMBERS):
        pred_path = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing: {pred_path}")
        print(f"  [{year} m{idx:02d}]", flush=True)
        tmax_da, tmin_da = extract_member_djf(pred_path, times[idx])
        tmax_members.append(tmax_da)
        tmin_members.append(tmin_da)

    # Member 12 (init Nov 1 center) as canonical time axis
    ref_times = tmax_members[12].time.values
    lat = tmax_members[12]["lat"].values
    lon = tmax_members[12]["lon"].values

    for varname, members, out_path in [
        ("tmax", tmax_members, out_tmax),
        ("tmin", tmin_members, out_tmin),
    ]:
        arr = np.stack(
            [m.reindex(time=ref_times, fill_value=np.nan).values for m in members],
            axis=0,
        )   # (25, n_djf_days, lat, lon)
        da = xr.DataArray(
            arr,
            dims=["member", "time", "lat", "lon"],
            coords={"member": np.arange(N_MEMBERS), "time": ref_times,
                    "lat": lat, "lon": lon},
            attrs={"units": "K",
                   "long_name": f"DJF daily {'max' if varname=='tmax' else 'min'} TMP2m",
                   "init_year": year},
        )
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
