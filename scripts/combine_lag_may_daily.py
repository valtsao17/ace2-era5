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


def parse_years(spec: str) -> list[int]:
    if spec == "all":
        return YEARS
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


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


def combine_year(year: int, allow_partial: bool = False, force: bool = False):
    out_path = OUT_DIR / f"tmax_jja_{year}.nc"
    if out_path.exists() and not force:
        print(f"  SKIP {year}: already exists")
        return

    times   = lag_times(year)
    members: dict[int, xr.DataArray] = {}
    missing = []

    for idx in range(N_MEMBERS):
        pred_path = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            if allow_partial:
                missing.append(idx)
                print(f"  [{year} m{idx:02d}] missing -> NaN", flush=True)
                continue
            raise FileNotFoundError(f"Missing: {pred_path}")
        try:
            print(f"  [{year} m{idx:02d}]", flush=True)
            members[idx] = extract_member_jja_tmax(pred_path, times[idx])
        except Exception as exc:
            if not allow_partial:
                raise
            missing.append(idx)
            print(f"  [{year} m{idx:02d}] unreadable -> NaN ({exc})", flush=True)

    if not members:
        raise RuntimeError(f"No usable members for {year}")

    # Use member 12 (init May 1 00:00) as the canonical time axis
    ref_idx = 12 if 12 in members else sorted(members)[0]
    ref_times = members[ref_idx].time.values
    lat = members[ref_idx]["lat"].values
    lon = members[ref_idx]["lon"].values

    tmax_arr = np.full((N_MEMBERS, len(ref_times), len(lat), len(lon)), np.nan, dtype=np.float32)
    for idx, member in members.items():
        tmax_arr[idx] = member.reindex(time=ref_times, fill_value=np.nan).values

    da = xr.DataArray(
        tmax_arr,
        dims=["member", "time", "lat", "lon"],
        coords={
            "member": np.arange(N_MEMBERS),
            "time":   ref_times,
            "lat":    lat,
            "lon":    lon,
        },
        attrs={
            "units": "K",
            "long_name": "JJA daily max TMP2m",
            "init_year": year,
            "n_members_available": len(members),
            "missing_members": ",".join(str(x) for x in missing),
            "partial_member_file": int(bool(missing)),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name="TMP2m").to_netcdf(out_path)
    print(f"  wrote {out_path.name} ({len(members)}/25 members)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all", help="'all' or comma-separated years")
    p.add_argument("--allow-partial", action="store_true",
                   help="Write a 25-member file with NaNs for missing/unreadable members")
    p.add_argument("--force", action="store_true", help="Overwrite existing combined files")
    args  = p.parse_args()
    years = parse_years(args.years)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        print(f"\n=== {year} ===", flush=True)
        combine_year(year, allow_partial=args.allow_partial, force=args.force)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
