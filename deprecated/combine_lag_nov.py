#!/usr/bin/env python3
"""Extract daily Tmax and Tmin at target dates from all lag member outputs.

For each year 1980-1989:
  - 25 member outputs, each 500 steps (3000 h)
  - Target dates: Dec 1 (+30d), Dec 31 (+60d), Jan 30 next year (+90d)
  - Extract daily max and min TMP2m at each target date
  - Stack across members: (sample=25, lat, lon) per year per target date

Output files (in outputs/lag_10yr/combined/):
  tmax_{year}_lead030d.nc   shape (sample=25, lat=180, lon=360)
  tmax_{year}_lead060d.nc
  tmax_{year}_lead090d.nc
  tmin_{year}_lead030d.nc   (same structure, daily min)
  tmin_{year}_lead060d.nc
  tmin_{year}_lead090d.nc
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RUNS_ROOT = Path("/home/jovyan/hiro_ace_clean_v4/outputs/lag_10yr/runs")
OUT_DIR   = Path("/home/jovyan/hiro_ace_clean_v4/outputs/lag_10yr/combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS     = list(range(1980, 1990))
N_MEMBERS = 25

# Target offsets from Nov 1 in days
LEADS = {"030d": 30, "060d": 60, "090d": 90}


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def target_date(year: int, lead_days: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, 11, 1)) + pd.Timedelta(days=lead_days)


def assign_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    """Assign absolute datetime coords to fme microsecond time axis."""
    vals = ds["time"].values.astype(np.int64)  # microseconds from init
    base = pd.Timestamp(init_time)
    abs_times = base + pd.to_timedelta(vals, unit="us")
    return ds.assign_coords(time=abs_times)


def extract_member(pred_path: Path, init_time: datetime,
                   targets: list[pd.Timestamp]) -> dict:
    """Return dict of lead_label -> (tmax, tmin) arrays shape (lat, lon)."""
    ds = xr.open_dataset(pred_path, decode_times=False)
    ds = assign_times(ds, init_time)
    tmp2m = ds["TMP2m"].isel(sample=0)  # (time, lat, lon)

    daily_max = tmp2m.resample(time="1D").max()
    daily_min = tmp2m.resample(time="1D").min()
    ds.close()

    results = {}
    for label, tgt in targets:
        tgt_str = str(tgt.date())
        try:
            tmax = daily_max.sel(time=tgt_str).values.astype(np.float32)
            tmin = daily_min.sel(time=tgt_str).values.astype(np.float32)
        except KeyError:
            raise KeyError(f"Target date {tgt_str} not found in {pred_path}")
        results[label] = (tmax, tmin)
    return results


def combine_year(year: int, lat, lon):
    times   = lag_times(year)
    targets = [(label, target_date(year, days)) for label, days in LEADS.items()]

    # Check all outputs already exist
    all_exist = all(
        (OUT_DIR / f"tmax_{year}_{label}.nc").exists() and
        (OUT_DIR / f"tmin_{year}_{label}.nc").exists()
        for label, _ in targets
    )
    if all_exist:
        print(f"  SKIP {year}: all combined files exist")
        return

    # Collect per member
    tmax_stacks = {label: [] for label, _ in targets}
    tmin_stacks = {label: [] for label, _ in targets}

    for idx in range(N_MEMBERS):
        pred_path = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing: {pred_path}")
        print(f"  [{year} m{idx:02d}]", flush=True)
        res = extract_member(pred_path, times[idx], targets)
        for label, (tmax, tmin) in res.items():
            tmax_stacks[label].append(tmax)
            tmin_stacks[label].append(tmin)

    # Write combined files
    for label, tgt in targets:
        tmax_data = np.stack(tmax_stacks[label], axis=0)   # (25, lat, lon)
        tmin_data = np.stack(tmin_stacks[label], axis=0)

        for varname, data in [("tmax", tmax_data), ("tmin", tmin_data)]:
            da = xr.DataArray(
                data,
                dims=["sample", "lat", "lon"],
                coords={"sample": np.arange(N_MEMBERS), "lat": lat, "lon": lon},
                attrs={"target_date": str(tgt.date()), "units": "K",
                       "long_name": f"daily {'max' if varname=='tmax' else 'min'} TMP2m"},
            )
            out = OUT_DIR / f"{varname}_{year}_{label}.nc"
            da.to_dataset(name="TMP2m").to_netcdf(out)
        print(f"  wrote {year} {label}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated, e.g. '1980,1981'")
    args = p.parse_args()
    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]

    # Get lat/lon from first member of first year
    sample_path = (RUNS_ROOT / str(years[0]) / "member_00" /
                   "autoregressive_predictions.nc")
    if not sample_path.exists():
        print(f"ERROR: sample output not found: {sample_path}")
        return
    with xr.open_dataset(sample_path, decode_times=False) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values

    for year in years:
        print(f"\n=== {year} ===", flush=True)
        combine_year(year, lat, lon)

    print("\nAll years combined.", flush=True)


if __name__ == "__main__":
    main()
