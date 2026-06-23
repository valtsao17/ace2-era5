#!/usr/bin/env python3
"""
Re-fetch ERA5 daily Tmin from ARCO-ERA5 (24-hourly T2m minimum) for all DJF
months needed by seasonal_djf_skill.py.

Uses the public Google Research ARCO-ERA5 zarr:
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

Replaces the existing cache in outputs/lag_nov/postprocess_djf/era5_cache/
which was built from 6-hourly TMP2m snapshots.  The 24-hour T2m minimum
captures the true daily minimum temperature far better than 4 snapshots.

Usage:
  python refetch_era5_tmin_arco.py            # skip already-cached months
  python refetch_era5_tmin_arco.py --force    # overwrite all
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = PROJECT_ROOT / "outputs/lag_nov/combined_djf"
ERA5_CACHE   = PROJECT_ROOT / "outputs/lag_nov/postprocess_djf/era5_cache"

YEARS      = list(range(1981, 2016))
DJF_MONTHS = [12, 1, 2]
ARCO_URL   = ("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
ERA5_REF   = pd.Timestamp("1900-01-01 00:00:00")


def abs_djf_months(init_year: int) -> list[tuple[int, int]]:
    return [(init_year, 12), (init_year + 1, 1), (init_year + 1, 2)]


def needed_months() -> list[tuple[int, int]]:
    months: set[tuple[int, int]] = set()
    for yr in YEARS:
        for y, m in abs_djf_months(yr):
            months.add((y, m))
    return sorted(months)


def time_index(dt: pd.Timestamp) -> int:
    return int((dt - ERA5_REF).total_seconds() / 3600)


def fetch_day(t2m_store, day_start_idx: int) -> np.ndarray:
    """Return (721, 1440) float32 daily Tmin (K) from 24 hourly T2m."""
    return t2m_store[day_start_idx: day_start_idx + 24, :, :].min(axis=0)


def fetch_month(
    t2m_store,
    arco_lat: np.ndarray,
    arco_lon: np.ndarray,
    year: int,
    month: int,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    n_workers: int = 6,
) -> xr.DataArray:
    n_days = calendar.monthrange(year, month)[1]
    days   = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")
    t_idxs = [time_index(d) for d in days]

    # Fetch all days in the month concurrently
    day_mins: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(fetch_day, t2m_store, ti): di
                for di, ti in enumerate(t_idxs)}
        for fut in as_completed(futs):
            day_mins[futs[fut]] = fut.result()

    # Stack in day order → (n_days, 721, 1440) in K
    daily_K = np.stack([day_mins[i] for i in range(n_days)], axis=0).astype(np.float32)

    # Build xr.DataArray on the ARCO 0.25-deg grid (lat is N→S, needs flip for interp)
    da_arco = xr.DataArray(
        daily_K,
        dims=["time", "latitude", "longitude"],
        coords={
            "time":      days,
            "latitude":  arco_lat,   # 90 → -90 (decreasing)
            "longitude": arco_lon,   # 0 → 359.75
        },
    )

    # Regrid to our 1-deg ACE2 grid via bilinear interpolation
    da_1deg = da_arco.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Convert K → °C and rename dims to match cache convention
    da_out = (da_1deg - 273.15).rename({"latitude": "lat", "longitude": "lon"})
    da_out.attrs = {"units": "degC", "long_name": "ERA5 daily Tmin (ARCO 24-hr T2m min)"}
    return da_out.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing cached months")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel day-fetch threads per month (default 6)")
    args = p.parse_args()

    ERA5_CACHE.mkdir(parents=True, exist_ok=True)

    # Load target grid from first combined file
    with xr.open_dataset(COMBINED_DIR / f"tmin_djf_{YEARS[0]}.nc") as ds:
        target_lat = ds["lat"].values
        target_lon = ds["lon"].values
    print(f"Target grid: {len(target_lat)} lats ({target_lat[0]:.2f}–{target_lat[-1]:.2f}), "
          f"{len(target_lon)} lons ({target_lon[0]:.2f}–{target_lon[-1]:.2f})")

    print("Opening ARCO-ERA5 zarr (anon) ...")
    fs    = gcsfs.GCSFileSystem(token="anon")
    store = zarr.open(fs.get_mapper(ARCO_URL), mode="r")
    t2m   = store["2m_temperature"]
    arco_lat = store["latitude"][:]
    arco_lon = store["longitude"][:]
    print(f"ARCO T2m: {t2m.shape}, chunks {t2m.chunks}, compressor {t2m.compressor.cname}")

    months = needed_months()
    print(f"\nMonths to process: {len(months)}  (force={args.force})")

    for year, month in tqdm(months, desc="months"):
        out_path = ERA5_CACHE / f"era5_tmin_{year}{month:02d}.nc"
        if out_path.exists() and not args.force:
            continue

        try:
            da = fetch_month(t2m, arco_lat, arco_lon, year, month,
                             target_lat, target_lon, n_workers=args.workers)
            da.to_dataset(name="era5_daily_tmin_C").to_netcdf(out_path)
            tmin_mean = float(da.mean())
            print(f"  {year}-{month:02d}: wrote {out_path.name}  mean={tmin_mean:.2f}°C",
                  flush=True)
        except Exception as exc:
            print(f"  ERROR {year}-{month:02d}: {exc}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
