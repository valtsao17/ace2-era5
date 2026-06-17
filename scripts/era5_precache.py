#!/usr/bin/env python3
"""Pre-cache all ERA5 daily data needed for rank_corr_analysis.

Opens the zarr store ONCE per stat/month combo and fetches all years sequentially.
Skips already-cached files, so safe to re-run after crashes.

Usage:
    python era5_precache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from hiro_ace_pipeline.io import subset_bbox  # noqa: F401 (import check)

CACHE_DIR    = PROJECT_ROOT / "outputs/lag_10yr/era5_cache_rankcorr"
COMBINED_DIR = PROJECT_ROOT / "outputs/lag_10yr/combined"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ERA5_ZARR      = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BASELINE_YEARS = list(range(1940, 2023))   # 83 years for thresholds
ANALYSIS_YEARS = list(range(1980, 1990))   # 10 years for observed

# stat/month pairs needed: m12 for lead030/060, m01 for lead090
NEEDED = [
    ("tmax", 12),
    ("tmin", 12),
    ("tmax",  1),
    ("tmin",  1),
]

# North America bbox (lat S, N; lon W, E in 0-360)
CONUS_BBOX = (18.0, 72.0, 195.0, 305.0)


def get_template() -> xr.DataArray:
    """Load a grid template from first combined file."""
    p = COMBINED_DIR / "tmax_1980_030d.nc"
    ds = xr.open_dataset(p)
    da = ds["TMP2m"].isel(sample=0)
    # Subset to NA bbox to get template on the target grid
    lat_s, lat_n, lon_w, lon_e = CONUS_BBOX
    da = da.sel(lat=slice(lat_s, lat_n), lon=slice(lon_w, lon_e))
    return da


def fetch_year(ds_zarr, vname: str, lat_name: str, lon_name: str,
               tlat_vals, tlon_vals,
               month: int, year: int, stat: str) -> xr.DataArray:
    """Extract daily max/min for one month/year from already-open zarr dataset."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    da    = ds_zarr[vname].sel(time=slice(str(start), str(end)))
    da    = da.sel({lat_name: tlat_vals, lon_name: tlon_vals}, method="nearest")
    da    = da.assign_coords({lat_name: tlat_vals, lon_name: tlon_vals})

    if stat == "tmax":
        daily = (da - 273.15).resample(time="1D").max()
    else:
        daily = (da - 273.15).resample(time="1D").min()

    daily = daily.sel(time=daily.time.dt.month == month).astype("float32").load()
    daily = daily.rename({lat_name: "lat", lon_name: "lon"})
    daily.name = "TMP2m"
    return daily


def precache_combo(stat: str, month: int, template: xr.DataArray):
    """Fetch all baseline years for one stat/month, opening zarr once."""
    needed_years = [y for y in BASELINE_YEARS
                    if not (CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{y:04d}.nc").exists()]

    if not needed_years:
        print(f"  [{stat} m{month:02d}] all {len(BASELINE_YEARS)} years already cached — skip",
              flush=True)
        return

    print(f"  [{stat} m{month:02d}] need {len(needed_years)} years "
          f"({needed_years[0]}–{needed_years[-1]}) — opening zarr ...", flush=True)

    tlat = next(c for c in template.coords if c in ("lat", "latitude"))
    tlon = next(c for c in template.coords if c in ("lon", "longitude"))
    tlat_vals = template[tlat].values
    tlon_vals = template[tlon].values

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            ds = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={})
            vname = next(v for v in ["2m_temperature", "t2m", "TMP2m"] if v in ds)
            lat_name = next(c for c in ds[vname].coords if c in ("latitude", "lat"))
            lon_name = next(c for c in ds[vname].coords if c in ("longitude", "lon"))
            break
        except Exception as e:
            print(f"    zarr open failed (attempt {attempt}/{max_attempts}): {e}", flush=True)
            if attempt == max_attempts:
                raise
            time.sleep(30 * attempt)

    failed = []
    for year in needed_years:
        path = CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{year:04d}.nc"
        if path.exists():
            continue

        for attempt in range(1, 4):
            try:
                daily = fetch_year(ds, vname, lat_name, lon_name,
                                   tlat_vals, tlon_vals, month, year, stat)
                daily.to_dataset().to_netcdf(path)
                print(f"    saved {path.name}", flush=True)
                break
            except Exception as e:
                print(f"    RETRY {year} attempt {attempt}/3: {e}", flush=True)
                if attempt == 3:
                    failed.append(year)
                    print(f"    SKIP {year} after 3 failures", flush=True)
                else:
                    time.sleep(20 * attempt)

    ds.close()

    if failed:
        print(f"  [{stat} m{month:02d}] WARNING: failed years: {failed}", flush=True)
    else:
        print(f"  [{stat} m{month:02d}] done", flush=True)


def precache_observed(stat: str, month: int, template: xr.DataArray):
    """Cache ERA5 observed value for each analysis year target date."""
    # Derive target dates from analysis years
    from datetime import datetime, timedelta
    lead_map = {"030d": 30, "060d": 60, "090d": 90}
    needed_dates = set()
    for year in ANALYSIS_YEARS:
        for lead_label, lead_days in lead_map.items():
            tgt = pd.Timestamp(datetime(year, 11, 1)) + pd.Timedelta(days=lead_days)
            if tgt.month == month:
                obs_path = CACHE_DIR / f"era5_obs_{stat}_{tgt.strftime('%Y%m%d')}.nc"
                if not obs_path.exists():
                    # The obs file is derived from the daily cache file
                    daily_path = CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{tgt.year:04d}.nc"
                    if daily_path.exists():
                        da = xr.open_dataset(daily_path)["TMP2m"]
                        obs = da.sel(time=str(tgt.date())).astype("float32")
                        obs.to_dataset(name="TMP2m").to_netcdf(obs_path)
                        print(f"    derived obs {obs_path.name}", flush=True)


def main():
    print("=== ERA5 pre-caching ===", flush=True)
    template = get_template()
    print(f"  template grid: {template.sizes}", flush=True)

    for stat, month in NEEDED:
        print(f"\n--- {stat} month {month:02d} ---", flush=True)
        precache_combo(stat, month, template)
        precache_observed(stat, month, template)

    # Final check
    print("\n=== Summary ===", flush=True)
    for stat, month in NEEDED:
        cached = sum(
            1 for y in BASELINE_YEARS
            if (CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{y:04d}.nc").exists()
        )
        print(f"  {stat} m{month:02d}: {cached}/{len(BASELINE_YEARS)} years cached", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
