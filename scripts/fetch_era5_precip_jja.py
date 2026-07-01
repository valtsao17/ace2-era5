#!/usr/bin/env python3
"""ERA5 observed JJA daily precip totals (mm) on the ACE2 grid, for the precip
skill panel. Pulls total_precipitation from the public ARCO-ERA5 zarr (same source
as heat_index_era5.py), one year-month at a time, regrids nearest to the ACE2
prediction grid, sums the 24 hourly accumulations (m) x 1000 -> daily mm, and
concatenates Jun+Jul+Aug.

    outputs/lag_may/precip_jja/era5_precip_cache/era5_precip_jja_{YYYY}.nc
        precip_mmday (time=92, lat, lon)   # full ACE2 grid

Usage:  fetch_era5_precip_jja.py [--years 2002-2016] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.io import subset_bbox, regrid_nearest_to_template, write_atomic  # noqa: E402

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BBOX      = (-90.0, 90.0, 0.0, 360.0)
MONTHS    = [6, 7, 8]
PJJA_DIR  = PROJECT_ROOT / "outputs/lag_may/precip_jja"
CACHE_DIR = PJJA_DIR / "era5_precip_cache"


def template_grid():
    """ACE2 prediction grid from any short-run member (full 180x360)."""
    f = sorted((PROJECT_ROOT / "outputs/lag_may/runs_precip_short").glob(
        "*/member_*/autoregressive_predictions.nc"))[0]
    with xr.open_dataset(f) as ds:
        da = ds["PRATEsfc"].isel(sample=0, time=0)
        return xr.DataArray(np.zeros((da.lat.size, da.lon.size), np.float32),
                            coords={"lat": da.lat.values, "lon": da.lon.values},
                            dims=("lat", "lon"))


def month_daily_mm(ds_zarr, year, month, template):
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    tp = subset_bbox(ds_zarr["total_precipitation"].sel(time=slice(str(start), str(end))), BBOX)
    tp = regrid_nearest_to_template(tp, template).load()
    nh, nlat, nlon = tp.shape
    nd = nh // 24
    assert nh == nd * 24, f"{year}-{month}: {nh} hours not divisible by 24"
    daily = tp.values.reshape(nd, 24, nlat, nlon).sum(axis=1) * 1000.0  # mm/day
    days = pd.date_range(start, periods=nd, freq="D")
    return daily.astype(np.float32), days


def fetch_year(ds_zarr, year, template, force=False):
    path = CACHE_DIR / f"era5_precip_jja_{year}.nc"
    if path.exists() and not force:
        print(f"  [{year}] cached", flush=True)
        return
    daily, times = [], []
    for m in MONTHS:
        print(f"  [{year}-{m:02d}] fetch+regrid ...", flush=True)
        d, t = month_daily_mm(ds_zarr, year, m, template)
        daily.append(d); times.append(t)
    arr = np.concatenate(daily, axis=0)
    tt = np.concatenate([t.values for t in times])
    out = xr.Dataset(
        {"precip_mmday": (("time", "lat", "lon"), arr)},
        coords={"time": tt, "lat": template.lat.values, "lon": template.lon.values},
        attrs={"long_name": f"ERA5 JJA daily total precip (mm), {year}, ACE2 grid"},
    )
    write_atomic(out, path)
    print(f"  wrote {path}  (ndays={arr.shape[0]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2002-2016")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if "-" in args.years:
        a, b = args.years.split("-"); years = list(range(int(a), int(b) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    template = template_grid()
    print("opening ARCO-ERA5 zarr ...", flush=True)
    ds = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={}, storage_options={"token": "anon"})
    for y in years:
        print(f"ERA5 {y} ...", flush=True)
        fetch_year(ds, y, template, force=args.force)
    ds.close()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
