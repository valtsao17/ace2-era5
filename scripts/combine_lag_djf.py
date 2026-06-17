#!/usr/bin/env python3
"""Extract DJF (Dec-Jan-Feb) daily Tmax/Tmin over CONUS from lag member outputs.

For each year (25-member ensemble initialized ~Nov 1), reads the 6-hourly
autoregressive_predictions.nc for each member, subsets to CONUS, resamples
to daily Tmax/Tmin, and keeps only December-January-February days.

Member init times span Oct 29 - Nov 4 (centered on Nov 1, 6-hour spacing).
All 500-step (125-day) runs cover the full DJF window.

Outputs (in outputs/lag_10yr/combined_djf/):
  tmax_djf_{year}.nc   shape (member=25, time=~90, lat, lon)  CONUS, units K
  tmin_djf_{year}.nc
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RUNS_ROOT = Path("/home/jovyan/hiro_ace_clean_v4/outputs/lag_10yr/runs")
OUT_DIR   = Path("/home/jovyan/hiro_ace_clean_v4/outputs/lag_10yr/combined_djf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS     = list(range(1980, 2001))
N_MEMBERS = 25

# CONUS bounding box — lat: S, N; lon: W, E in 0-360 (matches rank_corr_analysis.py)
CONUS_BBOX = (18.0, 72.0, 195.0, 305.0)


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def assign_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    vals = ds["time"].values.astype(np.int64)   # microseconds from init
    base = pd.Timestamp(init_time)
    abs_times = base + pd.to_timedelta(vals, unit="us")
    return ds.assign_coords(time=abs_times)


def subset_conus(da: xr.DataArray) -> xr.DataArray:
    lat_s, lat_n, lon_w, lon_e = CONUS_BBOX
    lat_name = next(c for c in da.coords if c in ("lat", "latitude"))
    lon_name = next(c for c in da.coords if c in ("lon", "longitude"))
    lat_vals = da[lat_name].values
    lat_sel  = slice(lat_n, lat_s) if lat_vals[0] > lat_vals[-1] else slice(lat_s, lat_n)
    return da.sel({lat_name: lat_sel, lon_name: slice(lon_w, lon_e)})


def extract_member_djf(pred_path: Path, init_time: datetime
                       ) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (tmax_djf, tmin_djf) with dim=time covering DJF calendar days."""
    ds     = xr.open_dataset(pred_path, decode_times=False)
    ds     = assign_times(ds, init_time)
    tmp2m  = ds["TMP2m"].isel(sample=0)   # (time, lat, lon)
    tmp2m  = subset_conus(tmp2m)           # clip to CONUS before resampling
    dmax   = tmp2m.resample(time="1D").max()
    dmin   = tmp2m.resample(time="1D").min()
    ds.close()
    djf    = dmax.time.dt.month.isin([12, 1, 2])
    return dmax.sel(time=djf).astype(np.float32), dmin.sel(time=djf).astype(np.float32)


def combine_year(year: int):
    out_tmax = OUT_DIR / f"tmax_djf_{year}.nc"
    out_tmin = OUT_DIR / f"tmin_djf_{year}.nc"
    if out_tmax.exists() and out_tmin.exists():
        print(f"  SKIP {year}: files exist")
        return

    times        = lag_times(year)
    tmax_members = []
    tmin_members = []

    for idx in range(N_MEMBERS):
        pred_path = (RUNS_ROOT / str(year) / f"member_{idx:02d}" /
                     "autoregressive_predictions.nc")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing: {pred_path}")
        print(f"  [{year} m{idx:02d}]", flush=True)
        tmax_djf, tmin_djf = extract_member_djf(pred_path, times[idx])
        tmax_members.append(tmax_djf)
        tmin_members.append(tmin_djf)

    # Use member 12 (closest to Nov 1 00:00) as the reference time axis
    ref_times = tmax_members[12].time.values

    tmax_arr = np.stack(
        [m.reindex(time=ref_times).values for m in tmax_members], axis=0)  # (25, ndays, lat, lon)
    tmin_arr = np.stack(
        [m.reindex(time=ref_times).values for m in tmin_members], axis=0)

    lat = tmax_members[12]["lat"].values
    lon = tmax_members[12]["lon"].values

    coords = {"member": np.arange(N_MEMBERS), "time": ref_times,
              "lat": lat, "lon": lon}

    for varname, arr in [("tmax", tmax_arr), ("tmin", tmin_arr)]:
        da = xr.DataArray(
            arr, dims=["member", "time", "lat", "lon"], coords=coords,
            attrs={"units": "K",
                   "long_name": f"DJF daily {'max' if varname=='tmax' else 'min'} TMP2m",
                   "init_year": year,
                   "n_members": N_MEMBERS})
        out = OUT_DIR / f"{varname}_djf_{year}.nc"
        da.to_dataset(name="TMP2m").to_netcdf(out)
        print(f"  wrote {out.name}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated years, e.g. '1980,1981'")
    args  = p.parse_args()
    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]

    for year in years:
        print(f"\n=== {year} ===", flush=True)
        combine_year(year)

    print("\nAll years combined (DJF).", flush=True)


if __name__ == "__main__":
    main()
