#!/usr/bin/env python3
"""Extract a compact CONUS JJA daily-precip cache from raw ACE2 PRATEsfc runs.

Mirrors hhe_ace2_biascorr.member_daily but for precipitation: opens each member's
500-step PRATEsfc prediction, assigns calendar times from the member's May-init
lag time, resamples to daily-MEAN rate (mm/day = PRATEsfc[kg/m2/s] x 86400),
keeps JJA (Jun-Aug) days, slices CONUS, and writes

    outputs/lag_may/precip_jja/ace2_precip_cache/precip_y{YYYY}_mem{ii}.nc
        precip_mmday (time<=92, lat, lon)

so the big raw run dirs can be deleted immediately afterwards (disk-safe pipeline).

Usage:  extract_precip_jja.py --year 2002 [--runs-root DIR] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hhe_ace2 import lag_times, assign_times, N_MEMBERS  # noqa: E402
from cluster_skill_analysis_sliding7d import CONUS_LAT_SLICE, CONUS_LON_SLICE  # noqa: E402

PJJA_DIR  = PROJECT_ROOT / "outputs/lag_may/precip_jja"
CACHE_DIR = PJJA_DIR / "ace2_precip_cache"
SEC_PER_DAY = 86400.0
LA, LO = CONUS_LAT_SLICE, CONUS_LON_SLICE


def member_precip_daily(pred_path: Path, init_time):
    """CONUS JJA daily-mean precip rate (mm/day) for one member, or None."""
    ds = xr.open_dataset(pred_path, decode_times=False)
    if "PRATEsfc" not in ds:
        ds.close()
        return None
    ds = assign_times(ds, init_time)
    pr = ds["PRATEsfc"]
    if "sample" in pr.dims:
        pr = pr.isel(sample=0)
    daily = (pr.resample(time="1D").mean() * SEC_PER_DAY).astype(np.float32)  # mm/day
    jja = daily.time.dt.month.isin([6, 7, 8])
    out = daily.sel(time=jja).isel(lat=LA, lon=LO)
    ds.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--runs-root", default=str(PJJA_DIR / "runs"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root)
    times = lag_times(args.year)

    n_ok = 0
    for idx in range(N_MEMBERS):
        cpath = CACHE_DIR / f"precip_y{args.year}_mem{idx:02d}.nc"
        if cpath.exists() and not args.force:
            n_ok += 1
            continue
        pred = runs_root / str(args.year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred.exists():
            print(f"  MISSING {pred}", flush=True)
            continue
        da = member_precip_daily(pred, times[idx])
        if da is None:
            print(f"  no PRATEsfc in {pred}", flush=True)
            continue
        tmp = cpath.with_suffix(".nc.tmp")
        xr.Dataset({"precip_mmday": da}).to_netcdf(tmp)
        tmp.replace(cpath)
        n_ok += 1
        print(f"  wrote {cpath.name}  (ndays={da.sizes['time']})", flush=True)

    print(f"[{args.year}] cached {n_ok}/{N_MEMBERS} members -> {CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
