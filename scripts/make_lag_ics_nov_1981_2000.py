#!/usr/bin/env python3
"""Generate Nov 1 lag-ensemble IC files for 1981-2001 (years missing from disk).

Uses ace2-ic-download billing project (requester-pays bucket).
Writes to: /home/vt55/ace2/data/lag_data/initial_conditions/
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import gcsfs
import zarr
import numpy as np
import netCDF4 as nc

BILLING_PROJECT = "ace2-ic-download"
ZARR_URL = (
    "gs://ai2cm-public-requester-pays/"
    "2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr"
)
ERA5_REF  = datetime(1940, 1, 1, 12, 0, 0)
OUT_DIR   = Path("/home/vt55/ace2/data/lag_data/initial_conditions")
N_MEMBERS = 25

# 1981-2000 are missing; 2001 already exists
YEARS = list(range(1981, 2001))

IC_VARS = [
    "PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m",
    *[f"air_temperature_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"eastward_wind_{i}" for i in range(8)],
    *[f"northward_wind_{i}" for i in range(8)],
]


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def zarr_idx(dt: datetime) -> int:
    return max(0, int((dt - ERA5_REF).total_seconds() / 3600 // 6))


def make_ic(store, year: int, lat, lon, out_path: Path):
    times      = lag_times(year)
    t_indices  = [zarr_idx(t) for t in times]
    time_vals  = [int((t - ERA5_REF).total_seconds() / 3600) for t in times]

    ds = nc.Dataset(str(out_path), "w")
    ds.createDimension("time",      N_MEMBERS)
    ds.createDimension("latitude",  len(lat))
    ds.createDimension("longitude", len(lon))

    tv = ds.createVariable("time", "i8", ("time",))
    tv.units    = "hours since 1940-01-01T12:00:00"
    tv.calendar = "proleptic_gregorian"
    tv[:]       = time_vals

    lv = ds.createVariable("latitude", "f4", ("latitude",))
    lv.units = "index"; lv.long_name = "y-index of cell center points"
    lv[:] = lat

    lo = ds.createVariable("longitude", "f4", ("longitude",))
    lo.units = "index"; lo.long_name = "x-index of cell center points"
    lo[:] = lon

    for vname in IC_VARS:
        src   = store[vname]
        attrs = dict(src.attrs)
        data  = src[t_indices, :, :]
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"),
                              zlib=True, complevel=1)
        v.units     = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:]        = data

    ds.close()
    print(f"  wrote: {out_path}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs    = gcsfs.GCSFileSystem(project=BILLING_PROJECT, requester_pays=True,
                                token="google_default")
    store = zarr.open(fs.get_mapper(ZARR_URL), mode="r")
    lat   = store["latitude"][:]
    lon   = store["longitude"][:]

    for year in YEARS:
        out_path = OUT_DIR / f"ic_lag_{year}1101_25m.nc"
        if out_path.exists():
            print(f"SKIP {year}: already exists", flush=True)
            continue
        print(f"\n=== {year} Nov 1 lag IC ===", flush=True)
        make_ic(store, year, lat, lon, out_path)

    print("\nAll IC files done.", flush=True)


if __name__ == "__main__":
    main()
