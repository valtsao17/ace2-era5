#!/usr/bin/env python3
"""Extract lag IC files for Nov 1 of each year 1980-1989 from the ERA5 zarr.

Each file: 25 members, 6 h spacing, centred on Nov 1 00:00.
  member  0: Oct 29 00:00 (-72 h)
  member 12: Nov  1 00:00 (centre)
  member 24: Nov  3 00:00 (+72 h)  [actually Nov 4 00:00 for member 24 = +12*6=+72h]
Wait: centre + 12*6h = Nov 1 + 72h = Nov 4 00:00. Range: Oct 29 - Nov 4.
"""

import gcsfs
import zarr
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

BILLING_PROJECT = "ace2-era5"
ZARR_URL = (
    "gs://ai2cm-public-requester-pays/"
    "2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr"
)
ERA5_REF = datetime(1940, 1, 1, 12, 0, 0)
OUT_DIR = "/home/jovyan/ace2_lag_data/initial_conditions"

YEARS = list(range(1980, 1990))
N_MEMBERS = 25

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


def make_ic(store, year: int, lat, lon, out_path: str):
    times = lag_times(year)
    t_indices  = [zarr_idx(t) for t in times]
    time_vals  = [int((t - ERA5_REF).total_seconds() / 3600) for t in times]

    ds = nc.Dataset(out_path, "w")
    ds.createDimension("time", N_MEMBERS)
    ds.createDimension("latitude", len(lat))
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
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"))
        v.units     = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:] = data
        print(f"    {vname}", flush=True)

    ds.close()
    print(f"  wrote: {out_path}", flush=True)


def main():
    from pathlib import Path
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    fs    = gcsfs.GCSFileSystem(project=BILLING_PROJECT, requester_pays=True)
    store = zarr.open(fs.get_mapper(ZARR_URL), mode="r")
    lat   = store["latitude"][:]
    lon   = store["longitude"][:]

    for year in YEARS:
        out_path = f"{OUT_DIR}/ic_lag_{year}1101_25m.nc"
        if Path(out_path).exists():
            print(f"  SKIP {year}: already exists")
            continue
        print(f"\n=== {year} Nov 1 lag IC ===", flush=True)
        make_ic(store, year, lat, lon, out_path)

    print("\nAll IC files done.", flush=True)


if __name__ == "__main__":
    main()
