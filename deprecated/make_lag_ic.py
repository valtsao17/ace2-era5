#!/usr/bin/env python3
"""Extract 25 lag IC snapshots from ERA5 zarr centred on 2014-02-01T00:00.

Members are spaced 6 h apart:
  member  0  ->  2014-01-29T00:00  (12 steps before centre)
  member 12  ->  2014-02-01T00:00  (centre)
  member 24  ->  2014-02-03T00:00  (12 steps after centre)
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
OUT_PATH = "/home/jovyan/ace2_lag_data/initial_conditions/ic_lag_20140201_25m.nc"

N_MEMBERS = 25
CENTER = datetime(2014, 2, 1, 0, 0, 0)
LAG_TIMES = [CENTER + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]

IC_VARS = [
    "PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m",
    *[f"air_temperature_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"eastward_wind_{i}" for i in range(8)],
    *[f"northward_wind_{i}" for i in range(8)],
]


def zarr_idx(dt: datetime) -> int:
    return int((dt - ERA5_REF).total_seconds() / 3600 // 6)


def main():
    print("Lag init times:")
    for i, t in enumerate(LAG_TIMES):
        print(f"  member {i:02d}: {t.isoformat()}")

    fs = gcsfs.GCSFileSystem(project=BILLING_PROJECT, requester_pays=True)
    store = zarr.open(fs.get_mapper(ZARR_URL), mode="r")
    lat = store["latitude"][:]
    lon = store["longitude"][:]

    t_indices = [zarr_idx(t) for t in LAG_TIMES]
    time_values = [int((t - ERA5_REF).total_seconds() / 3600) for t in LAG_TIMES]

    ds = nc.Dataset(OUT_PATH, "w")
    ds.createDimension("time", N_MEMBERS)
    ds.createDimension("latitude", len(lat))
    ds.createDimension("longitude", len(lon))

    tv = ds.createVariable("time", "i8", ("time",))
    tv.units = "hours since 1940-01-01T12:00:00"
    tv.calendar = "proleptic_gregorian"
    tv[:] = time_values

    lv = ds.createVariable("latitude", "f4", ("latitude",))
    lv.units = "index"
    lv.long_name = "y-index of cell center points"
    lv[:] = lat

    lo = ds.createVariable("longitude", "f4", ("longitude",))
    lo.units = "index"
    lo.long_name = "x-index of cell center points"
    lo[:] = lon

    for vname in IC_VARS:
        src = store[vname]
        attrs = dict(src.attrs)
        data = src[t_indices, :, :]
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"))
        v.units = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:] = data
        print(f"  wrote {vname}", flush=True)

    ds.close()
    print(f"\nwrote: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
