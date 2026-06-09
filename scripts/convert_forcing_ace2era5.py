#!/usr/bin/env python3
"""Convert HiRO-ACE ERA5 forcing files to ACE2-ERA5 format.

Differences:
  - Time ref: "{year-1}-01-01T06:00:00" julian -> "1940-01-01T12:00:00" proleptic_gregorian
  - land_fraction: (time, lat, lon) 3D -> (lat, lon) 2D
  - ak/bk: HiRO-ACE values -> ACE2-ERA5 values
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np

SRC_DIR = Path("/home/jovyan/hiro-ace-test/data/hiro_ace/forcing_data")
DST_DIR = Path("/home/jovyan/ace2_lag_data/forcing_data_ace2era5")
DST_DIR.mkdir(parents=True, exist_ok=True)

ERA5_REF = datetime(1940, 1, 1, 12, 0, 0)  # ACE2-ERA5 time reference

# ACE2-ERA5 ak/bk (from allenai/ACE2-ERA5 forcing_2014.nc)
AK = [0.0, 5119.89501953125, 13881.3310546875, 19343.51171875, 20087.0859375,
      15596.6953125, 8880.453125, 3057.265625, 0.0]
BK = [0.0, 0.0, 0.005377814639359713, 0.059728413820266724, 0.2034912109375,
      0.43839120864868164, 0.6806430220603943, 0.8739292621612549, 1.0]

TIME_VARS = {"DSWRFtoa", "ocean_fraction", "sea_ice_fraction",
             "surface_temperature", "global_mean_co2"}
STATIC_2D = {"HGTsfc"}
SKIP_VARS  = {"latitude", "longitude", "time",
              *[f"ak_{i}" for i in range(9)],
              *[f"bk_{i}" for i in range(9)]}


def decode_hiro_time(tv) -> list[datetime]:
    """Decode HiRO-ACE time variable to list of datetimes."""
    units = tv.units          # e.g. "hours since 1979-01-01T06:00:00"
    parts = units.split("since")[1].strip().replace("T", " ")
    ref = datetime.fromisoformat(parts)
    return [ref + timedelta(hours=float(v)) for v in tv[:]]


def convert_year(year: int):
    src = SRC_DIR / f"forcing_{year}.nc"
    dst = DST_DIR / f"forcing_{year}.nc"
    if not src.exists():
        print(f"  SKIP {year}: source not found")
        return
    if dst.exists():
        print(f"  SKIP {year}: already converted")
        return

    print(f"  converting {year} ...", flush=True)
    s = nc.Dataset(src, "r")
    d = nc.Dataset(str(dst) + ".tmp", "w")

    lat = s["latitude"][:]
    lon = s["longitude"][:]
    abs_times = decode_hiro_time(s["time"])
    time_vals = [int((t - ERA5_REF).total_seconds() / 3600) for t in abs_times]
    n_t = len(time_vals)

    d.createDimension("time", n_t)
    d.createDimension("latitude", len(lat))
    d.createDimension("longitude", len(lon))

    tv = d.createVariable("time", "i8", ("time",))
    tv.units    = "hours since 1940-01-01T12:00:00"
    tv.calendar = "proleptic_gregorian"
    tv[:]       = time_vals

    lv = d.createVariable("latitude", "f4", ("latitude",))
    lv.units = s["latitude"].units; lv.long_name = s["latitude"].long_name
    lv[:] = lat

    lo = d.createVariable("longitude", "f4", ("longitude",))
    lo.units = s["longitude"].units; lo.long_name = s["longitude"].long_name
    lo[:] = lon

    # ak / bk scalars (replace with ACE2-ERA5 values)
    for i, (a, b) in enumerate(zip(AK, BK)):
        va = d.createVariable(f"ak_{i}", "f4"); va.units = "Pa"; va.long_name = "ak"
        va.assignValue(a)
        vb = d.createVariable(f"bk_{i}", "f4"); vb.units = "";  vb.long_name = "bk"
        vb.assignValue(b)

    # HGTsfc: static 2D (unchanged)
    hgt = d.createVariable("HGTsfc", "f4", ("latitude", "longitude"))
    hgt.units     = s["HGTsfc"].units
    hgt.long_name = getattr(s["HGTsfc"], "long_name", "HGTsfc")
    hgt[:] = s["HGTsfc"][:]

    # land_fraction: collapse from 3D -> 2D (take first time slice)
    lf = d.createVariable("land_fraction", "f4", ("latitude", "longitude"))
    lf.units     = s["land_fraction"].units
    lf.long_name = getattr(s["land_fraction"], "long_name", "land_fraction")
    src_lf = s["land_fraction"]
    if "time" in src_lf.dimensions:
        lf[:] = src_lf[0, :, :]
    else:
        lf[:] = src_lf[:]

    # Time-varying 3D / 1D variables (copy in time chunks to avoid OOM)
    CHUNK = 100
    for vname in TIME_VARS:
        if vname not in s.variables:
            continue
        sv = s[vname]
        dims = sv.dimensions
        if len(dims) == 3:
            dv = d.createVariable(vname, "f4", ("time", "latitude", "longitude"),
                                  zlib=True, complevel=1)
            for t0 in range(0, n_t, CHUNK):
                t1 = min(t0 + CHUNK, n_t)
                dv[t0:t1] = sv[t0:t1]
        else:
            dv = d.createVariable(vname, "f4", ("time",))
            dv[:] = sv[:]
        dv.units     = getattr(sv, "units", "")
        dv.long_name = getattr(sv, "long_name", vname)

    s.close()
    d.close()
    Path(str(dst) + ".tmp").replace(dst)
    print(f"    -> {dst}", flush=True)


def main():
    years = list(range(1980, 1991))  # 1980-1990 inclusive
    print(f"Converting {len(years)} forcing files to ACE2-ERA5 format ...")
    for year in years:
        convert_year(year)
    print("Done.")


if __name__ == "__main__":
    main()
