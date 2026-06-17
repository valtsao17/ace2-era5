#!/usr/bin/env python3
"""Generate lag ICs (1990-2000) and ACE2-ERA5 forcing (1991-2001) for the Nov 1 experiment.

Writes variables one at a time directly to the output file (no .tmp),
so partial progress is preserved across crashes. Checks completeness
by verifying all required variables exist in the file.

Usage:
    python make_lag_data_1990_2000.py --ic-year 1990
    python make_lag_data_1990_2000.py --forcing-year 1991
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from pathlib import Path

import gcsfs
import netCDF4 as nc
import numpy as np
import zarr
import zarr.storage

BILLING_PROJECT = "ace2-era5"
ZARR_URL = (
    "gs://ai2cm-public-requester-pays/"
    "2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr"
)
ERA5_REF    = datetime(1940, 1, 1, 12, 0, 0)
IC_DIR      = Path("/home/jovyan/ace2_lag_data/initial_conditions")
FORCING_DIR = Path("/home/jovyan/ace2_lag_data/forcing_data_ace2era5")
IC_DIR.mkdir(parents=True, exist_ok=True)
FORCING_DIR.mkdir(parents=True, exist_ok=True)

IC_YEARS      = list(range(1990, 2001))
FORCING_YEARS = list(range(1991, 2002))
N_MEMBERS = 25

AK = [0.0, 5119.89501953125, 13881.3310546875, 19343.51171875, 20087.0859375,
      15596.6953125, 8880.453125, 3057.265625, 0.0]
BK = [0.0, 0.0, 0.005377814639359713, 0.059728413820266724, 0.2034912109375,
      0.43839120864868164, 0.6806430220603943, 0.8739292621612549, 1.0]

IC_VARS = [
    "PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m",
    *[f"air_temperature_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"eastward_wind_{i}" for i in range(8)],
    *[f"northward_wind_{i}" for i in range(8)],
]

FORCING_TIME_VARS_3D = ["DSWRFtoa", "ocean_fraction", "sea_ice_fraction", "surface_temperature"]
FORCING_TIME_VARS_1D = ["global_mean_co2"]


def zarr_idx(dt: datetime) -> int:
    return max(0, int((dt - ERA5_REF).total_seconds() / 3600 // 6))


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def nc_vars(path: Path) -> set:
    """Return set of variable names in an existing netCDF4 file, or empty set."""
    try:
        with nc.Dataset(path, "r") as ds:
            return set(ds.variables.keys())
    except Exception:
        return set()


def make_ic(store, year: int, lat, lon):
    out = IC_DIR / f"ic_lag_{year}1101_25m.nc"
    existing = nc_vars(out)

    if existing >= set(IC_VARS):
        print(f"  IC {year}: complete (skip)", flush=True)
        return

    times     = lag_times(year)
    t_indices = [zarr_idx(t) for t in times]
    time_vals = [int((t - ERA5_REF).total_seconds() / 3600) for t in times]

    if existing:
        print(f"  IC {year}: resuming ({len(existing)} vars already written)", flush=True)
        mode = "a"
    else:
        print(f"  IC {year}: creating ...", flush=True)
        mode = "w"

    ds = nc.Dataset(out, mode)
    if mode == "w":
        ds.createDimension("time", N_MEMBERS)
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))

        tv = ds.createVariable("time", "i8", ("time",))
        tv.units = "hours since 1940-01-01T12:00:00"
        tv.calendar = "proleptic_gregorian"
        tv[:] = time_vals

        lv = ds.createVariable("latitude", "f4", ("latitude",))
        lv.units = "index"; lv.long_name = "y-index of cell center points"
        lv[:] = lat

        lo = ds.createVariable("longitude", "f4", ("longitude",))
        lo.units = "index"; lo.long_name = "x-index of cell center points"
        lo[:] = lon

    for vname in IC_VARS:
        if vname in existing:
            continue
        src   = store[vname]
        attrs = dict(src.attrs)
        data  = src[t_indices, :, :]
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"))
        v.units     = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:] = data
        ds.sync()
        print(f"    {vname}", flush=True)

    ds.close()
    print(f"  IC {year}: done -> {out}", flush=True)


def make_forcing(store, year: int, lat, lon):
    out = FORCING_DIR / f"forcing_{year}.nc"

    leap    = calendar.isleap(year)
    n_steps = (366 if leap else 365) * 4
    start   = datetime(year, 1, 1, 0, 0, 0)
    start_idx = zarr_idx(start)

    # Completeness check: all 3D time vars + 1D + static
    needed_vars = set(FORCING_TIME_VARS_3D + FORCING_TIME_VARS_1D + ["HGTsfc", "land_fraction"])
    existing = nc_vars(out)
    if existing >= needed_vars:
        print(f"  Forcing {year}: complete (skip)", flush=True)
        return

    time_vals = [int((start + timedelta(hours=6 * i) - ERA5_REF).total_seconds() / 3600)
                 for i in range(n_steps)]

    if existing:
        print(f"  Forcing {year}: resuming ({len(existing)} vars present)", flush=True)
        mode = "a"
    else:
        print(f"  Forcing {year}: creating ...", flush=True)
        mode = "w"

    ds = nc.Dataset(out, mode)
    if mode == "w":
        ds.createDimension("time", n_steps)
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))

        tv = ds.createVariable("time", "i8", ("time",))
        tv.units = "hours since 1940-01-01T12:00:00"
        tv.calendar = "proleptic_gregorian"
        tv[:] = time_vals

        lv = ds.createVariable("latitude", "f4", ("latitude",))
        lv.units = "degrees_north"; lv.long_name = "latitude"
        lv[:] = lat

        lo = ds.createVariable("longitude", "f4", ("longitude",))
        lo.units = "degrees_east"; lo.long_name = "longitude"
        lo[:] = lon

        for i, (a, b) in enumerate(zip(AK, BK)):
            va = ds.createVariable(f"ak_{i}", "f4"); va.units = "Pa"; va.long_name = "ak"
            va.assignValue(a)
            vb = ds.createVariable(f"bk_{i}", "f4"); vb.units = "1"; vb.long_name = "bk"
            vb.assignValue(b)

    if "HGTsfc" not in existing:
        src = store["HGTsfc"]
        hgt = ds.createVariable("HGTsfc", "f4", ("latitude", "longitude"))
        hgt.units = dict(src.attrs).get("units", "m")
        hgt.long_name = dict(src.attrs).get("long_name", "HGTsfc")
        hgt[:] = src[:]
        ds.sync()
        print(f"    HGTsfc", flush=True)

    if "land_fraction" not in existing:
        src = store["land_fraction"]
        lf = ds.createVariable("land_fraction", "f4", ("latitude", "longitude"))
        lf.units = dict(src.attrs).get("units", "1")
        lf.long_name = dict(src.attrs).get("long_name", "land_fraction")
        raw = src[:]
        lf[:] = raw[0, :, :] if raw.ndim == 3 else raw[:]
        ds.sync()
        print(f"    land_fraction", flush=True)

    # Time-varying 3D — read and write in chunks; sync after each chunk
    CHUNK = 60
    for vname in FORCING_TIME_VARS_3D:
        if vname not in store:
            continue
        if vname in existing:
            print(f"    {vname} (skip)", flush=True)
            continue
        src = store[vname]
        attrs = dict(src.attrs)
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"),
                              zlib=True, complevel=1)
        v.units     = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        for t0 in range(0, n_steps, CHUNK):
            t1 = min(t0 + CHUNK, n_steps)
            v[t0:t1] = src[start_idx + t0: start_idx + t1, :, :]
            ds.sync()
            print(f"    {vname} {t0}/{n_steps}", flush=True)
        print(f"    {vname} complete", flush=True)

    for vname in FORCING_TIME_VARS_1D:
        if vname not in store or vname in existing:
            continue
        src = store[vname]
        attrs = dict(src.attrs)
        v = ds.createVariable(vname, "f4", ("time",))
        v.units     = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:] = src[slice(start_idx, start_idx + n_steps)]
        ds.sync()
        print(f"    {vname}", flush=True)

    ds.close()
    print(f"  Forcing {year}: done -> {out}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ic-year",      type=int, default=None)
    p.add_argument("--forcing-year", type=int, default=None)
    args = p.parse_args()

    print("Connecting to zarr ...", flush=True)
    fs    = gcsfs.GCSFileSystem(project=BILLING_PROJECT, requester_pays=True)
    zpath = ZARR_URL.replace("gs://", "")
    zstore = zarr.storage.FsspecStore(fs, path=zpath, read_only=True)
    store = zarr.open_group(zstore, mode="r")
    lat   = store["latitude"][:]
    lon   = store["longitude"][:]
    print(f"  zarr open: {len(lat)} lat × {len(lon)} lon", flush=True)

    if args.ic_year is not None:
        make_ic(store, args.ic_year, lat, lon)
    elif args.forcing_year is not None:
        make_forcing(store, args.forcing_year, lat, lon)
    else:
        for year in IC_YEARS:
            make_ic(store, year, lat, lon)
        for year in FORCING_YEARS:
            make_forcing(store, year, lat, lon)

    print("Done.", flush=True)
    import os; os._exit(0)   # bypass gcsfs/aiohttp teardown segfault


if __name__ == "__main__":
    main()
