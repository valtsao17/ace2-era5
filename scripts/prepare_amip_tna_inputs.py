#!/usr/bin/env python3
"""Prepare one year of ACE2 AMIP or AMIP-TNA inputs.

``global`` keeps the observed ACE2/ERA5 SST, SIC, and initial-condition files.
``tna`` keeps observed SST/SIC only in 0--23N, 80--35W and replaces the rest
of the ocean with the supplied 1980--2022 calendar climatology.  The files
are written under the usual ``outputs/.../inputs/<tag>`` layout so the
existing ACE2 inference runner can consume them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr


def dt_keys(values) -> list[tuple[int, int, int]]:
    if hasattr(values, "dt"):
        return list(zip(
            np.asarray(values.dt.month, dtype=int).tolist(),
            np.asarray(values.dt.day, dtype=int).tolist(),
            np.asarray(values.dt.hour, dtype=int).tolist(),
        ))
    return [(int(x.month), int(x.day), int(x.hour)) for x in values]


def box_mask(lat: np.ndarray, lon: np.ndarray, south: float, north: float, west: float, east: float) -> np.ndarray:
    lon360 = np.asarray(lon, dtype=float) % 360.0
    in_lon = (lon360 >= west) & (lon360 <= east)
    return (np.asarray(lat)[:, None] >= south) & (np.asarray(lat)[:, None] <= north) & in_lon[None, :]


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def load_climatology(path: Path):
    with xr.open_dataset(path) as ds:
        time = ds["time"].load()
        lat = np.asarray(ds["latitude"], dtype=float)
        lon = np.asarray(ds["longitude"], dtype=float)
        sst = np.asarray(ds["surface_temperature_climatology"], dtype=np.float32)
        sic = np.asarray(ds["sea_ice_fraction_climatology"], dtype=np.float32)
    lookup = {key: i for i, key in enumerate(dt_keys(time))}
    return lookup, lat, lon, sst, sic


def prepare_tna_forcing(source: Path, destination: Path, clim_path: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    clim_lookup, clim_lat, clim_lon, clim_sst, clim_sic = load_climatology(clim_path)
    with nc.Dataset(source) as src:
        lat = np.asarray(src.variables["latitude"][:], dtype=float)
        lon = np.asarray(src.variables["longitude"][:], dtype=float)
    if not (np.allclose(lat, clim_lat) and np.allclose(lon, clim_lon)):
        raise ValueError("forcing and climatology grids do not match")
    tna = box_mask(lat, lon, 0.0, 23.0, 280.0, 325.0)

    with nc.Dataset(destination, "r+") as ds:
        dates = nc.num2date(
            ds.variables["time"][:],
            ds.variables["time"].units,
            calendar=getattr(ds.variables["time"], "calendar", "standard"),
        )
        forcing_keys = dt_keys(dates)
        st = ds.variables["surface_temperature"]
        sic = ds.variables["sea_ice_fraction"]
        ocean = ds.variables["ocean_fraction"]
        for i, key in enumerate(forcing_keys):
            j = clim_lookup[key]
            outside = (np.asarray(ocean[i]) > 0.5) & (~tna)
            values = np.asarray(st[i], dtype=np.float32)
            values[outside] = clim_sst[j][outside]
            st[i] = values
            values_ice = np.asarray(sic[i], dtype=np.float32)
            values_ice[outside] = clim_sic[j][outside]
            sic[i] = values_ice
        ds.setncattr("sst_experiment_role", "AMIP-TNA: observed SST/SIC in TNA, climatological SST/SIC elsewhere")
        ds.setncattr("tna_bounds", "0N-23N, 80W-35W")
    print(f"wrote {destination}", flush=True)


def prepare_tna_ic(source: Path, destination: Path, forcing: Path, clim_path: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)
    clim_lookup, clim_lat, clim_lon, clim_sst, _ = load_climatology(clim_path)
    with nc.Dataset(forcing) as f:
        flat = np.asarray(f.variables["latitude"][:], dtype=float)
        flon = np.asarray(f.variables["longitude"][:], dtype=float)
        fdates = nc.num2date(
            f.variables["time"][:], f.variables["time"].units,
            calendar=getattr(f.variables["time"], "calendar", "standard"),
        )
        flookup = {key: i for i, key in enumerate(dt_keys(fdates))}
        ocean_by_key = {key: np.asarray(f.variables["ocean_fraction"][i]) > 0.5 for key, i in flookup.items()}
    if not (np.allclose(flat, clim_lat) and np.allclose(flon, clim_lon)):
        raise ValueError("forcing and climatology grids do not match")
    tna = box_mask(flat, flon, 0.0, 23.0, 280.0, 325.0)

    with nc.Dataset(destination, "r+") as ds:
        dates = nc.num2date(
            ds.variables["time"][:], ds.variables["time"].units,
            calendar=getattr(ds.variables["time"], "calendar", "standard"),
        )
        st = ds.variables["surface_temperature"]
        for i, key in enumerate(dt_keys(dates)):
            if key not in clim_lookup or key not in ocean_by_key:
                raise KeyError(f"missing climatology/forcing calendar slot {key}")
            outside = ocean_by_key[key] & (~tna)
            values = np.asarray(st[i], dtype=np.float32)
            values[outside] = clim_sst[clim_lookup[key]][outside]
            st[i] = values
        ds.setncattr("sst_experiment_role", "AMIP-TNA climatological SST outside TNA in initial conditions")
        ds.setncattr("tna_bounds", "0N-23N, 80W-35W")
    print(f"wrote {destination}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--forcing-dir", required=True, type=Path)
    p.add_argument("--ic-dir", required=True, type=Path)
    p.add_argument("--climatology", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=("global", "tna"), required=True)
    p.add_argument("--init-month", type=int, default=5)
    p.add_argument("--init-day", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    source_forcing = args.forcing_dir / f"forcing_{args.year}.nc"
    source_ic = args.ic_dir / f"ic_lag_{args.year}{args.init_month:02d}{args.init_day:02d}_25m.nc"
    if not source_forcing.exists() or not source_ic.exists():
        raise FileNotFoundError(f"missing source forcing or IC for {args.year}")
    if args.mode == "tna" and not args.climatology.exists():
        raise FileNotFoundError(args.climatology)

    root = args.out_root / "inputs" / args.tag
    forcing_out = root / "forcing" / source_forcing.name
    ic_out = root / "initial_conditions" / source_ic.name
    if args.mode == "global":
        link(source_forcing, forcing_out)
        link(source_ic, ic_out)
    else:
        prepare_tna_forcing(source_forcing, forcing_out, args.climatology, args.overwrite)
        prepare_tna_ic(source_ic, ic_out, source_forcing, args.climatology, args.overwrite)

    manifest = {
        "year": args.year,
        "mode": args.mode,
        "tag": args.tag,
        "members": 25,
        "observed_region": "TNA 0-23N, 80-35W" if args.mode == "tna" else "global ocean",
        "climatology": str(args.climatology.resolve()) if args.mode == "tna" else None,
        "forcing": str(forcing_out),
        "initial_conditions": str(ic_out),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"manifest_{args.year}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
