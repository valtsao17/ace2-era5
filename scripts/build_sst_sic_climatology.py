#!/usr/bin/env python3
"""Build a 6-hourly calendar climatology for SST and sea ice fraction.

The AMIP-TNA experiment needs both fields: observed SST and SIC are retained
inside the TNA, while the 1980--2022 calendar climatology is used elsewhere.
The output uses a leap-year template (default 2000) and is aligned by
(month, day, hour), so it can be applied to any target year.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def years(spec: str) -> list[int]:
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":", 1))
        return list(range(a, b + 1))
    if "-" in spec and "," not in spec:
        a, b = (int(x) for x in spec.split("-", 1))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def keys(time: xr.DataArray) -> list[tuple[int, int, int]]:
    return list(zip(
        np.asarray(time.dt.month, dtype=int).tolist(),
        np.asarray(time.dt.day, dtype=int).tolist(),
        np.asarray(time.dt.hour, dtype=int).tolist(),
    ))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forcing-dir", required=True, type=Path)
    p.add_argument("--years", default="1980:2022")
    p.add_argument("--template-year", type=int, default=2000)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    template_path = args.forcing_dir / f"forcing_{args.template_year}.nc"
    if not template_path.exists():
        raise FileNotFoundError(template_path)
    with xr.open_dataset(template_path) as ds:
        target_time = ds["time"].load()
        lat = np.asarray(ds["latitude"], dtype=np.float32)
        lon = np.asarray(ds["longitude"], dtype=np.float32)
    target_keys = keys(target_time)
    lookup = {key: i for i, key in enumerate(target_keys)}
    shape = (len(target_keys), len(lat), len(lon))
    total_sst = np.zeros(shape, dtype=np.float64)
    total_sic = np.zeros(shape, dtype=np.float64)
    count = np.zeros(len(target_keys), dtype=np.int16)

    for year in years(args.years):
        path = args.forcing_dir / f"forcing_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            source_keys = keys(ds["time"])
            pairs = [(i, lookup[k]) for i, k in enumerate(source_keys) if k in lookup]
            source_i = np.asarray([x[0] for x in pairs], dtype=int)
            target_i = np.asarray([x[1] for x in pairs], dtype=int)
            sst = np.asarray(ds["surface_temperature"].isel(time=source_i), dtype=np.float64)
            sic = np.asarray(ds["sea_ice_fraction"].isel(time=source_i), dtype=np.float64)
        total_sst[target_i] += sst
        total_sic[target_i] += sic
        count[target_i] += 1
        print(f"accumulated {year}", flush=True)

    if np.any(count == 0):
        missing = [target_keys[i] for i in np.where(count == 0)[0][:10]]
        raise ValueError(f"calendar slots without samples: {missing}")
    sst_clim = (total_sst / count[:, None, None]).astype(np.float32)
    sic_clim = (total_sic / count[:, None, None]).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        {
            "surface_temperature_climatology": (("time", "latitude", "longitude"), sst_clim),
            "sea_ice_fraction_climatology": (("time", "latitude", "longitude"), sic_clim),
            "sample_count": (("time",), count),
        },
        coords={"time": target_time.values, "latitude": lat, "longitude": lon},
        attrs={
            "description": "6-hourly 1980-2022 calendar climatology of ACE2 prescribed SST and SIC",
            "years": args.years,
            "template_year": args.template_year,
            "alignment": "calendar month/day/hour",
        },
    )
    encoding = {
        "surface_temperature_climatology": {"zlib": True, "complevel": 1, "chunksizes": (32, len(lat), len(lon))},
        "sea_ice_fraction_climatology": {"zlib": True, "complevel": 1, "chunksizes": (32, len(lat), len(lon))},
    }
    out.to_netcdf(args.out, encoding=encoding)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
