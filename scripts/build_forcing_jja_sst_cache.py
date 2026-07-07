#!/usr/bin/env python3
"""Build yearly JJA SST cache from ACE2 ERA5 forcing files.

This is the SST source used by the HHE teleconnection scripts: the prescribed
ERA5 lower-boundary surface_temperature in
data/lag_data/forcing_data_ace2era5/forcing_<year>.nc. It is already on the
ACE2/HHE grid and is the field ACE2 was forced with.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORCING_DIR = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"
DEFAULT_OUT = PROJECT_ROOT / "outputs/lag_may/extreme_sst_pipeline/era5_forcing_jja_sst_mean.nc"


def parse_years(spec: str, forcing_dir: Path) -> list[int]:
    if spec == "all":
        years = []
        for path in sorted(forcing_dir.glob("forcing_*.nc")):
            stem = path.stem
            try:
                years.append(int(stem.split("_")[-1]))
            except ValueError:
                continue
        if not years:
            raise FileNotFoundError(f"no forcing_*.nc files under {forcing_dir}")
        return years
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def coord_name(ds: xr.Dataset, *names: str) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"missing coordinate among {names}")


def _static_or_mean(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da.mean("time", skipna=True)
    return da


def jja_mean_for_year(path: Path, year: int, ocean_min: float, max_sea_ice: float | None):
    with xr.open_dataset(path) as ds:
        if "surface_temperature" not in ds:
            raise KeyError(f"{path} has no surface_temperature variable")
        lat_name = coord_name(ds, "lat", "latitude", "y")
        lon_name = coord_name(ds, "lon", "longitude", "x")

        st = ds["surface_temperature"]
        if "time" not in st.dims:
            raise ValueError(f"{path} surface_temperature has no time dimension")

        time = st["time"]
        tmask = (time.dt.year.values.astype(int) == year) & np.isin(
            time.dt.month.values.astype(int), [6, 7, 8]
        )
        if not np.any(tmask):
            raise ValueError(f"{path} has no JJA samples for {year}")

        mean = st.isel(time=tmask).mean("time", skipna=True)

        mask = xr.ones_like(mean, dtype=bool)
        if "ocean_fraction" in ds:
            ocn = _static_or_mean(ds["ocean_fraction"])
            mask = mask & (ocn > ocean_min)
        if max_sea_ice is not None and "sea_ice_fraction" in ds:
            ice = ds["sea_ice_fraction"].isel(time=tmask).mean("time", skipna=True)
            mask = mask & (ice <= max_sea_ice)

        mean = mean.where(mask)
        arr = np.asarray(mean.transpose(lat_name, lon_name).values, dtype=np.float32)
        lat = np.asarray(ds[lat_name].values, dtype=np.float32)
        lon = np.asarray(ds[lon_name].values, dtype=np.float32)
        return arr, lat, lon


def atomic_write(ds: xr.Dataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all", help="'all', START:END, or comma-separated years")
    p.add_argument("--forcing-dir", type=Path, default=DEFAULT_FORCING_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ocean-min", type=float, default=0.5)
    p.add_argument("--max-sea-ice", type=float, default=0.15)
    args = p.parse_args()

    years = parse_years(args.years, args.forcing_dir)
    arrays, lat, lon = [], None, None
    for year in years:
        path = args.forcing_dir / f"forcing_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        arr, la, lo = jja_mean_for_year(path, year, args.ocean_min, args.max_sea_ice)
        if lat is None:
            lat, lon = la, lo
        arrays.append(arr)
        print(f"  forcing SST JJA {year}: mean={float(np.nanmean(arr)):.3f} K", flush=True)

    out = xr.Dataset(
        {
            "surface_temperature": (
                ("year", "lat", "lon"),
                np.stack(arrays, axis=0).astype(np.float32),
            )
        },
        coords={"year": years, "lat": lat, "lon": lon},
        attrs={
            "source": str(args.forcing_dir),
            "description": "JJA mean prescribed ERA5 surface_temperature from ACE2 forcing files",
            "ocean_mask": f"ocean_fraction > {args.ocean_min}, sea_ice_fraction <= {args.max_sea_ice}",
        },
    )
    atomic_write(out, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
