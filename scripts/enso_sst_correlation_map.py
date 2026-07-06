#!/usr/bin/env python3
"""ENSO/Nino3.4 seasonal-temperature correlation maps for ERA5 and ACE2.

For each requested year, this script builds a seasonal-mean temperature cube
(year, lat, lon), builds a seasonal Nino3.4 index, and correlates each grid
cell's seasonal temperature time series against that index. It writes a side-by-side
ERA5 vs ACE2 figure, NetCDF fields, and a small JSON summary with a weighted
spatial pattern correlation between the two maps.

Default Nino3.4 box:
  5S-5N, 170W-120W == lat -5..5, lon 190..240 in 0-360 degrees.

Typical cluster use:
  python scripts/enso_sst_correlation_map.py \
    --root /home/vt55/ace2 \
    --years 1980:2016 \
    --ace2-runs-root /home/vt55/ace2/outputs/lag_may/runs \
    --ace2-sst-var TMP2m

The default map field is TMP2m. If you set --ace2-sst-var surface_temperature,
remember that ACE2 surface_temperature is usually prescribed lower-boundary SST
forcing in this experiment, so ERA5 and ACE2 maps can be identical.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("ACE2_ROOT", PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_COAST_WARNED = False
PRESCRIBED_SST_NOTE = (
    "ACE2 surface_temperature is usually the prescribed lower-boundary/forcing "
    "SST in this experiment, not an independent prognostic SST forecast. If the "
    "ERA5 and ACE2 SST cubes are identical, identical correlation maps are expected."
)

FIELD_CANDIDATES = (
    "TMP2m",
    "2m_temperature",
    "t2m",
    "temperature_2m",
    "sst",
    "SST",
    "tos",
    "sea_surface_temperature",
    "surface_temperature",
)
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            start, end = [int(x) for x in piece.split(":", 1)]
            out.extend(range(start, end + 1))
        elif "-" in piece:
            start, end = [int(x) for x in piece.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(piece))
    return sorted(dict.fromkeys(out))


def parse_int_list(spec: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("Need at least one month.")
    return vals


def parse_box(vals: list[float]) -> tuple[float, float, float, float]:
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("Box needs four values: lat_s lat_n lon_w lon_e")
    lat_s, lat_n, lon_w, lon_e = [float(x) for x in vals]
    if lat_n < lat_s:
        raise argparse.ArgumentTypeError("lat_n must be >= lat_s")
    return lat_s, lat_n, lon_w % 360.0, lon_e % 360.0


def find_coord_name(da: xr.DataArray | xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    raise KeyError(f"Could not find coordinate among {candidates}")


def pick_var(ds: xr.Dataset, requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in ds.data_vars:
            lower = {name.lower(): name for name in ds.data_vars}
            if requested.lower() in lower:
                return lower[requested.lower()]
            tmp2m_aliases = {"tmp2m", "2m_temperature", "t2m", "temperature_2m", "2t"}
            if requested.lower() in tmp2m_aliases:
                for name in ds.data_vars:
                    if name.lower() in tmp2m_aliases:
                        return name
            raise KeyError(f"Variable {requested!r} not found. Available: {list(ds.data_vars)}")
        return requested
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not infer field variable. Available: {list(ds.data_vars)}")


def ensure_ascending_lat(arr: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float32)
    if lat.size < 2 or lat[0] <= lat[-1]:
        return arr, lat
    return np.flip(arr, axis=-2), lat[::-1]


def normalize_lon(arr: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_360 = np.asarray(lon, dtype=np.float32) % 360.0
    order = np.argsort(lon_360)
    return np.take(arr, order, axis=-1), lon_360[order]


def collapse_extra_dims(da: xr.DataArray, keep: set[str]) -> xr.DataArray:
    extra = [dim for dim in da.dims if dim not in keep]
    if extra:
        da = da.mean(extra, skipna=True)
    return da


def select_years_from_year_coord(da: xr.DataArray, years: list[int]) -> xr.DataArray:
    available = [int(y) for y in da["year"].values]
    missing = [y for y in years if y not in available]
    if missing:
        raise ValueError(f"Missing requested years in year coordinate: {missing[:10]}")
    return da.sel(year=years)


def finalize_cube(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    da = da.transpose("year", lat_name, lon_name)
    arr = np.asarray(da.values, dtype=np.float32) * float(scale)
    lat = np.asarray(da[lat_name].values, dtype=np.float32)
    lon = np.asarray(da[lon_name].values, dtype=np.float32)
    arr, lat = ensure_ascending_lat(arr, lat)
    arr, lon = normalize_lon(arr, lon)
    years = [int(y) for y in da["year"].values]
    return arr, lat, lon, years


def yearly_cube_from_nc(
    path: Path,
    var_name: str | None,
    years: list[int],
    season_months: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        var = pick_var(ds, var_name, FIELD_CANDIDATES)
        da = ds[var]
        lat_name = find_coord_name(da, LAT_NAMES)
        lon_name = find_coord_name(da, LON_NAMES)

        if "year" in da.dims:
            da = select_years_from_year_coord(da, years)
            da = collapse_extra_dims(da, {"year", lat_name, lon_name})
        elif "time" in da.dims:
            time = da["time"]
            mask = np.isin(time.dt.year.values.astype(int), years)
            mask &= np.isin(time.dt.month.values.astype(int), season_months)
            da = da.isel(time=mask).groupby("time.year").mean("time", skipna=True)
            da = select_years_from_year_coord(da, years)
            da = collapse_extra_dims(da, {"year", lat_name, lon_name})
        else:
            raise ValueError(f"{path} variable {var!r} needs a time or year dimension")

        arr, lat, lon, year_out = finalize_cube(da, lat_name, lon_name, scale)
    return arr, lat, lon, year_out, var


def yearly_cube_from_pattern(
    pattern: str,
    var_name: str | None,
    years: list[int],
    season_months: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    cubes: list[np.ndarray] = []
    lat_ref = lon_ref = None
    used_var = ""
    for year in years:
        path = Path(pattern.format(year=year, yyyy=year))
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            var = pick_var(ds, var_name, FIELD_CANDIDATES)
            used_var = var
            da = ds[var]
            lat_name = find_coord_name(da, LAT_NAMES)
            lon_name = find_coord_name(da, LON_NAMES)
            if "time" in da.dims:
                mask = np.isin(da["time"].dt.month.values.astype(int), season_months)
                da = da.isel(time=mask).mean("time", skipna=True)
            da = collapse_extra_dims(da, {lat_name, lon_name})
            arr = np.asarray(da.transpose(lat_name, lon_name).values, dtype=np.float32)
            arr = arr[np.newaxis] * float(scale)
            lat = np.asarray(da[lat_name].values, dtype=np.float32)
            lon = np.asarray(da[lon_name].values, dtype=np.float32)
            arr, lat = ensure_ascending_lat(arr, lat)
            arr, lon = normalize_lon(arr, lon)
        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        elif not (np.allclose(lat_ref, lat) and np.allclose(lon_ref, lon)):
            raise ValueError(f"Grid changed in {path}")
        cubes.append(arr[0])
    return np.stack(cubes).astype(np.float32), lat_ref, lon_ref, years, used_var


def yearly_cube_from_forcing_dir(
    forcing_dir: Path,
    var_name: str | None,
    years: list[int],
    season_months: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    pattern = str(forcing_dir / "forcing_{year}.nc")
    return yearly_cube_from_pattern(pattern, var_name or "surface_temperature", years, season_months, scale)


def is_tmp2m_var(var_name: str | None) -> bool:
    if var_name is None:
        return False
    return var_name.lower() in {"tmp2m", "2m_temperature", "t2m", "temperature_2m", "2t"}


def complete_yearly_field(path: Path, var: str, year: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return var in ds and int(ds["year"].values[0]) == int(year)
    except Exception:
        return False


def target_lon_for_source(src_lon: np.ndarray, target_lon_360: np.ndarray) -> np.ndarray:
    src = np.asarray(src_lon, dtype=np.float64)
    target = np.asarray(target_lon_360, dtype=np.float64)
    if np.nanmax(src) <= 180.0 and np.nanmax(target) > 180.0:
        return np.where(target > 180.0, target - 360.0, target)
    if np.nanmin(src) >= 0.0 and np.nanmin(target) < 0.0:
        return np.where(target < 0.0, target + 360.0, target)
    return target


def yearly_cube_era5_tmp2m_from_arco(
    cache_dir: Path,
    zarr_url: str,
    years: list[int],
    season_months: tuple[int, ...],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    """Build/load ERA5 seasonal mean 2m temperature on the ACE2 grid."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    yearly_paths = {year: cache_dir / f"era5_tmp2m_jja_y{year}.nc" for year in years}
    if not all(complete_yearly_field(path, "TMP2m", year) for year, path in yearly_paths.items()):
        print("Opening ARCO ERA5 for ERA5 TMP2m JJA means ...", flush=True)
        import gcsfs  # noqa: F401

        ds = xr.open_dataset(zarr_url, engine="zarr", chunks={}, storage_options={"token": "anon"})
        try:
            var = pick_var(ds, "2m_temperature", FIELD_CANDIDATES)
            da0 = ds[var]
            lat_name = find_coord_name(da0, LAT_NAMES)
            lon_name = find_coord_name(da0, LON_NAMES)
            sel_lon = target_lon_for_source(np.asarray(da0[lon_name].values), target_lon)
            for year in years:
                path = yearly_paths[year]
                if complete_yearly_field(path, "TMP2m", year):
                    print(f"  {year}: using yearly ERA5 TMP2m cache {path.name}", flush=True)
                    continue
                print(f"  {year}: ERA5 TMP2m seasonal mean", flush=True)
                da = da0.sel(time=slice(f"{year}-06-01", f"{year}-08-31T23:00:00"))
                da = da.sel(time=da.time.dt.month.isin(season_months))
                da = da.sel(time=da.time.dt.hour.isin([0, 6, 12, 18]))
                da = da.sel({lat_name: target_lat, lon_name: sel_lon}, method="nearest")
                da = da.mean("time", skipna=True).load()
                arr = np.asarray(da.values, dtype=np.float32) * float(scale)
                out = xr.Dataset(
                    {"TMP2m": (("year", "lat", "lon"), arr[np.newaxis, :, :].astype(np.float32))},
                    coords={"year": [year], "lat": target_lat.astype(np.float32), "lon": target_lon.astype(np.float32)},
                    attrs={
                        "source": zarr_url,
                        "variable": var,
                        "time_sampling": "00/06/12/18 UTC samples",
                        "note": "ERA5 2m temperature regridded nearest to ACE2 grid before load.",
                    },
                )
                out["TMP2m"].attrs["units"] = "K"
                out.to_netcdf(path)
        finally:
            ds.close()

    pieces = []
    for year in years:
        with xr.open_dataset(yearly_paths[year]) as ds_y:
            pieces.append(ds_y["TMP2m"].load())
    da_all = xr.concat(pieces, dim="year").assign_coords(year=years)
    arr = np.asarray(da_all.values, dtype=np.float32)
    return arr, target_lat.astype(np.float32), target_lon.astype(np.float32), years, "TMP2m"


def lag_times(year: int, init_month: int, init_day: int, n_members: int) -> list[datetime]:
    center = datetime(year, init_month, init_day, 0, 0, 0)
    mid = (n_members - 1) / 2.0
    return [center + timedelta(hours=6 * (idx - mid)) for idx in range(n_members)]


def assign_run_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    time = ds["time"]
    if np.issubdtype(time.dtype, np.datetime64):
        return ds
    vals = time.values.astype(np.int64)
    return ds.assign_coords(time=pd.Timestamp(init_time) + pd.to_timedelta(vals, unit="us"))


def yearly_cube_from_ace2_runs(
    runs_root: Path,
    var_name: str,
    years: list[int],
    season_months: tuple[int, ...],
    scale: float,
    n_members: int,
    init_month: int,
    init_day: int,
    min_members: int,
    allow_partial: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], str, list[int]]:
    cubes: list[np.ndarray] = []
    kept_years: list[int] = []
    member_counts: list[int] = []
    lat_ref = lon_ref = None

    for year in years:
        member_means: list[np.ndarray] = []
        times = lag_times(year, init_month, init_day, n_members)
        missing_var_examples: list[Path] = []
        for member, init_time in enumerate(times):
            path = runs_root / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc"
            if not path.exists() or path.stat().st_size == 0:
                if allow_partial:
                    continue
                raise FileNotFoundError(path)
            with xr.open_dataset(path, decode_times=False) as ds_raw:
                if var_name not in ds_raw.data_vars:
                    missing_var_examples.append(path)
                    if allow_partial:
                        continue
                    raise KeyError(f"{var_name!r} not in {path}")
                ds = assign_run_times(ds_raw, init_time)
                da = ds[var_name]
                if "sample" in da.dims:
                    da = da.isel(sample=0)
                lat_name = find_coord_name(da, LAT_NAMES)
                lon_name = find_coord_name(da, LON_NAMES)
                mask = np.isin(da["time"].dt.month.values.astype(int), season_months)
                da = da.isel(time=mask).mean("time", skipna=True)
                da = collapse_extra_dims(da, {lat_name, lon_name})
                arr = np.asarray(da.transpose(lat_name, lon_name).values, dtype=np.float32)
                arr = arr[np.newaxis] * float(scale)
                lat = np.asarray(da[lat_name].values, dtype=np.float32)
                lon = np.asarray(da[lon_name].values, dtype=np.float32)
                arr, lat = ensure_ascending_lat(arr, lat)
                arr, lon = normalize_lon(arr, lon)
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            elif not (np.allclose(lat_ref, lat) and np.allclose(lon_ref, lon)):
                raise ValueError(f"Grid changed in {path}")
            member_means.append(arr[0])

        if len(member_means) < min_members:
            hint = ""
            if missing_var_examples:
                hint = (
                    f" First file without {var_name!r}: {missing_var_examples[0]}. "
                    f"Rerun inference with --writer-names including {var_name}."
                )
            raise RuntimeError(
                f"Only {len(member_means)} usable {var_name} members for {year} under {runs_root}; "
                f"need at least {min_members}.{hint}"
            )
        cubes.append(np.nanmean(np.stack(member_means, axis=0), axis=0).astype(np.float32))
        kept_years.append(year)
        member_counts.append(len(member_means))
        print(f"  ACE2 {var_name} {year}: {len(member_means)}/{n_members} members", flush=True)

    return np.stack(cubes).astype(np.float32), lat_ref, lon_ref, kept_years, var_name, member_counts


def collapse_mask_field(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    extra = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    if extra:
        da = da.mean(extra, skipna=True)
    return da


def load_ocean_mask_from_forcing(
    forcing_dir: Path,
    years: list[int],
    season_months: tuple[int, ...],
    lat_ref: np.ndarray,
    lon_ref: np.ndarray,
    ocean_min: float,
    max_sea_ice: float,
) -> np.ndarray:
    ocean_mask = None
    ice_means: list[np.ndarray] = []
    for year in years:
        path = forcing_dir / f"forcing_{year}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            template = ds["surface_temperature"] if "surface_temperature" in ds else next(iter(ds.data_vars.values()))
            lat_name = find_coord_name(template, LAT_NAMES)
            lon_name = find_coord_name(template, LON_NAMES)
            if ocean_mask is None and "ocean_fraction" in ds:
                ocn = collapse_mask_field(ds["ocean_fraction"], lat_name, lon_name)
                arr = np.asarray(ocn.transpose(lat_name, lon_name).values, dtype=np.float32)
                arr = arr[np.newaxis]
                lat = np.asarray(ocn[lat_name].values, dtype=np.float32)
                lon = np.asarray(ocn[lon_name].values, dtype=np.float32)
                arr, lat = ensure_ascending_lat(arr, lat)
                arr, lon = normalize_lon(arr, lon)
                check_grid(lat, lon, lat_ref, lon_ref, "ocean_fraction")
                ocean_mask = arr[0] > ocean_min
            if "sea_ice_fraction" in ds:
                ice = ds["sea_ice_fraction"]
                if "time" in ice.dims:
                    mask = np.isin(ice["time"].dt.month.values.astype(int), season_months)
                    ice = ice.isel(time=mask)
                ice = collapse_mask_field(ice, lat_name, lon_name)
                arr = np.asarray(ice.transpose(lat_name, lon_name).values, dtype=np.float32)
                arr = arr[np.newaxis]
                lat = np.asarray(ice[lat_name].values, dtype=np.float32)
                lon = np.asarray(ice[lon_name].values, dtype=np.float32)
                arr, lat = ensure_ascending_lat(arr, lat)
                arr, lon = normalize_lon(arr, lon)
                check_grid(lat, lon, lat_ref, lon_ref, "sea_ice_fraction")
                ice_means.append(arr[0])
    if ocean_mask is None:
        raise RuntimeError(f"No ocean_fraction found in forcing files under {forcing_dir}")
    if ice_means:
        ice_clim = np.nanmean(np.stack(ice_means, axis=0), axis=0)
        ocean_mask = ocean_mask & (ice_clim <= max_sea_ice)
    return ocean_mask


def grids_match(a_lat: np.ndarray, a_lon: np.ndarray, b_lat: np.ndarray, b_lon: np.ndarray) -> bool:
    return (
        a_lat.shape == b_lat.shape
        and a_lon.shape == b_lon.shape
        and np.allclose(a_lat, b_lat)
        and np.allclose(a_lon, b_lon)
    )


def check_grid(a_lat: np.ndarray, a_lon: np.ndarray, b_lat: np.ndarray, b_lon: np.ndarray, label: str) -> None:
    if not grids_match(a_lat, a_lon, b_lat, b_lon):
        raise ValueError(f"{label} grid does not match field grid")


def apply_mask_if_matching(
    cube: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray | None,
    mask_lat: np.ndarray | None,
    mask_lon: np.ndarray | None,
    label: str,
) -> np.ndarray:
    if mask is None:
        return cube
    check_grid(mask_lat, mask_lon, lat, lon, label)
    out = cube.copy()
    out[:, ~mask] = np.nan
    return out


def bbox_mean(
    arr: np.ndarray,
    lat: np.ndarray,
    lon_360: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    lat_s, lat_n, lon_w, lon_e = box
    lat_sel = (lat >= lat_s) & (lat <= lat_n)
    if lon_w <= lon_e:
        lon_sel = (lon_360 >= lon_w) & (lon_360 <= lon_e)
    else:
        lon_sel = (lon_360 >= lon_w) | (lon_360 <= lon_e)
    sub = arr[:, lat_sel, :][:, :, lon_sel]
    weights = np.cos(np.deg2rad(lat[lat_sel]))[:, np.newaxis]
    valid = np.isfinite(sub)
    num = np.nansum(sub * weights[np.newaxis] * valid, axis=(1, 2))
    den = np.nansum(weights[np.newaxis] * valid, axis=(1, 2))
    return np.where(den > 0.0, num / den, np.nan).astype(np.float32)


def detrend_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return out
    t = np.arange(y.size, dtype=np.float64)
    coef = np.polyfit(t[mask], y[mask], 1)
    out[mask] = (y[mask] - np.polyval(coef, t[mask])).astype(np.float32)
    return out


def detrend_along_year(arr: np.ndarray) -> np.ndarray:
    y = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype(np.float64)
    t = np.arange(y.shape[0], dtype=np.float64)[:, np.newaxis, np.newaxis]
    yy = np.where(valid, y, 0.0)
    sum_t = np.sum(np.where(valid, t, 0.0), axis=0)
    sum_y = np.sum(yy, axis=0)
    sum_tt = np.sum(np.where(valid, t * t, 0.0), axis=0)
    sum_ty = np.sum(np.where(valid, t * yy, 0.0), axis=0)
    denom = n * sum_tt - sum_t * sum_t
    slope = np.full(y.shape[1:], np.nan, dtype=np.float64)
    intercept = np.full(y.shape[1:], np.nan, dtype=np.float64)
    good = (n >= 2.0) & (np.abs(denom) > 0.0)
    slope[good] = (n[good] * sum_ty[good] - sum_t[good] * sum_y[good]) / denom[good]
    intercept[good] = (sum_y[good] - slope[good] * sum_t[good]) / n[good]
    trend = slope[np.newaxis] * t + intercept[np.newaxis]
    out = np.where(valid, y - trend, np.nan)
    return out.astype(np.float32)


def pearson_corr_map(field: np.ndarray, index: np.ndarray, min_years: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(field, dtype=np.float64)
    y = np.asarray(index, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)[:, np.newaxis, np.newaxis]
    n = valid.sum(axis=0)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y[:, np.newaxis, np.newaxis], 0.0)
    x_mean = np.divide(xx.sum(axis=0), n, out=np.full(x.shape[1:], np.nan), where=n > 0)
    y_mean = np.divide(yy.sum(axis=0), n, out=np.full(x.shape[1:], np.nan), where=n > 0)
    x_anom = np.where(valid, x - x_mean[np.newaxis], 0.0)
    y_anom = np.where(valid, y[:, np.newaxis, np.newaxis] - y_mean[np.newaxis], 0.0)
    num = np.sum(x_anom * y_anom, axis=0)
    den = np.sqrt(np.sum(x_anom * x_anom, axis=0) * np.sum(y_anom * y_anom, axis=0))
    corr = np.divide(num, den, out=np.full(x.shape[1:], np.nan), where=(den > 0.0) & (n >= min_years))
    corr = np.clip(corr, -1.0, 1.0)

    pval = np.full_like(corr, np.nan, dtype=np.float64)
    df = n - 2
    ok = np.isfinite(corr) & (df > 0) & (np.abs(corr) < 1.0)
    tstat = np.zeros_like(corr, dtype=np.float64)
    tstat[ok] = corr[ok] * np.sqrt(df[ok] / (1.0 - corr[ok] * corr[ok]))
    pval[ok] = 2.0 * stats.t.sf(np.abs(tstat[ok]), df[ok])
    perfect = np.isfinite(corr) & (df > 0) & (np.abs(corr) >= 1.0)
    pval[perfect] = 0.0
    return corr.astype(np.float32), pval.astype(np.float32), n.astype(np.int16)


def lon180_and_order(lon_360: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    order = np.argsort(lon_180)
    return lon_180[order], order


def add_coastlines_if_available(ax) -> None:
    """Add cached Natural Earth coastlines, but do not download or fail offline."""
    global _COAST_WARNED
    filename = "ne_110m_coastline.shp"
    roots = [
        cartopy.config.get("pre_existing_data_dir"),
        cartopy.config.get("data_dir"),
    ]
    cached = any(
        root and (Path(root) / "shapefiles" / "natural_earth" / "physical" / filename).exists()
        for root in roots
    )
    if cached:
        ax.coastlines(resolution="110m", linewidth=0.5, zorder=3)
    elif not _COAST_WARNED:
        print("  warning: cached cartopy coastlines unavailable; plotting without coastlines", flush=True)
        _COAST_WARNED = True


def render_map(
    ax,
    corr: np.ndarray,
    pval: np.ndarray,
    lat: np.ndarray,
    lon_360: np.ndarray,
    title: str,
    extent: tuple[float, float, float, float],
    stipple_p: float | None,
):
    lon_180, order = lon180_and_order(lon_360)
    corr_s = corr[:, order]
    pval_s = pval[:, order]
    lon_min, lon_max, lat_min, lat_max = extent
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_180 >= lon_min) & (lon_180 <= lon_max)
    lat_sub = lat[lat_sel]
    lon_sub = lon_180[lon_sel]
    field = corr_s[lat_sel, :][:, lon_sel]
    psub = pval_s[lat_sel, :][:, lon_sel]
    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        field,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="nearest",
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_coastlines_if_available(ax)
    if stipple_p is not None:
        sig = np.isfinite(psub) & (psub < stipple_p)
        idx = np.flatnonzero(sig)
        if idx.size > 9000:
            rng = np.random.default_rng(0)
            keep = rng.choice(idx, size=9000, replace=False)
            thin = np.zeros(sig.shape, dtype=bool)
            thin.flat[keep] = True
            sig = thin
        ax.scatter(
            lon2d[sig],
            lat2d[sig],
            s=1.2,
            c="0.15",
            alpha=0.45,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.45", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=11, weight="bold")
    return mesh


def plot_pair(
    corr_era5: np.ndarray,
    p_era5: np.ndarray,
    corr_ace2: np.ndarray,
    p_ace2: np.ndarray,
    era5_lat: np.ndarray,
    era5_lon: np.ndarray,
    ace2_lat: np.ndarray,
    ace2_lon: np.ndarray,
    out_png: Path,
    extent: tuple[float, float, float, float],
    index_source: str,
    stipple_p: float | None,
    era5_label: str,
    ace2_label: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.2),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    render_map(
        axes[0],
        corr_era5,
        p_era5,
        era5_lat,
        era5_lon,
        f"ERA5 seasonal {era5_label} x Nino3.4 ({index_source})",
        extent,
        stipple_p,
    )
    mesh = render_map(
        axes[1],
        corr_ace2,
        p_ace2,
        ace2_lat,
        ace2_lon,
        f"ACE2 seasonal {ace2_label} x Nino3.4 ({index_source})",
        extent,
        stipple_p,
    )
    cbar = fig.colorbar(mesh, ax=axes, orientation="vertical", shrink=0.82, pad=0.02)
    cbar.set_label("Pearson r")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def interp_to_grid(field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray, dst_lat: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    da = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": src_lat, "lon": src_lon})
    return da.interp(lat=dst_lat, lon=dst_lon, method="linear").values.astype(np.float32)


def extent_mask(lat: np.ndarray, lon_360: np.ndarray, extent: tuple[float, float, float, float]) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = extent
    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_180 >= lon_min) & (lon_180 <= lon_max)
    mask = np.zeros((lat.size, lon_360.size), dtype=bool)
    mask[np.ix_(lat_sel, lon_sel)] = True
    return mask


def weighted_spatial_corr(a: np.ndarray, b: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= mask
    if int(valid.sum()) < 3:
        return float("nan")
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    w = np.broadcast_to(weights, a.shape)
    w = np.where(valid, w, 0.0)
    wsum = float(w.sum())
    if wsum <= 0.0:
        return float("nan")
    a0 = np.where(valid, a, 0.0)
    b0 = np.where(valid, b, 0.0)
    am = float((a0 * w).sum() / wsum)
    bm = float((b0 * w).sum() / wsum)
    aa = np.where(valid, a - am, 0.0)
    bb = np.where(valid, b - bm, 0.0)
    cov = float((w * aa * bb).sum())
    va = float((w * aa * aa).sum())
    vb = float((w * bb * bb).sum())
    if va <= 0.0 or vb <= 0.0:
        return float("nan")
    return cov / np.sqrt(va * vb)


def difference_summary(a: np.ndarray, b: np.ndarray) -> dict:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) == 0:
        return {"n_common": 0, "max_abs": None, "mean_abs": None, "rmse": None}
    diff = a[valid] - b[valid]
    return {
        "n_common": int(valid.sum()),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def parse_psl_monthly_index(text: str) -> dict[tuple[int, int], float]:
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(tokens) < 15:
        raise ValueError("Not enough numeric tokens in PSL index file.")
    start = int(float(tokens[0]))
    end = int(float(tokens[1]))
    if start < 1800 or end < start:
        raise ValueError("First two numbers do not look like PSL start/end years.")
    out: dict[tuple[int, int], float] = {}
    idx = 2
    for expected_year in range(start, end + 1):
        if idx + 12 >= len(tokens):
            break
        year = int(float(tokens[idx]))
        idx += 1
        if year != expected_year:
            break
        for month in range(1, 13):
            val = float(tokens[idx])
            idx += 1
            if np.isfinite(val) and -90.0 < val < 900.0:
                out[(year, month)] = val
            else:
                out[(year, month)] = np.nan
    if not out:
        raise ValueError("No monthly rows parsed from PSL index.")
    return out


def noaa_nino34_index(cache_file: Path, years: list[int], season_months: tuple[int, ...], force: bool) -> np.ndarray:
    if cache_file.exists() and not force:
        text = cache_file.read_text()
    else:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(NINO34_URL, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
        cache_file.write_text(text)
    monthly = parse_psl_monthly_index(text)
    vals = []
    for year in years:
        season = np.array([monthly.get((year, month), np.nan) for month in season_months], dtype=np.float64)
        if not np.all(np.isfinite(season)):
            raise ValueError(f"NOAA Nino3.4 index is missing at least one month for {year}")
        vals.append(float(np.mean(season)))
    return np.asarray(vals, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--years", default="1980:2016", help="Example: 1980:2016 or 1980,1981,1982")
    p.add_argument("--season-months", default="6,7,8", help="Comma-separated months, default JJA.")
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--era5-sst-nc", type=Path, default=None)
    p.add_argument("--era5-yearly-pattern", default=None)
    p.add_argument("--era5-forcing-dir", type=Path, default=None)
    p.add_argument("--era5-sst-var", default="TMP2m")
    p.add_argument("--era5-zarr", default=ERA5_ZARR)
    p.add_argument("--era5-cache-dir", type=Path, default=None)
    p.add_argument("--era5-scale", type=float, default=1.0)

    p.add_argument("--ace2-runs-root", type=Path, default=None)
    p.add_argument("--ace2-sst-nc", type=Path, default=None)
    p.add_argument("--ace2-yearly-pattern", default=None)
    p.add_argument("--ace2-sst-var", default="TMP2m")
    p.add_argument("--ace2-scale", type=float, default=1.0)
    p.add_argument("--n-members", type=int, default=25)
    p.add_argument("--init-month", type=int, default=5)
    p.add_argument("--init-day", type=int, default=1)
    p.add_argument("--min-members", type=int, default=1)
    p.add_argument("--no-allow-partial", action="store_true")

    p.add_argument("--mask-source-forcing-dir", type=Path, default=None)
    p.add_argument("--no-ocean-mask", action="store_true")
    p.add_argument("--ocean-min", type=float, default=0.5)
    p.add_argument("--max-sea-ice", type=float, default=0.15)

    p.add_argument(
        "--index-source",
        choices=("self", "era5", "noaa"),
        default="self",
        help=(
            "self: ERA5 map uses ERA5 Nino3.4 and ACE2 map uses ACE2 Nino3.4; "
            "era5: both maps use ERA5-derived Nino3.4; noaa: both use NOAA/PSL index."
        ),
    )
    p.add_argument("--nino34-box", nargs=4, type=float, default=[-5.0, 5.0, 190.0, 240.0])
    p.add_argument("--noaa-index-cache", type=Path, default=None)
    p.add_argument("--force-download-index", action="store_true")
    p.add_argument("--no-detrend", action="store_true")
    p.add_argument("--min-years", type=int, default=10)

    p.add_argument("--extent", nargs=4, type=float, default=[-180.0, 180.0, -60.0, 75.0])
    p.add_argument("--stipple-p", type=float, default=0.05, help="Set negative to disable.")
    return p


def load_era5(
    args,
    years: list[int],
    season_months: tuple[int, ...],
    template_lat: np.ndarray | None = None,
    template_lon: np.ndarray | None = None,
):
    if args.era5_sst_nc is not None:
        print(f"Loading ERA5 field from {args.era5_sst_nc}", flush=True)
        return yearly_cube_from_nc(args.era5_sst_nc, args.era5_sst_var, years, season_months, args.era5_scale)
    if args.era5_yearly_pattern is not None:
        print(f"Loading ERA5 field from pattern {args.era5_yearly_pattern}", flush=True)
        return yearly_cube_from_pattern(args.era5_yearly_pattern, args.era5_sst_var, years, season_months, args.era5_scale)
    if is_tmp2m_var(args.era5_sst_var):
        if template_lat is None or template_lon is None:
            raise ValueError("ERA5 TMP2m ARCO loading needs a template grid.")
        cache_dir = args.era5_cache_dir or args.output_dir / "cache/era5_tmp2m"
        print(f"Loading/building ERA5 TMP2m from ARCO cache in {cache_dir}", flush=True)
        return yearly_cube_era5_tmp2m_from_arco(
            cache_dir,
            args.era5_zarr,
            years,
            season_months,
            template_lat,
            template_lon,
            args.era5_scale,
        )
    forcing_dir = args.era5_forcing_dir or args.root / "data/lag_data/forcing_data_ace2era5"
    print(f"Loading ERA5/prescribed field from forcing files in {forcing_dir}", flush=True)
    return yearly_cube_from_forcing_dir(forcing_dir, args.era5_sst_var, years, season_months, args.era5_scale)


def load_ace2(args, years: list[int], season_months: tuple[int, ...]):
    if args.ace2_sst_nc is not None:
        print(f"Loading ACE2 field from {args.ace2_sst_nc}", flush=True)
        arr, lat, lon, year_out, var = yearly_cube_from_nc(
            args.ace2_sst_nc, args.ace2_sst_var, years, season_months, args.ace2_scale
        )
        return arr, lat, lon, year_out, var, None
    if args.ace2_yearly_pattern is not None:
        print(f"Loading ACE2 field from pattern {args.ace2_yearly_pattern}", flush=True)
        arr, lat, lon, year_out, var = yearly_cube_from_pattern(
            args.ace2_yearly_pattern, args.ace2_sst_var, years, season_months, args.ace2_scale
        )
        return arr, lat, lon, year_out, var, None
    runs_root = args.ace2_runs_root or args.root / "outputs/lag_may/runs"
    print(f"Loading ACE2 field from member runs in {runs_root}", flush=True)
    return yearly_cube_from_ace2_runs(
        runs_root,
        args.ace2_sst_var,
        years,
        season_months,
        args.ace2_scale,
        args.n_members,
        args.init_month,
        args.init_day,
        args.min_members,
        allow_partial=not args.no_allow_partial,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.root = args.root.resolve()
    years = parse_years(args.years)
    season_months = parse_int_list(args.season_months)
    nino_box = parse_box(args.nino34_box)
    extent = tuple(float(x) for x in args.extent)
    out_dir = args.output_dir or args.root / "outputs/lag_may/enso_tmp2m_correlation"
    args.output_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ace2_sst, ace2_lat, ace2_lon, ace2_years, ace2_var, member_counts = load_ace2(args, years, season_months)
    era5_sst, era5_lat, era5_lon, era5_years, era5_var = load_era5(
        args, years, season_months, template_lat=ace2_lat, template_lon=ace2_lon
    )
    if era5_years != years or ace2_years != years:
        raise ValueError(f"Loaded years differ from requested years: ERA5={era5_years}, ACE2={ace2_years}")

    if is_tmp2m_var(era5_var) and is_tmp2m_var(ace2_var) and not args.no_ocean_mask:
        print("TMP2m selected for both ERA5 and ACE2; skipping ocean/sea-ice mask.", flush=True)
        args.no_ocean_mask = True

    mask_source = args.mask_source_forcing_dir or args.era5_forcing_dir or args.root / "data/lag_data/forcing_data_ace2era5"
    mask_used = False
    if not args.no_ocean_mask and mask_source.exists():
        print(f"Applying ocean/sea-ice mask from {mask_source}", flush=True)
        era5_mask = None
        try:
            era5_mask = load_ocean_mask_from_forcing(
                mask_source, years, season_months, era5_lat, era5_lon, args.ocean_min, args.max_sea_ice
            )
            era5_sst = apply_mask_if_matching(
                era5_sst, era5_lat, era5_lon, era5_mask, era5_lat, era5_lon, "ERA5 mask"
            )
            mask_used = True
        except Exception as exc:
            print(f"  warning: could not apply forcing mask to ERA5 grid ({exc})", flush=True)

        try:
            if era5_mask is not None and grids_match(era5_lat, era5_lon, ace2_lat, ace2_lon):
                ace2_mask = era5_mask
            else:
                ace2_mask = load_ocean_mask_from_forcing(
                    mask_source, years, season_months, ace2_lat, ace2_lon, args.ocean_min, args.max_sea_ice
                )
            ace2_sst = apply_mask_if_matching(
                ace2_sst, ace2_lat, ace2_lon, ace2_mask, ace2_lat, ace2_lon, "ACE2 mask"
            )
            mask_used = True
        except Exception as exc:
            print(f"  warning: could not apply forcing mask to ACE2 grid ({exc})", flush=True)
    elif not args.no_ocean_mask:
        print(f"Ocean mask source not found ({mask_source}); proceeding without external mask.", flush=True)

    field_difference = None
    if grids_match(era5_lat, era5_lon, ace2_lat, ace2_lon):
        field_difference = difference_summary(era5_sst, ace2_sst)
        if field_difference["max_abs"] is not None and field_difference["max_abs"] < 1.0e-6:
            print("WARNING: ERA5 and ACE2 field cubes are identical after seasonal averaging/masking.", flush=True)
            if ace2_var.lower() == "surface_temperature":
                print(f"         {PRESCRIBED_SST_NOTE}", flush=True)
    else:
        print("ERA5 and ACE2 grids differ; skipping direct field equality diagnostic.", flush=True)

    era5_nino = bbox_mean(era5_sst, era5_lat, era5_lon, nino_box)
    ace2_nino = bbox_mean(ace2_sst, ace2_lat, ace2_lon, nino_box)
    if args.index_source == "self":
        era5_index = era5_nino
        ace2_index = ace2_nino
    elif args.index_source == "era5":
        era5_index = era5_nino
        ace2_index = era5_nino
    else:
        cache = args.noaa_index_cache or out_dir / "indices/nina34.anom.data"
        noaa = noaa_nino34_index(cache, years, season_months, args.force_download_index)
        era5_index = noaa
        ace2_index = noaa

    if args.no_detrend:
        era5_field_for_corr = era5_sst
        ace2_field_for_corr = ace2_sst
        era5_index_for_corr = era5_index.astype(np.float32)
        ace2_index_for_corr = ace2_index.astype(np.float32)
        detrend_method = "none"
    else:
        print("Detrending seasonal fields and Nino3.4 indices", flush=True)
        era5_field_for_corr = detrend_along_year(era5_sst)
        ace2_field_for_corr = detrend_along_year(ace2_sst)
        era5_index_for_corr = detrend_1d(era5_index)
        ace2_index_for_corr = detrend_1d(ace2_index)
        detrend_method = "linear"

    print("Computing ERA5 correlation map", flush=True)
    corr_era5, p_era5, n_era5 = pearson_corr_map(era5_field_for_corr, era5_index_for_corr, args.min_years)
    print("Computing ACE2 correlation map", flush=True)
    corr_ace2, p_ace2, n_ace2 = pearson_corr_map(ace2_field_for_corr, ace2_index_for_corr, args.min_years)

    era5_on_ace2 = interp_to_grid(corr_era5, era5_lat, era5_lon, ace2_lat, ace2_lon)
    full_pattern_r = weighted_spatial_corr(era5_on_ace2, corr_ace2, ace2_lat)
    plot_mask = extent_mask(ace2_lat, ace2_lon, extent)
    extent_pattern_r = weighted_spatial_corr(era5_on_ace2, corr_ace2, ace2_lat, plot_mask)

    ds_out = xr.Dataset(
        {
            "corr_era5": (("era5_lat", "era5_lon"), corr_era5),
            "pval_era5": (("era5_lat", "era5_lon"), p_era5),
            "n_era5": (("era5_lat", "era5_lon"), n_era5),
            "corr_ace2": (("ace2_lat", "ace2_lon"), corr_ace2),
            "pval_ace2": (("ace2_lat", "ace2_lon"), p_ace2),
            "n_ace2": (("ace2_lat", "ace2_lon"), n_ace2),
            "nino34_era5_field": (("year",), era5_nino),
            "nino34_ace2_field": (("year",), ace2_nino),
            "index_used_for_era5": (("year",), era5_index.astype(np.float32)),
            "index_used_for_ace2": (("year",), ace2_index.astype(np.float32)),
        },
        coords={
            "year": years,
            "era5_lat": era5_lat,
            "era5_lon": era5_lon,
            "ace2_lat": ace2_lat,
            "ace2_lon": ace2_lon,
        },
        attrs={
            "description": "Seasonal field correlation with seasonal Nino3.4 index",
            "era5_var": era5_var,
            "ace2_var": ace2_var,
            "season_months": ",".join(str(m) for m in season_months),
            "nino34_box_0_360": json.dumps(list(nino_box)),
            "index_source": args.index_source,
            "detrend": detrend_method,
            "ocean_mask_applied": str(mask_used),
            "source_note": PRESCRIBED_SST_NOTE if ace2_var.lower() == "surface_temperature" else "",
        },
    )
    nc_path = out_dir / "enso_field_correlation_maps.nc"
    ds_out.to_netcdf(nc_path)

    stipple_p = None if args.stipple_p < 0 else float(args.stipple_p)
    fig_path = out_dir / "figures/enso_field_corr_era5_vs_ace2.png"
    plot_pair(
        corr_era5,
        p_era5,
        corr_ace2,
        p_ace2,
        era5_lat,
        era5_lon,
        ace2_lat,
        ace2_lon,
        fig_path,
        extent,
        args.index_source,
        stipple_p,
        era5_var,
        ace2_var,
    )

    summary = {
        "years": years,
        "season_months": list(season_months),
        "era5_source": (
            str(args.era5_sst_nc)
            if args.era5_sst_nc
            else (
                args.era5_yearly_pattern
                if args.era5_yearly_pattern
                else (
                    f"{args.era5_zarr} (cached in {args.era5_cache_dir or args.output_dir / 'cache/era5_tmp2m'})"
                    if is_tmp2m_var(args.era5_sst_var)
                    else str(args.era5_forcing_dir or args.root / "data/lag_data/forcing_data_ace2era5")
                )
            )
        ),
        "era5_var": era5_var,
        "ace2_source": str(args.ace2_sst_nc or args.ace2_yearly_pattern or (args.ace2_runs_root or args.root / "outputs/lag_may/runs")),
        "ace2_var": ace2_var,
        "member_counts": member_counts,
        "index_source": args.index_source,
        "nino34_box_0_360": list(nino_box),
        "detrend": detrend_method,
        "ocean_mask_applied": mask_used,
        "source_note": PRESCRIBED_SST_NOTE if ace2_var.lower() == "surface_temperature" else None,
        "field_difference_summary_on_common_grid": field_difference,
        "weighted_pattern_corr_full_domain_on_ace2_grid": full_pattern_r,
        "weighted_pattern_corr_plot_extent_on_ace2_grid": extent_pattern_r,
        "outputs": {
            "netcdf": str(nc_path),
            "figure": str(fig_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {nc_path}", flush=True)
    print(f"wrote {fig_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"weighted pattern r over plotted extent: {extent_pattern_r:.3f}", flush=True)


if __name__ == "__main__":
    main()
