from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import xarray as xr

TEMP_CANDIDATES = ("TMP2m", "2m_temperature", "t2m", "2t", "temperature_2m")
LAT_NAMES = ("lat", "latitude", "grid_yt", "y")
LON_NAMES = ("lon", "longitude", "grid_xt", "x")


def coord_name(obj: xr.Dataset | xr.DataArray, candidates: Iterable[str]) -> str | None:
    names = set(obj.coords) | set(obj.dims)
    return next((x for x in candidates if x in names), None)


def pick_var(ds: xr.Dataset, candidates: Iterable[str], label: str) -> str:
    lower = {k.lower(): k for k in ds.data_vars}
    for name in candidates:
        if name in ds.data_vars:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Cannot find {label}; vars={list(ds.data_vars)}")


def assign_valid_time(obj: xr.Dataset | xr.DataArray, init_time: pd.Timestamp | str) -> xr.Dataset | xr.DataArray:
    if "time" not in obj.coords and "time" not in obj.dims:
        return obj

    vals = np.asarray(obj["time"].values)
    if vals.size == 0 or np.issubdtype(vals.dtype, np.datetime64):
        return obj

    base = pd.Timestamp(init_time)
    units = str(obj["time"].attrs.get("units", "")).lower()

    if np.issubdtype(vals.dtype, np.timedelta64):
        valid = base + pd.to_timedelta(vals)
    elif "microsecond" in units or units in {"us", "µs"}:
        valid = base + pd.to_timedelta(vals.astype("int64"), unit="us")
    elif "hour" in units:
        valid = base + pd.to_timedelta(vals.astype(float), unit="h")
    elif "day" in units:
        valid = base + pd.to_timedelta(vals.astype(float), unit="D")
    else:
        valid = base + pd.to_timedelta(6.0 + np.arange(vals.size) * 6.0, unit="h")

    return obj.assign_coords(time=valid)


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    probe = da
    for dim in ("sample", "time"):
        if dim in probe.dims and probe.sizes.get(dim, 0) > 0:
            probe = probe.isel({dim: 0})
    if float(probe.mean(skipna=True).compute()) > 100:
        da = da - 273.15
    da.attrs["units"] = "degC"
    return da


def subset_bbox(obj: xr.Dataset | xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.Dataset | xr.DataArray:
    south, north, west, east = bbox
    lat = coord_name(obj, LAT_NAMES)
    lon = coord_name(obj, LON_NAMES)
    if lat is None or lon is None:
        raise KeyError(f"Missing lat/lon. dims={obj.dims}, coords={list(obj.coords)}")

    lat_vals = obj[lat]
    lat_slice = slice(north, south) if float(lat_vals[0]) > float(lat_vals[-1]) else slice(south, north)

    lon_vals = obj[lon]
    if float(lon_vals.max()) <= 180 and west > 180:
        west, east = west - 360, east - 360
    lon_slice = slice(west, east)

    return obj.sel({lat: lat_slice, lon: lon_slice})


def write_atomic(ds: xr.Dataset | xr.DataArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    ds.to_netcdf(tmp)
    tmp.replace(path)


def complete_netcdf(path: Path, required_vars: Iterable[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return all(v in ds.data_vars for v in required_vars)
    except Exception:
        return False


def regrid_nearest_to_template(da: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    tlat, tlon = coord_name(template, LAT_NAMES), coord_name(template, LON_NAMES)
    dlat, dlon = coord_name(da, LAT_NAMES), coord_name(da, LON_NAMES)
    if None in (tlat, tlon, dlat, dlon):
        raise KeyError("Cannot identify lat/lon for regrid")

    out = da.sel({dlat: template[tlat].values, dlon: template[tlon].values}, method="nearest")

    for old, new in ((dlat, tlat), (dlon, tlon)):
        if old != new and new in out.coords and new not in out.dims:
            out = out.drop_vars(new, errors="ignore")

    rename = {}
    if dlat != tlat and dlat in out.dims:
        rename[dlat] = tlat
    if dlon != tlon and dlon in out.dims:
        rename[dlon] = tlon
    if rename:
        out = out.rename(rename)

    return out.assign_coords({tlat: template[tlat].values, tlon: template[tlon].values})
