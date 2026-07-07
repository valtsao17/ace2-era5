#!/usr/bin/env python3
"""Screen available precursors for the RAW JJA heat-extreme skill box.

This is meant for the cluster, where the actual May-initialized inference
products, initial-condition files, and ACE2-ERA5 forcing files live. It does not
need a new inference run and it does not assume saved VPD/RH diagnostics.

Target:
  - The same US/Canada land box selected by maximum ACE2-vs-ERA5 Kendall tau in
    sst_raw_rankcorr_box_pipeline.py.
  - Scalar RAW JJA heat-extreme frequency index over that box, separately for
    ERA5 and ACE2.

Predictors screened:
  - May initial-condition fields from data/lag_data/initial_conditions.
  - Derived May IC fields when possible: 2m RH and 10m wind speed.
  - Prescribed surface_temperature windows from data/lag_data/forcing_data_ace2era5,
    split into land surface temperature and ocean SST masks.

For every predictor, this script computes detrended Pearson-r maps against the
RAW index and ranks the strongest coherent 8x12 deg land and 10x15 deg ocean
boxes. The result is a source-screen, not a causal attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from seasonal_jja_skill import YEARS  # noqa: E402
from sst_teleconnection_jja_sliding7d import _draw_coast_and_borders, _thin_mask  # noqa: E402


SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
FREQ_NC = SLIDING_DIR / "jja_seasonal_freqs.nc"
SKILL_NC = SLIDING_DIR / "skill_jja_seasonal.nc"
IC_DIR = PROJECT_ROOT / "data/lag_data/initial_conditions"
FORCING_DIR = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"
OUT_DIR = PROJECT_ROOT / "outputs/lag_may/raw_source_screen"

CONUS_LAT_SLICE = slice(105, 163)
CONUS_LON_SLICE = slice(200, 305)
RAW_BOX_LAT_SPANS = (8.0, 10.0, 12.0, 15.0)
RAW_BOX_LON_SPANS = (10.0, 12.0, 15.0, 18.0)
SIGNIFICANCE_ALPHA = 0.05
MIN_LAND_CELLS = 12
MIN_SIGNIFICANT_LAND_CELLS = 8
MIN_SIGNIFICANT_LAND_FRACTION = 0.70
MIN_ALLOWED_LAND_FRACTION = 0.70
RAW_BOX_EXTEND_SOUTH_CELLS = 5
RAW_BOX_EXTEND_EAST_CELLS = 2
ALLOWED_RAW_BOX_COUNTRIES = ("United States of America", "Canada")

SURFACE_WINDOWS = {
    "lag0_JJA": ("06-01", "08-31"),
    "lag30_May": ("05-01", "05-31"),
    "lag60_AprMay": ("04-01", "05-31"),
    "lag90_MarMay": ("03-01", "05-31"),
}

BASE_IC_VARS = (
    "TMP2m",
    "Q2m",
    "PRESsfc",
    "UGRD10m",
    "VGRD10m",
)
VERTICAL_PREFIXES = (
    "air_temperature_",
    "specific_total_water_",
    "eastward_wind_",
    "northward_wind_",
)


def _box_json(box):
    la0, la1, lo0, lo1 = box
    return {
        "latN": [round(float(la0), 2), round(float(la1), 2)],
        "lonW": [round(float(360.0 - lo1), 2), round(float(360.0 - lo0), 2)],
    }


def expand_box_on_grid(
    box: tuple[float, float, float, float],
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    south_cells: int = RAW_BOX_EXTEND_SOUTH_CELLS,
    east_cells: int = RAW_BOX_EXTEND_EAST_CELLS,
) -> tuple[float, float, float, float]:
    la0, la1, lo0, lo1 = box
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    i0 = int(np.argmin(np.abs(lat - la0)))
    j1 = int(np.argmin(np.abs(lon - lo1)))
    i0_new = max(0, i0 - int(south_cells))
    j1_new = min(len(lon) - 1, j1 + int(east_cells))
    return float(lat[i0_new]), float(la1), float(lo0), float(lon[j1_new])


def _country_mask(lat, lon_360, country_names):
    """Boolean mask for grid-cell centers inside Natural Earth countries."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point

    wanted = {name.lower() for name in country_names}
    shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries",
    )
    geoms = []
    for rec in shpreader.Reader(shp).records():
        attrs = rec.attributes
        names = {
            str(attrs.get("ADMIN", "")).lower(),
            str(attrs.get("NAME", "")).lower(),
            str(attrs.get("NAME_LONG", "")).lower(),
            str(attrs.get("SOVEREIGNT", "")).lower(),
        }
        if names & wanted:
            geoms.append(rec.geometry)
    if not geoms:
        raise RuntimeError(f"No Natural Earth geometry found for {country_names}")

    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    out = np.zeros((len(lat), len(lon_360)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon_180):
            pt = Point(float(lo), float(la))
            out[i, j] = any(g.contains(pt) or g.touches(pt) for g in geoms)
    return out


def find_rankcorr_box(tau, pval, lat, lon, land, allowed_land):
    """Select a compact RAW index box with coherent significant skill."""
    best = None
    for require_fraction in (True, False):
        for lat_span in RAW_BOX_LAT_SPANS:
            for lon_span in RAW_BOX_LON_SPANS:
                for la0 in lat:
                    la1 = float(la0) + lat_span
                    if la1 > float(lat[-1]):
                        continue
                    sla = (lat >= la0) & (lat <= la1)
                    for lo0 in lon:
                        lo1 = float(lo0) + lon_span
                        if lo1 > float(lon[-1]):
                            continue
                        slo = (lon >= lo0) & (lon <= lo1)
                        sub_tau = tau[np.ix_(sla, slo)]
                        sub_pval = pval[np.ix_(sla, slo)]
                        sub_land = land[np.ix_(sla, slo)]
                        sub_allowed_land = allowed_land[np.ix_(sla, slo)]
                        if np.any(sub_land & ~sub_allowed_land):
                            continue
                        allowed_land_fraction = float(sub_allowed_land.mean())
                        if allowed_land_fraction < MIN_ALLOWED_LAND_FRACTION:
                            continue
                        valid = np.isfinite(sub_tau) & sub_allowed_land
                        n_land = int(valid.sum())
                        if n_land < MIN_LAND_CELLS:
                            continue
                        sig = valid & np.isfinite(sub_pval) & (sub_pval < SIGNIFICANCE_ALPHA)
                        n_sig = int(sig.sum())
                        if n_sig < MIN_SIGNIFICANT_LAND_CELLS:
                            continue
                        sig_fraction = float(n_sig / n_land)
                        if require_fraction and sig_fraction < MIN_SIGNIFICANT_LAND_FRACTION:
                            continue
                        mean_tau = float(np.nanmean(sub_tau[valid]))
                        mean_tau_sig = float(np.nanmean(sub_tau[sig]))
                        score = mean_tau_sig * sig_fraction
                        if best is None or score > best["score"]:
                            best = {
                                "bbox": (float(la0), float(la1), float(lo0), float(lo1)),
                                "score": score,
                                "mean_tau": mean_tau,
                                "mean_tau_significant": mean_tau_sig,
                                "n_land_cells": n_land,
                                "n_significant_land_cells": n_sig,
                                "significant_land_fraction": sig_fraction,
                                "allowed_land_fraction": allowed_land_fraction,
                                "lat_span": float(lat_span),
                                "lon_span": float(lon_span),
                            }
        if best is not None:
            return best
    if best is None:
        raise RuntimeError("No valid land rank-correlation box found.")
    return best


@dataclass
class Predictor:
    name: str
    group: str
    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    land_mask: np.ndarray | None
    ocean_mask: np.ndarray | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq-nc", type=Path, default=FREQ_NC)
    p.add_argument("--skill-nc", type=Path, default=SKILL_NC)
    p.add_argument("--ic-dir", type=Path, default=IC_DIR)
    p.add_argument("--forcing-dir", type=Path, default=FORCING_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--years", default="all", help="'all' or comma-separated years")
    p.add_argument(
        "--target",
        choices=("era5", "ace2", "both"),
        default="both",
        help="Which RAW index to screen against.",
    )
    p.add_argument(
        "--ic-aggregation",
        choices=("center", "mean", "both"),
        default="mean",
        help="Use May 1 center member, mean over all 25 lagged IC times, or both.",
    )
    p.add_argument(
        "--ic-vars",
        default="auto",
        help="'auto' or comma-separated IC variables. Auto includes surface vars and vertical state vars.",
    )
    p.add_argument("--skip-ic", action="store_true")
    p.add_argument("--skip-forcing", action="store_true")
    p.add_argument("--top-n", type=int, default=6, help="Number of top ERA5 maps to plot.")
    p.add_argument("--no-save-maps", action="store_true", help="Skip writing correlation maps NetCDF.")
    p.add_argument("--search-lat", nargs=2, type=float, default=(15.0, 75.0))
    p.add_argument("--search-lon", nargs=2, type=float, default=(190.0, 330.0))
    return p.parse_args()


def parse_years(spec: str) -> list[int]:
    return YEARS if spec == "all" else [int(y) for y in spec.split(",")]


def safe_name(name: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    return out[:120]


def coord_name(ds: xr.Dataset | xr.DataArray, options: tuple[str, ...]) -> str:
    for name in options:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find any coordinate from {options}")


def lon_to_360(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float32)
    return np.where(lon < 0.0, lon + 360.0, lon)


def roll_to_180(field: np.ndarray, lon_360: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_360 = np.asarray(lon_360)
    split = int(np.searchsorted(lon_360, 180.0))
    lon_r = np.concatenate([lon_360[split:] - 360.0, lon_360[:split]])
    return np.roll(field, len(lon_360) - split, axis=-1), lon_r


def detrend_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64)
    valid = np.isfinite(x)
    out = np.full_like(x, np.nan)
    if valid.sum() < 3:
        return out.astype(np.float32)
    slope, intercept, *_ = stats.linregress(t[valid], x[valid])
    out[valid] = x[valid] - (slope * t[valid] + intercept)
    return out.astype(np.float32)


def detrend_axis0(arr: np.ndarray) -> np.ndarray:
    """Vectorized linear detrend along year axis with NaN support."""
    arr = np.asarray(arr, dtype=np.float64)
    t = np.arange(arr.shape[0], dtype=np.float64)[:, None, None]
    valid = np.isfinite(arr)
    n = valid.sum(axis=0).astype(np.float64)

    x = np.where(valid, arr, 0.0)
    tv = np.where(valid, t, 0.0)
    sum_t = tv.sum(axis=0)
    sum_t2 = (tv * tv).sum(axis=0)
    sum_x = x.sum(axis=0)
    sum_tx = (tv * x).sum(axis=0)
    denom = n * sum_t2 - sum_t * sum_t

    slope = np.full(arr.shape[1:], np.nan, dtype=np.float64)
    ok = (n >= 3) & (denom != 0.0)
    slope[ok] = (n[ok] * sum_tx[ok] - sum_t[ok] * sum_x[ok]) / denom[ok]
    intercept = np.full_like(slope, np.nan)
    intercept[ok] = (sum_x[ok] - slope[ok] * sum_t[ok]) / n[ok]

    trend = slope[None, :, :] * t + intercept[None, :, :]
    out = arr - trend
    out[~valid] = np.nan
    return out.astype(np.float32)


def corr_p_map(field: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field_dt = detrend_axis0(field)
    target_dt = detrend_1d(target).astype(np.float64)
    y = target_dt[:, None, None]

    valid = np.isfinite(field_dt) & np.isfinite(y)
    n = valid.sum(axis=0).astype(np.float64)
    x = np.where(valid, field_dt, np.nan)
    yy = np.where(valid, y, np.nan)

    xm = x - np.nanmean(x, axis=0, keepdims=True)
    ym = yy - np.nanmean(yy, axis=0, keepdims=True)
    xm = np.where(valid, xm, 0.0)
    ym = np.where(valid, ym, 0.0)

    num = (xm * ym).sum(axis=0)
    den = np.sqrt((xm * xm).sum(axis=0) * (ym * ym).sum(axis=0))
    r = np.full(field.shape[1:], np.nan, dtype=np.float32)
    ok = (n >= 10) & (den > 0.0)
    r[ok] = (num[ok] / den[ok]).astype(np.float32)

    p = np.full_like(r, np.nan, dtype=np.float32)
    rr = np.clip(r[ok].astype(np.float64), -0.999999, 0.999999)
    tstat = rr * np.sqrt((n[ok] - 2.0) / np.maximum(1.0 - rr * rr, 1e-12))
    p[ok] = (2.0 * stats.t.sf(np.abs(tstat), df=n[ok] - 2.0)).astype(np.float32)
    return r, p


def weighted_box_mean(
    arr: np.ndarray,
    lat: np.ndarray,
    lon_360: np.ndarray,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    lat_s, lat_n, lon_w, lon_e = bbox
    lat_sel = (lat >= lat_s) & (lat <= lat_n)
    lon_sel = (lon_360 >= lon_w) & (lon_360 <= lon_e)
    sub = arr[:, lat_sel, :][:, :, lon_sel]
    if mask is not None:
        sub_mask = mask[lat_sel, :][:, lon_sel]
        sub = np.where(sub_mask[None, :, :], sub, np.nan)
    weights = np.cos(np.deg2rad(lat[lat_sel]))[:, None]
    valid = np.isfinite(sub)
    num = np.nansum(sub * weights[None, :, :] * valid, axis=(1, 2))
    den = np.nansum(weights[None, :, :] * valid, axis=(1, 2))
    return np.where(den > 0.0, num / den, np.nan).astype(np.float32)


def load_land_mask_from_forcing(forcing_dir: Path, lat: np.ndarray, lon_360: np.ndarray) -> np.ndarray | None:
    f = next(forcing_dir.glob("forcing_*.nc"), None)
    if f is None:
        return None
    with xr.open_dataset(f) as ds:
        if "land_fraction" not in ds:
            return None
        lf = ds["land_fraction"]
        lf = lf.isel(time=0) if "time" in lf.dims else lf
        lat_n = coord_name(lf, ("lat", "latitude"))
        lon_n = coord_name(lf, ("lon", "longitude"))
        lf2 = lf.interp({lat_n: lat, lon_n: lon_360}, method="nearest")
        return lf2.values > 0.5


def load_targets(args: argparse.Namespace, years: list[int]):
    with xr.open_dataset(args.freq_nc) as ds:
        lat = ds["lat"].values.astype(np.float32)
        lon = lon_to_360(ds["lon"].values.astype(np.float32))
        all_years = ds["year"].values.astype(int).tolist()
        year_idx = [all_years.index(y) for y in years]
        era5 = ds["era5_freq"].isel(year=year_idx).values.astype(np.float32)
        ace2 = ds["ace2_freq"].isel(year=year_idx).values.astype(np.float32)

    with xr.open_dataset(args.skill_nc) as ds:
        tau = ds["kendall_tau"].values.astype(np.float32)
        tau_p = ds["tau_p_value"].values.astype(np.float32)

    land = load_land_mask_from_forcing(args.forcing_dir, lat, lon)
    if land is None:
        raise RuntimeError("Could not build land mask from forcing files.")
    allowed_land = _country_mask(lat, lon, ALLOWED_RAW_BOX_COUNTRIES) & land

    lat_c = lat[CONUS_LAT_SLICE]
    lon_c = lon[CONUS_LON_SLICE]
    raw_box = find_rankcorr_box(
        tau[CONUS_LAT_SLICE, CONUS_LON_SLICE],
        tau_p[CONUS_LAT_SLICE, CONUS_LON_SLICE],
        lat_c,
        lon_c,
        land[CONUS_LAT_SLICE, CONUS_LON_SLICE],
        allowed_land[CONUS_LAT_SLICE, CONUS_LON_SLICE],
    )

    selected_bbox = raw_box["bbox"]
    bbox = expand_box_on_grid(selected_bbox, lat_c, lon_c)
    era5_idx = weighted_box_mean(era5, lat, lon, bbox, allowed_land)
    ace2_idx = weighted_box_mean(ace2, lat, lon, bbox, allowed_land)

    targets = {}
    if args.target in ("era5", "both"):
        targets["era5"] = era5_idx
    if args.target in ("ace2", "both"):
        targets["ace2"] = ace2_idx

    box_info = {
        "raw_rankcorr_box": {
            **_box_json(bbox),
            "selected_box_before_expansion": _box_json(selected_bbox),
            "expansion_grid_cells": {
                "south": RAW_BOX_EXTEND_SOUTH_CELLS,
                "east": RAW_BOX_EXTEND_EAST_CELLS,
            },
            "selection_score": float(raw_box["score"]),
            "land_mean_tau": float(raw_box["mean_tau"]),
            "significant_land_mean_tau": float(raw_box["mean_tau_significant"]),
            "n_land_cells": int(raw_box["n_land_cells"]),
            "n_significant_land_cells": int(raw_box["n_significant_land_cells"]),
            "significant_land_fraction": float(raw_box["significant_land_fraction"]),
            "allowed_land_fraction": float(raw_box["allowed_land_fraction"]),
            "lat_span": float(raw_box["lat_span"]),
            "lon_span": float(raw_box["lon_span"]),
            "allowed_land_countries": list(ALLOWED_RAW_BOX_COUNTRIES),
        },
        "years": years,
        "era5_index_mean_pct": float(np.nanmean(era5_idx) * 100.0),
        "ace2_index_mean_pct": float(np.nanmean(ace2_idx) * 100.0),
    }
    return targets, box_info, lat, lon, land, allowed_land


def extract_time_field(da: xr.DataArray, aggregation: str) -> np.ndarray:
    if "time" not in da.dims:
        return da.values.astype(np.float32)
    if aggregation == "center":
        idx = min(12, da.sizes["time"] // 2)
        return da.isel(time=idx).values.astype(np.float32)
    if aggregation == "mean":
        return da.mean("time").values.astype(np.float32)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def transform_ic_var(name: str, arr: np.ndarray) -> tuple[str, np.ndarray]:
    if name == "TMP2m" or name.startswith("air_temperature_"):
        return f"{name}_C", arr - 273.15
    if name == "Q2m" or name.startswith("specific_total_water_"):
        return f"{name}_gkg", arr * 1000.0
    if name == "PRESsfc":
        return "PRESsfc_hPa", arr / 100.0
    return name, arr


def rh_from_q(t_k: np.ndarray, q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    exponent = 17.67 * (t_k - 273.16) / (t_k - 29.65)
    return np.clip(0.263 * p_pa * q / np.exp(exponent), 0.0, 100.0).astype(np.float32)


def available_ic_vars(ic_dir: Path, years: list[int], requested: str) -> list[str]:
    first = ic_dir / f"ic_lag_{years[0]}0501_25m.nc"
    if not first.exists():
        raise FileNotFoundError(f"Missing IC file: {first}")
    with xr.open_dataset(first) as ds:
        if requested != "auto":
            names = [v.strip() for v in requested.split(",") if v.strip()]
            return [v for v in names if v in ds.data_vars]
        names = [v for v in BASE_IC_VARS if v in ds.data_vars]
        for v in ds.data_vars:
            if any(v.startswith(prefix) for prefix in VERTICAL_PREFIXES):
                names.append(v)
    return sorted(set(names), key=names.index)


def ic_coords_and_masks(
    ic_dir: Path,
    forcing_dir: Path,
    years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with xr.open_dataset(ic_dir / f"ic_lag_{years[0]}0501_25m.nc") as ds:
        lat_n = coord_name(ds, ("lat", "latitude"))
        lon_n = coord_name(ds, ("lon", "longitude"))
        lat = ds[lat_n].values.astype(np.float32)
        lon = lon_to_360(ds[lon_n].values.astype(np.float32))
    land = load_land_mask_from_forcing(forcing_dir, lat, lon)
    ocean = None if land is None else ~land
    return lat, lon, land, ocean


def iter_ic_predictors(args: argparse.Namespace, years: list[int]):
    aggs = ("center", "mean") if args.ic_aggregation == "both" else (args.ic_aggregation,)
    names = available_ic_vars(args.ic_dir, years, args.ic_vars)
    lat, lon, land, ocean = ic_coords_and_masks(args.ic_dir, args.forcing_dir, years)

    for agg in aggs:
        for var in names:
            arrs = []
            for year in years:
                with xr.open_dataset(args.ic_dir / f"ic_lag_{year}0501_25m.nc") as ds:
                    if var not in ds:
                        raise KeyError(f"{var} missing in {year} IC file")
                    arrs.append(extract_time_field(ds[var], agg))
            label, data = transform_ic_var(var, np.stack(arrs, axis=0))
            yield Predictor(f"ic_{agg}_{label}", "initial_condition", data.astype(np.float32), lat, lon, land, ocean)

        if {"TMP2m", "Q2m", "PRESsfc"}.issubset(names):
            rh = []
            for year in years:
                with xr.open_dataset(args.ic_dir / f"ic_lag_{year}0501_25m.nc") as ds:
                    t = extract_time_field(ds["TMP2m"], agg)
                    q = extract_time_field(ds["Q2m"], agg)
                    p = extract_time_field(ds["PRESsfc"], agg)
                rh.append(rh_from_q(t, q, p))
            yield Predictor(f"ic_{agg}_RH2m_pct", "initial_condition_derived", np.stack(rh, axis=0), lat, lon, land, ocean)

        if {"UGRD10m", "VGRD10m"}.issubset(names):
            wspd = []
            for year in years:
                with xr.open_dataset(args.ic_dir / f"ic_lag_{year}0501_25m.nc") as ds:
                    u = extract_time_field(ds["UGRD10m"], agg)
                    v = extract_time_field(ds["VGRD10m"], agg)
                wspd.append(np.sqrt(u * u + v * v).astype(np.float32))
            yield Predictor(f"ic_{agg}_wind10m_speed", "initial_condition_derived", np.stack(wspd, axis=0), lat, lon, land, ocean)


def forcing_masks(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    land = ds["land_fraction"]
    land = land.isel(time=0) if "time" in land.dims else land
    land_mask = land.values > 0.5
    ocean = ds["ocean_fraction"]
    ocean = ocean.isel(time=0) if "time" in ocean.dims else ocean
    ocean_mask = ocean.values > 0.5
    return land_mask, ocean_mask


def iter_forcing_predictors(args: argparse.Namespace, years: list[int]):
    first = args.forcing_dir / f"forcing_{years[0]}.nc"
    if not first.exists():
        raise FileNotFoundError(f"Missing forcing file: {first}")
    with xr.open_dataset(first) as ds0:
        lat = ds0["latitude"].values.astype(np.float32)
        lon = lon_to_360(ds0["longitude"].values.astype(np.float32))
        land_mask, ocean_mask = forcing_masks(ds0)

    for win_name, (start, end) in SURFACE_WINDOWS.items():
        for domain in ("land", "ocean"):
            arrs = []
            ice_acc = []
            for year in years:
                with xr.open_dataset(args.forcing_dir / f"forcing_{year}.nc") as ds:
                    st = ds["surface_temperature"].sel(time=slice(f"{year}-{start}", f"{year}-{end}")).mean("time")
                    values = st.values.astype(np.float32)
                    if domain == "ocean" and "sea_ice_fraction" in ds:
                        ice = ds["sea_ice_fraction"].sel(time=slice(f"{year}-{start}", f"{year}-{end}")).mean("time")
                        ice_acc.append(ice.values.astype(np.float32))
                    arrs.append(values)
            data = np.stack(arrs, axis=0)
            if domain == "land":
                data = np.where(land_mask[None, :, :], data, np.nan)
                label = f"forcing_land_surface_temperature_{win_name}_K"
                mask = land_mask
                other = ocean_mask
                group = "forcing_land_surface_temperature"
            else:
                mask = ocean_mask.copy()
                if ice_acc:
                    clim_ice = np.nanmean(np.stack(ice_acc, axis=0), axis=0)
                    mask = mask & (clim_ice <= 0.15)
                data = np.where(mask[None, :, :], data, np.nan)
                label = f"forcing_ocean_sst_{win_name}_K"
                other = land_mask
                group = "forcing_sst"
            yield Predictor(label, group, data.astype(np.float32), lat, lon, mask if domain == "land" else other, mask if domain == "ocean" else other)


def find_best_box(
    corr: np.ndarray,
    pval: np.ndarray,
    lat: np.ndarray,
    lon_360: np.ndarray,
    mask: np.ndarray | None,
    search_lat: tuple[float, float],
    search_lon: tuple[float, float],
    lat_span: float,
    lon_span: float,
    min_cells: int,
) -> dict:
    if mask is None:
        mask = np.ones_like(corr, dtype=bool)
    best = None
    cand_lat = lat[(lat >= search_lat[0]) & (lat <= search_lat[1])]
    cand_lon = lon_360[(lon_360 >= search_lon[0]) & (lon_360 <= search_lon[1])]
    for la0 in cand_lat:
        la1 = float(la0) + lat_span
        if la1 > search_lat[1]:
            continue
        sla = (lat >= la0) & (lat <= la1)
        for lo0 in cand_lon:
            lo1 = float(lo0) + lon_span
            if lo1 > search_lon[1]:
                continue
            slo = (lon_360 >= lo0) & (lon_360 <= lo1)
            sub = corr[np.ix_(sla, slo)]
            sub_p = pval[np.ix_(sla, slo)]
            sub_m = mask[np.ix_(sla, slo)]
            finite = np.isfinite(sub) & sub_m
            n = int(finite.sum())
            if n < min_cells:
                continue
            mean_r = float(np.nanmean(sub[finite]))
            score = abs(mean_r)
            if best is None or score > best["abs_mean_r"]:
                sig = np.isfinite(sub_p) & (sub_p < 0.05) & finite
                best = {
                    "lat_s": float(la0),
                    "lat_n": float(la1),
                    "lon_w_360": float(lo0),
                    "lon_e_360": float(lo1),
                    "lon_w": float(360.0 - lo1),
                    "lon_e": float(360.0 - lo0),
                    "mean_r": mean_r,
                    "abs_mean_r": score,
                    "n_cells": n,
                    "p05_frac": float(sig.sum() / max(n, 1)),
                }
    if best is None:
        return {
            "lat_s": np.nan,
            "lat_n": np.nan,
            "lon_w_360": np.nan,
            "lon_e_360": np.nan,
            "lon_w": np.nan,
            "lon_e": np.nan,
            "mean_r": np.nan,
            "abs_mean_r": np.nan,
            "n_cells": 0,
            "p05_frac": np.nan,
        }
    return best


def map_summary(corr: np.ndarray) -> tuple[float, float]:
    finite = np.abs(corr[np.isfinite(corr)])
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.nanmax(finite)), float(np.nanpercentile(finite, 95))


def analyze_predictor(
    predictor: Predictor,
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
    map_records: dict[tuple[str, str], dict],
    ds_vars: dict[str, tuple],
) -> list[dict]:
    rows = []
    lon_360 = lon_to_360(predictor.lon)
    pred_safe = safe_name(predictor.name)
    print(f"screening {predictor.name}", flush=True)
    for target_name, target in targets.items():
        corr, pval = corr_p_map(predictor.data, target)
        max_abs, p95_abs = map_summary(corr)
        for domain, mask, lat_span, lon_span, min_cells in (
            ("land", predictor.land_mask, 8.0, 12.0, 8),
            ("ocean", predictor.ocean_mask, 10.0, 15.0, 12),
        ):
            box = find_best_box(
                corr,
                pval,
                predictor.lat,
                lon_360,
                mask,
                tuple(args.search_lat),
                tuple(args.search_lon),
                lat_span,
                lon_span,
                min_cells,
            )
            rows.append(
                {
                    "predictor": predictor.name,
                    "predictor_group": predictor.group,
                    "target": target_name,
                    "domain": domain,
                    "mean_r": box["mean_r"],
                    "abs_mean_r": box["abs_mean_r"],
                    "p05_frac": box["p05_frac"],
                    "n_cells": box["n_cells"],
                    "lat_s": box["lat_s"],
                    "lat_n": box["lat_n"],
                    "lon_w": box["lon_w"],
                    "lon_e": box["lon_e"],
                    "lon_w_360": box["lon_w_360"],
                    "lon_e_360": box["lon_e_360"],
                    "max_abs_cell_r": max_abs,
                    "p95_abs_cell_r": p95_abs,
                }
            )
        key = (target_name, pred_safe)
        map_records[key] = {
            "predictor": predictor.name,
            "target": target_name,
            "corr": corr,
            "pval": pval,
            "lat": predictor.lat,
            "lon": lon_360,
        }
        if not args.no_save_maps:
            ds_vars[f"corr_{target_name}_{pred_safe}"] = (("lat", "lon"), corr.astype(np.float32))
            ds_vars[f"pval_{target_name}_{pred_safe}"] = (("lat", "lon"), pval.astype(np.float32))
    return rows


def write_rows(rows: list[dict], out_csv: Path):
    rows = sorted(rows, key=lambda r: (r["target"], -float(np.nan_to_num(r["abs_mean_r"], nan=-1.0))))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_box(ax, row: dict):
    if not np.isfinite(row["lat_s"]):
        return
    x0 = row["lon_w_360"] - 360.0 if row["lon_w_360"] > 180.0 else row["lon_w_360"]
    x1 = row["lon_e_360"] - 360.0 if row["lon_e_360"] > 180.0 else row["lon_e_360"]
    ax.add_patch(
        plt.Rectangle(
            (x0, row["lat_s"]),
            x1 - x0,
            row["lat_n"] - row["lat_s"],
            fill=False,
            edgecolor="#00b050",
            lw=2.0,
            zorder=8,
        )
    )


def render_top_maps(rows: list[dict], map_records: dict[tuple[str, str], dict], out_path: Path, top_n: int):
    era5_rows = [r for r in rows if r["target"] == "era5" and np.isfinite(r["abs_mean_r"])]
    best_by_pred = {}
    for r in era5_rows:
        key = r["predictor"]
        if key not in best_by_pred or r["abs_mean_r"] > best_by_pred[key]["abs_mean_r"]:
            best_by_pred[key] = r
    top = sorted(best_by_pred.values(), key=lambda r: -r["abs_mean_r"])[:top_n]
    if not top:
        return

    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.7 * nrows), squeeze=False, constrained_layout=True)
    mesh = None
    for ax, row in zip(axes.flat, top):
        rec = map_records[("era5", safe_name(row["predictor"]))]
        corr_r, lon_r = roll_to_180(rec["corr"], rec["lon"])
        pval_r, _ = roll_to_180(rec["pval"], rec["lon"])
        lat = rec["lat"]
        lon_sel = (lon_r >= -170.0) & (lon_r <= -30.0)
        lat_sel = (lat >= 15.0) & (lat <= 75.0)
        corr_sub = corr_r[lat_sel, :][:, lon_sel]
        pval_sub = pval_r[lat_sel, :][:, lon_sel]
        lat_sub = lat[lat_sel]
        lon_sub = lon_r[lon_sel]
        lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
        ax.set_facecolor("#d0e8f0")
        mesh = ax.pcolormesh(lon2d, lat2d, corr_sub, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="nearest", zorder=1)
        _draw_coast_and_borders(ax, (-170.0, -30.0), (15.0, 75.0))
        sig = _thin_mask(np.isfinite(pval_sub) & (pval_sub < 0.05), max_points=2500)
        ax.scatter(lon2d[sig], lat2d[sig], s=2.0, c="k", alpha=0.45, linewidths=0, zorder=6)
        draw_box(ax, row)
        ax.set_xlim(-170.0, -30.0)
        ax.set_ylim(15.0, 75.0)
        ax.set_aspect("equal")
        ax.set_title(
            f"{row['predictor']}\n{row['domain']} box mean r={row['mean_r']:.2f}",
            fontsize=8,
        )
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    fig.colorbar(mesh, ax=axes, shrink=0.82, label="Pearson r vs detrended ERA5 RAW box index")
    fig.suptitle("Top available RAW JJA source-screen predictors", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    years = parse_years(args.years)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    targets, box_info, _, _, _, _ = load_targets(args, years)
    print("RAW target box:", json.dumps(box_info["raw_rankcorr_box"]), flush=True)

    series_ds = xr.Dataset(
        {f"{name}_raw_box_index": (("year",), values.astype(np.float32)) for name, values in targets.items()},
        coords={"year": years},
        attrs={
            "description": "RAW JJA heat-extreme frequency index over the selected rank-correlation box",
            "target_box_json": json.dumps(box_info),
        },
    )
    series_ds.to_netcdf(args.out_dir / "raw_box_indices.nc")

    rows: list[dict] = []
    map_records: dict[tuple[str, str], dict] = {}
    ds_vars: dict[str, tuple] = {}
    map_lat = map_lon = None

    if not args.skip_ic:
        for pred in iter_ic_predictors(args, years):
            map_lat = pred.lat if map_lat is None else map_lat
            map_lon = pred.lon if map_lon is None else map_lon
            rows.extend(analyze_predictor(pred, targets, args, map_records, ds_vars))

    if not args.skip_forcing:
        for pred in iter_forcing_predictors(args, years):
            map_lat = pred.lat if map_lat is None else map_lat
            map_lon = pred.lon if map_lon is None else map_lon
            rows.extend(analyze_predictor(pred, targets, args, map_records, ds_vars))

    if not rows:
        raise RuntimeError("No predictors were screened.")

    summary_csv = args.out_dir / "raw_source_screen_summary.csv"
    write_rows(rows, summary_csv)
    print(f"wrote {summary_csv}", flush=True)

    summary_json = args.out_dir / "raw_source_screen_summary.json"
    top_rows = sorted(rows, key=lambda r: -float(np.nan_to_num(r["abs_mean_r"], nan=-1.0)))[:20]
    summary_json.write_text(json.dumps({"target_box": box_info, "top_rows": top_rows}, indent=2))
    print(f"wrote {summary_json}", flush=True)

    if not args.no_save_maps and ds_vars:
        xr.Dataset(
            ds_vars,
            coords={"lat": map_lat.astype(np.float32), "lon": lon_to_360(map_lon).astype(np.float32)},
            attrs={
                "description": "Detrended Pearson-r maps vs RAW JJA rank-correlation-box index",
                "target_box": json.dumps(box_info),
            },
        ).to_netcdf(args.out_dir / "raw_source_screen_maps.nc")
        print(f"wrote {args.out_dir / 'raw_source_screen_maps.nc'}", flush=True)

    render_top_maps(rows, map_records, args.out_dir / "figures" / "raw_source_screen_top_era5.png", args.top_n)
    print(f"wrote {args.out_dir / 'figures' / 'raw_source_screen_top_era5.png'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
