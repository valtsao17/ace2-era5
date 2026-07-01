#!/usr/bin/env python3
"""DJF heat/cold extreme postprocessing for the Nov-1 lag ensemble (2002-2015).

Approach (Jia et al. 2023 style, mirroring postprocess_jja_lag.py):
  - Raw daily Tmax/Tmin values used directly (no anomaly removal).
  - Extreme threshold: 90th pct (heat) or 10th pct (cold) of a ±WINDOW_DAYS
    rolling calendar window from all OTHER years (LOO) — separate for ERA5
    and ACE2, absorbing mean and variance bias.
  - Predicted probability: fraction of 25 ACE2 members exceeding (heat) or
    below (cold) the ACE2 LOO threshold on the target date.
  - Observed extreme: ERA5 Tmax/Tmin exceeds / falls below ERA5 LOO threshold.

Target dates: Dec 1 (lead ~30d), Jan 1 (lead ~61d), Feb 1 (lead ~92d)
  Jan 1 and Feb 1 are calendar year init_year+1.
Init years: 2002-2015 (14 complete runs)  →  14 × 3 = 42 events per grid cell

WINDOW_DAYS = 7   (±7 days = 15-day total window, per Jia et al. 2023)

Outputs in outputs/lag_nov/postprocess_djf/:
  thresholds/era5_loo_{heat|cold}_{mmdd}.nc   (year, lat, lon)
  thresholds/ace2_loo_{heat|cold}_{mmdd}.nc   (year, lat, lon)
  metrics/tau_map_djf_{heat|cold}.nc           (lat, lon)
  figures/tau_djf_{heat|cold}_conus.png        bb_conus.png style
  figures/tau_djf_{heat|cold}_global.png       global map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.prepared import prep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.era5 import cache_era5_month
from hiro_ace_pipeline.io import write_atomic

COMBINED_DIR  = PROJECT_ROOT / "outputs/lag_nov/combined_djf"
OUT_ROOT      = PROJECT_ROOT / "outputs/lag_nov/postprocess_djf"
ERA5_CACHE    = OUT_ROOT / "era5_cache"
THRESH_DIR    = OUT_ROOT / "thresholds"
METRICS_DIR   = OUT_ROOT / "metrics"
FIGURES_DIR   = OUT_ROOT / "figures"
FORCING_DIR   = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"

ERA5_ZARR   = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BBOX        = (-90.0, 90.0, 0.0, 360.0)

YEARS       = list(range(2002, 2016))   # init years (14 complete)
N_MEMBERS   = 25
WINDOW_DAYS = 7                         # ±7 days = 15-day total window

# (label, month, day, nominal lead from Nov 1 center)
# Jan1 and Feb1 targets are in calendar year init_year+1
TARGETS = [
    ("Dec1", 12, 1, 30),
    ("Jan1",  1, 1, 61),
    ("Feb1",  2, 1, 92),
]

CONUS_EXTENT = (-175.0, -50.0, 12.0, 77.0)   # lon_min, lon_max, lat_min, lat_max

_SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"],
    N=256,
)
_SKILL_CMAP.set_bad("white")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        10,
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
})


# ── helpers ────────────────────────────────────────────────────────────────

def abs_target_year(init_year: int, month: int) -> int:
    """Dec targets stay in init_year; Jan/Feb targets are init_year+1."""
    return init_year if month == 12 else init_year + 1


def get_template() -> xr.DataArray:
    """1-degree lat/lon template from first combined DJF file."""
    path = COMBINED_DIR / f"tmax_djf_{YEARS[0]}.nc"
    with xr.open_dataset(path) as ds:
        return ds["TMP2m"].isel(member=0, time=0).load()


def ace2_window_data(init_year: int, tm: int, td: int, var: str) -> np.ndarray:
    """ACE2 window data around target date: (member*n_window, lat, lon) in °C."""
    target = pd.Timestamp(year=abs_target_year(init_year, tm), month=tm, day=td)
    t_lo   = target - pd.Timedelta(days=WINDOW_DAYS)
    t_hi   = target + pd.Timedelta(days=WINDOW_DAYS)

    path = COMBINED_DIR / f"{var}_djf_{init_year}.nc"
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        mask  = (times >= t_lo) & (times <= t_hi)
        da    = ds["TMP2m"].isel(time=mask).values   # (member, n_window, lat, lon)

    n_m, n_t, n_lat, n_lon = da.shape
    return (da.reshape(n_m * n_t, n_lat, n_lon) - 273.15).astype(np.float32)


def ace2_at_target(init_year: int, tm: int, td: int, var: str) -> np.ndarray:
    """ACE2 Tmax/Tmin at exact target date: (member, lat, lon) in °C."""
    target = pd.Timestamp(year=abs_target_year(init_year, tm), month=tm, day=td)
    path   = COMBINED_DIR / f"{var}_djf_{init_year}.nc"
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idx   = int(np.argmin(np.abs(times - target)))
        da    = ds["TMP2m"].isel(time=idx).values   # (member, lat, lon)
    return (da - 273.15).astype(np.float32)


def era5_window_data(abs_year: int, tm: int, td: int,
                     template: xr.DataArray) -> np.ndarray:
    """ERA5 ±WINDOW_DAYS window around (abs_year, tm, td): (n_days, lat, lon) in °C."""
    target = pd.Timestamp(year=abs_year, month=tm, day=td)
    t_lo   = target - pd.Timedelta(days=WINDOW_DAYS)
    t_hi   = target + pd.Timedelta(days=WINDOW_DAYS)

    months_needed = set()
    for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
        d = target + pd.Timedelta(days=delta)
        months_needed.add((d.year, d.month))

    arrays = []
    for y, m in sorted(months_needed):
        p = cache_era5_month(ERA5_ZARR, y, m, BBOX, template, ERA5_CACHE)
        with xr.open_dataset(p) as ds_m:
            da_m  = ds_m["era5_daily_tmax_C"].load()
            times = pd.DatetimeIndex(da_m.time.values)
            mask  = (times >= t_lo) & (times <= t_hi)
            if mask.any():
                arrays.append(da_m.isel(time=mask).values)

    if not arrays:
        tmpl = template.values
        return np.full((1, tmpl.shape[-2], tmpl.shape[-1]), np.nan, dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def era5_tmin_window_data(abs_year: int, tm: int, td: int,
                          template: xr.DataArray) -> np.ndarray:
    """ERA5 daily Tmin window: (n_days, lat, lon) in °C.
    Fetches from ZARR since cache_era5_month only stores Tmax.
    """
    target = pd.Timestamp(year=abs_year, month=tm, day=td)
    t_lo   = target - pd.Timedelta(days=WINDOW_DAYS)
    t_hi   = target + pd.Timedelta(days=WINDOW_DAYS)

    months_needed = set()
    for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
        d = target + pd.Timedelta(days=delta)
        months_needed.add((d.year, d.month))

    arrays = []
    for y, m in sorted(months_needed):
        cache_path = ERA5_CACHE / f"era5_tmin_{y}{m:02d}.nc"
        if not cache_path.exists():
            _cache_era5_tmin_month(y, m, template, cache_path)
        with xr.open_dataset(cache_path) as ds_m:
            da_m  = ds_m["era5_daily_tmin_C"].load()
            times = pd.DatetimeIndex(da_m.time.values)
            mask  = (times >= t_lo) & (times <= t_hi)
            if mask.any():
                arrays.append(da_m.isel(time=mask).values)

    if not arrays:
        tmpl = template.values
        return np.full((1, tmpl.shape[-2], tmpl.shape[-1]), np.nan, dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _cache_era5_tmin_month(year: int, month: int, template: xr.DataArray,
                           out_path: Path, retries: int = 5) -> None:
    import time as _time
    for attempt in range(retries):
        try:
            import socket
            socket.setdefaulttimeout(120)
            ds    = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={})
            vname = next(v for v in ["2m_temperature", "t2m", "TMP2m"] if v in ds)
            start = pd.Timestamp(year=year, month=month, day=1)
            end   = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
            da    = ds[vname].sel(time=slice(str(start), str(end)))
            lat_n = next(c for c in da.coords if c in ("latitude", "lat"))
            lon_n = next(c for c in da.coords if c in ("longitude", "lon"))
            t_lat = next(c for c in template.coords if c in ("lat", "latitude"))
            t_lon = next(c for c in template.coords if c in ("lon", "longitude"))
            da    = da.sel({lat_n: template[t_lat].values,
                            lon_n: template[t_lon].values}, method="nearest")
            da    = da.assign_coords({lat_n: template[t_lat].values,
                                      lon_n: template[t_lon].values})
            daily = (da - 273.15).resample(time="1D").min()
            daily = daily.sel(time=daily.time.dt.month == month).astype("float32").load()
            daily = daily.rename({lat_n: "lat", lon_n: "lon"})
            ds.close()
            write_atomic(
                daily.rename("era5_daily_tmin_C").to_dataset(), out_path
            )
            return
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"    RETRY ERA5 tmin {year}-{month:02d} attempt {attempt+1}/{retries}: {e}",
                  flush=True)
            _time.sleep(wait)
    raise RuntimeError(f"Failed ERA5 tmin {year}-{month:02d}")


def era5_at_target_tmax(abs_year: int, tm: int, td: int,
                        template: xr.DataArray) -> np.ndarray:
    """ERA5 Tmax on exact date: (lat, lon) in °C."""
    target = pd.Timestamp(year=abs_year, month=tm, day=td)
    p = cache_era5_month(ERA5_ZARR, abs_year, tm, BBOX, template, ERA5_CACHE)
    with xr.open_dataset(p) as ds_m:
        da_m  = ds_m["era5_daily_tmax_C"].load()
        times = pd.DatetimeIndex(da_m.time.values)
        idx   = int(np.argmin(np.abs(times - target)))
        return da_m.isel(time=idx).values.astype(np.float32)


def era5_at_target_tmin(abs_year: int, tm: int, td: int,
                        template: xr.DataArray) -> np.ndarray:
    """ERA5 Tmin on exact date: (lat, lon) in °C."""
    target = pd.Timestamp(year=abs_year, month=tm, day=td)
    cache_path = ERA5_CACHE / f"era5_tmin_{abs_year}{tm:02d}.nc"
    if not cache_path.exists():
        _cache_era5_tmin_month(abs_year, tm, template, cache_path)
    with xr.open_dataset(cache_path) as ds_m:
        da_m  = ds_m["era5_daily_tmin_C"].load()
        times = pd.DatetimeIndex(da_m.time.values)
        idx   = int(np.argmin(np.abs(times - target)))
        return da_m.isel(time=idx).values.astype(np.float32)


# ── LOO threshold computation ──────────────────────────────────────────────

def _loo_thresh(pool: list[np.ndarray], pct: float) -> np.ndarray:
    """For each year index, compute LOO quantile from all other years."""
    n_years = len(pool)
    n_lat, n_lon = pool[0].shape[1], pool[0].shape[2]
    thresh = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    CHUNK = 10
    for lat0 in tqdm(range(0, n_lat, CHUNK), desc="  lat chunks", leave=False):
        lat1 = min(lat0 + CHUNK, n_lat)
        for y_idx in range(n_years):
            others = [pool[i][:, lat0:lat1, :] for i in range(n_years) if i != y_idx]
            pooled = np.concatenate(others, axis=0)
            thresh[y_idx, lat0:lat1, :] = np.nanquantile(pooled, pct / 100.0, axis=0)
    return thresh


def compute_era5_loo_thresholds(extreme: str, tm: int, td: int,
                                template: xr.DataArray,
                                force: bool = False) -> np.ndarray:
    """Compute LOO ERA5 threshold for all init years.  Returns (n_years, lat, lon)."""
    pct   = 90.0 if extreme == "heat" else 10.0
    label = f"{tm:02d}{td:02d}"
    out   = THRESH_DIR / f"era5_loo_{extreme}_{label}.nc"
    if out.exists() and not force:
        with xr.open_dataset(out) as ds:
            return ds["threshold"].values

    print(f"  Loading ERA5 window data ({extreme} {label}) ...", flush=True)
    if extreme == "heat":
        pool = [era5_window_data(abs_target_year(y, tm), tm, td, template)
                for y in tqdm(YEARS, desc=f"  ERA5 {label}", leave=False)]
    else:
        pool = [era5_tmin_window_data(abs_target_year(y, tm), tm, td, template)
                for y in tqdm(YEARS, desc=f"  ERA5 {label}", leave=False)]

    print(f"  Computing LOO ERA5 thresholds ...", flush=True)
    thresh = _loo_thresh(pool, pct)

    lat = template["lat"].values if "lat" in template.coords else np.arange(thresh.shape[1])
    lon = template["lon"].values if "lon" in template.coords else np.arange(thresh.shape[2])
    da  = xr.DataArray(thresh, dims=["year", "lat", "lon"],
                       coords={"year": YEARS, "lat": lat, "lon": lon},
                       attrs={"units": "degC", "pct": pct, "window_days": WINDOW_DAYS})
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(da.rename("threshold").to_dataset(), out)
    print(f"  saved {out.name}", flush=True)
    return thresh


def compute_ace2_loo_thresholds(extreme: str, tm: int, td: int,
                                template: xr.DataArray,
                                force: bool = False) -> np.ndarray:
    """Compute LOO ACE2 threshold for all init years.  Returns (n_years, lat, lon)."""
    pct  = 90.0 if extreme == "heat" else 10.0
    var  = "tmax" if extreme == "heat" else "tmin"
    label = f"{tm:02d}{td:02d}"
    out   = THRESH_DIR / f"ace2_loo_{extreme}_{label}.nc"
    if out.exists() and not force:
        with xr.open_dataset(out) as ds:
            return ds["threshold"].values

    print(f"  Loading ACE2 window data ({extreme} {label}) ...", flush=True)
    pool = [ace2_window_data(y, tm, td, var)
            for y in tqdm(YEARS, desc=f"  ACE2 {label}", leave=False)]

    print(f"  Computing LOO ACE2 thresholds ...", flush=True)
    thresh = _loo_thresh(pool, pct)

    lat = template["lat"].values if "lat" in template.coords else np.arange(thresh.shape[1])
    lon = template["lon"].values if "lon" in template.coords else np.arange(thresh.shape[2])
    da  = xr.DataArray(thresh, dims=["year", "lat", "lon"],
                       coords={"year": YEARS, "lat": lat, "lon": lon},
                       attrs={"units": "degC", "pct": pct, "window_days": WINDOW_DAYS})
    write_atomic(da.rename("threshold").to_dataset(), out)
    print(f"  saved {out.name}", flush=True)
    return thresh


# ── events ────────────────────────────────────────────────────────────────

def compute_events(extreme: str, tm: int, td: int,
                   era5_thresh: np.ndarray, ace2_thresh: np.ndarray,
                   template: xr.DataArray):
    """Return (pred_prob, obs_ext), each (n_years, lat, lon)."""
    var     = "tmax" if extreme == "heat" else "tmin"
    n_years = len(YEARS)
    n_lat   = era5_thresh.shape[1]
    n_lon   = era5_thresh.shape[2]
    pred_prob = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)
    obs_ext   = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    for y_idx, year in enumerate(tqdm(YEARS, desc=f"  events {tm:02d}{td:02d}")):
        abs_yr = abs_target_year(year, tm)
        ace2   = ace2_at_target(year, tm, td, var)       # (25, lat, lon)
        if extreme == "heat":
            era5 = era5_at_target_tmax(abs_yr, tm, td, template)
            at   = ace2_thresh[y_idx]
            et   = era5_thresh[y_idx]
            pred_prob[y_idx] = (ace2 > at[np.newaxis]).mean(axis=0)
            obs_ext[y_idx]   = (era5 > et).astype(np.float32)
        else:
            era5 = era5_at_target_tmin(abs_yr, tm, td, template)
            at   = ace2_thresh[y_idx]
            et   = era5_thresh[y_idx]
            pred_prob[y_idx] = (ace2 < at[np.newaxis]).mean(axis=0)
            obs_ext[y_idx]   = (era5 < et).astype(np.float32)

    return pred_prob, obs_ext


# ── scoring ────────────────────────────────────────────────────────────────

def kendall_tau_map(pred: np.ndarray, obs: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Kendall tau and p-value per grid cell.  Returns (tau_map, pval_map)."""
    n_events, n_lat, n_lon = pred.shape
    tau_map  = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    pval_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    for i in range(n_lat):
        for j in range(n_lon):
            p = pred[:, i, j]
            o = obs[:, i, j]
            valid = np.isfinite(p) & np.isfinite(o)
            if valid.sum() < 5:
                continue
            if o[valid].std() == 0:
                continue
            t, pv = kendalltau(p[valid], o[valid])
            tau_map[i, j]  = t
            pval_map[i, j] = pv

    return tau_map, pval_map


def cos_lat_mean(field: np.ndarray, lat: np.ndarray,
                 mask: np.ndarray | None = None) -> float:
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    f = field if mask is None else np.where(mask, field, np.nan)
    valid = np.isfinite(f)
    num = np.nansum(f * weights * valid)
    den = np.nansum(weights * valid)
    return float(num / den) if den > 0 else float("nan")


# ── land mask & borders ────────────────────────────────────────────────────

_LAND_GEOMS_CACHE = None
_STATE_GEOMS_CACHE = None


def _get_border_geoms():
    global _LAND_GEOMS_CACHE, _STATE_GEOMS_CACHE
    if _LAND_GEOMS_CACHE is None:
        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMS_CACHE = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
        _STATE_GEOMS_CACHE = list(shpreader.Reader(shp).geometries())
    return _LAND_GEOMS_CACHE, _STATE_GEOMS_CACHE


def _draw_borders(ax, xlim, ylim):
    from shapely.geometry import box
    viewport = box(xlim[0], ylim[0], xlim[1], ylim[1])
    land_geoms, state_geoms = _get_border_geoms()

    def _plot_geom(geom, color, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot_geom(g, color, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)

    for geom in land_geoms:
        try:
            _plot_geom(geom.intersection(viewport), "black", 0.5)
        except Exception:
            continue

    for geom in state_geoms:
        try:
            _plot_geom(geom.intersection(viewport), "0.4", 0.3)
        except Exception:
            continue


def load_land_mask(lat: np.ndarray, lon_360: np.ndarray) -> np.ndarray:
    """Boolean land mask (True=land) from forcing data land_fraction."""
    forcing_file = next(FORCING_DIR.glob("forcing_*.nc"), None)
    if forcing_file is None:
        return None
    with xr.open_dataset(forcing_file) as ds:
        lf = ds["land_fraction"]
        lat_n = "latitude" if "latitude" in lf.dims else "lat"
        lon_n = "longitude" if "longitude" in lf.dims else "lon"
        lf2 = lf.interp({lat_n: lat, lon_n: lon_360}, method="nearest")
    return lf2.values > 0.5


# ── figures ───────────────────────────────────────────────────────────────

def plot_tau_conus(tau: np.ndarray, pval: np.ndarray,
                   lat: np.ndarray, lon_360: np.ndarray,
                   extreme: str, title: str, out_path: Path,
                   land_mask: np.ndarray | None):
    """CONUS+Alaska signed Kendall tau map, land only, stippled non-sig points."""
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT

    # Convert lon 0-360 → -180..180
    lon = np.where(lon_360 > 180, lon_360 - 360.0, lon_360)

    # Subset to CONUS+AK extent
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon >= lon_min) & (lon <= lon_max)
    lat_s   = lat[lat_sel]
    lon_s   = lon[lon_sel]

    tau_s  = tau[np.ix_(lat_sel, lon_sel)]
    pval_s = pval[np.ix_(lat_sel, lon_sel)]
    lm_s   = land_mask[np.ix_(lat_sel, lon_sel)] if land_mask is not None else None

    # Signed tau, NaN over ocean.
    plot_data = tau_s
    if lm_s is not None:
        plot_data = np.where(lm_s, plot_data, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white")

    extent = [lon_s.min() - 0.5, lon_s.max() + 0.5,
              lat_s.min() - 0.5, lat_s.max() + 0.5]
    im = ax.imshow(plot_data, extent=extent, origin="lower",
                   cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   aspect="auto", interpolation="nearest", zorder=1)

    # Coastlines and state borders
    _draw_borders(ax, (lon_min - 2, lon_max + 2), (lat_min - 2, lat_max + 2))

    # Stipple non-significant land points
    LON2D, LAT2D = np.meshgrid(lon_s, lat_s)
    not_sig = (pval_s > 0.05)
    if lm_s is not None:
        not_sig = not_sig & lm_s & np.isfinite(tau_s)
    ax.plot(LON2D[not_sig].ravel(), LAT2D[not_sig].ravel(),
            "k.", markersize=1.5, alpha=0.55, zorder=4)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Tick labels
    lon_ticks = np.arange(-160, -40, 20)
    lat_ticks = np.arange(20, 80, 10)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{abs(x)}°W" for x in lon_ticks], fontsize=8)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{y}°N" for y in lat_ticks], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    # Area-weighted land mean
    if lm_s is not None:
        wavg = cos_lat_mean(np.abs(tau_s), lat_s, mask=lm_s & np.isfinite(tau_s))
    else:
        wavg = cos_lat_mean(np.abs(tau_s), lat_s)
    ax.text(0.98, 0.04, f"{wavg:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3), zorder=5)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.65, pad=0.07, aspect=30, extend="neither")
    cbar.set_label("Rank correlation (Kendall τ)", fontsize=9)
    ticks = np.linspace(-1.0, 1.0, 9)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=7)

    ax.set_title(title, fontsize=10, pad=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote: {out_path}", flush=True)


def plot_tau_global(tau: np.ndarray, lat: np.ndarray, lon_360: np.ndarray,
                    extreme: str, title: str, out_path: Path):
    """Simple global signed tau map on plain axes (no cartopy to avoid crashes)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(figsize=(14, 6),
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    mesh = ax.pcolormesh(lon_360, lat, tau, shading="auto",
                  cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                  transform=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black",
                   linewidth=0.4, zorder=3)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="Kendall τ")
    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote: {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────

def run_extreme(extreme: str, template: xr.DataArray, lat: np.ndarray,
                lon: np.ndarray, land_mask: np.ndarray, force: bool):
    season_label = "2002/03–2015/16"
    n_events = len(YEARS) * len(TARGETS)
    print(f"\n{'='*60}\n{extreme.upper()}  ({n_events} events)\n{'='*60}", flush=True)

    all_pred, all_obs = [], []

    for label, tm, td, lead in TARGETS:
        print(f"\n--- Target {label} (lead ~{lead}d) ---", flush=True)
        era5_thresh = compute_era5_loo_thresholds(extreme, tm, td, template, force)
        ace2_thresh = compute_ace2_loo_thresholds(extreme, tm, td, template, force)
        pred, obs   = compute_events(extreme, tm, td, era5_thresh, ace2_thresh, template)
        all_pred.append(pred)
        all_obs.append(obs)

    pred_djf = np.concatenate(all_pred, axis=0)   # (n_events, lat, lon)
    obs_djf  = np.concatenate(all_obs,  axis=0)

    print(f"\nComputing Kendall tau across {n_events} pooled events ...", flush=True)
    tau_map, pval_map = kendall_tau_map(pred_djf, obs_djf)

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(tau_map, dims=["lat", "lon"],
                      coords={"lat": lat, "lon": lon},
                      attrs={"n_events": n_events, "window_days": WINDOW_DAYS,
                             "extreme": extreme})
    dap = xr.DataArray(pval_map, dims=["lat", "lon"],
                       coords={"lat": lat, "lon": lon})
    write_atomic(xr.Dataset({"kendall_tau": da, "p_value": dap}),
                 METRICS_DIR / f"tau_map_djf_{extreme}.nc")

    land_tau = cos_lat_mean(np.abs(tau_map), lat,
                            mask=land_mask & np.isfinite(tau_map))
    print(f"  Land-mean |τ| = {land_tau:.3f}", flush=True)

    # CONUS+AK figure
    ext_label = "DJF Tmax > 90th pct" if extreme == "heat" else "DJF Tmin < 10th pct"
    conus_title = (
        f"ACE2-ERA5  |  {extreme.title()} extreme skill  ({ext_label})\n"
        f"Kendall τ  |  DJF pooled  |  {season_label}  "
        f"|  n={n_events}  |  window=±{WINDOW_DAYS}d"
    )
    plot_tau_conus(tau_map, pval_map, lat, lon, extreme, conus_title,
                   FIGURES_DIR / f"tau_djf_{extreme}_conus.png", land_mask)

    # Global figure
    plot_tau_global(tau_map, lat, lon, extreme,
                    f"Kendall τ  DJF {extreme}  {season_label}",
                    FIGURES_DIR / f"tau_djf_{extreme}_global.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Recompute cached thresholds")
    p.add_argument("--extreme", choices=["heat", "cold", "both"], default="both")
    return p.parse_args()


def main():
    args = parse_args()
    for d in [ERA5_CACHE, THRESH_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    template  = get_template()
    lat       = template["lat"].values
    lon       = template["lon"].values
    land_mask = load_land_mask(lat, lon)

    # Pre-cache ERA5 months needed across all targets and years
    print("Pre-caching ERA5 months ...", flush=True)
    months_needed = set()
    for _, tm, td, _ in TARGETS:
        for init_yr in YEARS:
            abs_yr = abs_target_year(init_yr, tm)
            target = pd.Timestamp(year=abs_yr, month=tm, day=td)
            for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
                d = target + pd.Timedelta(days=delta)
                months_needed.add((d.year, d.month))
    for y, m in tqdm(sorted(months_needed), desc="ERA5 month cache"):
        try:
            cache_era5_month(ERA5_ZARR, y, m, BBOX, template, ERA5_CACHE)
        except Exception as e:
            print(f"  WARN ERA5 tmax {y}-{m:02d}: {e}", flush=True)

    extremes = ["heat", "cold"] if args.extreme == "both" else [args.extreme]
    for extreme in extremes:
        run_extreme(extreme, template, lat, lon, land_mask, args.force)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
