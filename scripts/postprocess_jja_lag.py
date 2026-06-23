#!/usr/bin/env python3
"""JJA heat-extreme postprocessing for the May-1 lag ensemble (1980-2016).

Approach (Jia et al. 2023 style):
  - Raw daily Tmax values are used directly (no anomaly baseline removed).
  - Extreme threshold: 90th percentile of a ±WINDOW_DAYS rolling calendar
    window from all OTHER years (leave-one-out) — defined separately for
    ERA5 and ACE2, so model bias in the mean and variance is absorbed.
  - Predicted probability: fraction of 25 ACE2 members exceeding the ACE2
    threshold on the target date.
  - Observed extreme: ERA5 Tmax on the target date exceeds the ERA5 threshold.

Target dates: June 1, July 1, August 1 (lead ~31, ~61, ~92 days from May 1)
Years: 1980–2016  →  37 × 3 = 111 events per grid cell (JJA pooled)

Outputs (in outputs/lag_may/postprocess_jja/):
  thresholds/era5_loo_p90_{mmdd}.nc     – (year, lat, lon)
  thresholds/ace2_loo_p90_{mmdd}.nc     – (year, lat, lon)
  metrics/jja_brier_tau.nc              – per-lead and JJA-pooled scalars
  figures/tau_map_jja.png               – cos-lat-weighted Kendall tau map
  figures/tau_map_{mmdd}.png            – per-lead tau maps
  figures/skill_vs_lead.png             – BS and tau vs lead time
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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat_skill",
    ["white", "#FFE066", "#FF8C00", "#CC0000", "#67000d"],
    N=256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.era5 import cache_era5_month
from hiro_ace_pipeline.io import write_atomic

# ── paths ──────────────────────────────────────────────────────────────────
COMBINED_DIR = PROJECT_ROOT / "outputs/lag_may/combined_jja"
OUT_ROOT     = PROJECT_ROOT / "outputs/lag_may/postprocess_jja"
CACHE_DIR    = OUT_ROOT / "era5_cache"
THRESH_DIR   = OUT_ROOT / "thresholds"
FIGURES_DIR  = OUT_ROOT / "figures"
METRICS_DIR  = OUT_ROOT / "metrics"

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ── constants ──────────────────────────────────────────────────────────────
YEARS      = list(range(1980, 2017))
N_MEMBERS  = 25
WINDOW_DAYS = 15          # ±15 days → 31-day window
BBOX        = (-90.0, 90.0, 0.0, 360.0)   # global 1-degree grid
CHUNK_LAT   = 10          # lat bands processed at once for threshold computation

# (label, month, day, nominal lead from May 1 center)
TARGETS = [
    ("Jun1",  6, 1,  31),
    ("Jul1",  7, 1,  61),
    ("Aug1",  8, 1,  92),
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        10,
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
})

_PLATE = ccrs.PlateCarree()


# ── helpers ────────────────────────────────────────────────────────────────

def get_template() -> xr.DataArray:
    """1-degree lat/lon template from the first combined file."""
    path = COMBINED_DIR / f"tmax_jja_{YEARS[0]}.nc"
    with xr.open_dataset(path) as ds:
        return ds["TMP2m"].isel(member=0, time=0).load()


def ace2_window_data(year: int, target_month: int, target_day: int) -> np.ndarray:
    """Return ACE2 daily Tmax in the ±WINDOW_DAYS window as (samples, lat, lon) in °C."""
    target = pd.Timestamp(year=year, month=target_month, day=target_day)
    t_lo   = target - pd.Timedelta(days=WINDOW_DAYS)
    t_hi   = target + pd.Timedelta(days=WINDOW_DAYS)

    path = COMBINED_DIR / f"tmax_jja_{year}.nc"
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        mask  = (times >= t_lo) & (times <= t_hi)
        da    = ds["TMP2m"].isel(time=mask).values  # (member, n_window, lat, lon)

    n_m, n_t, n_lat, n_lon = da.shape
    data_C = da.reshape(n_m * n_t, n_lat, n_lon) - 273.15   # K → °C
    return data_C.astype(np.float32)


def ace2_at_target(year: int, target_month: int, target_day: int) -> np.ndarray:
    """Return ACE2 daily Tmax at the target date for all 25 members, (member, lat, lon) °C."""
    target = pd.Timestamp(year=year, month=target_month, day=target_day)
    path   = COMBINED_DIR / f"tmax_jja_{year}.nc"
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        idx   = int(np.argmin(np.abs(times - target)))
        da    = ds["TMP2m"].isel(time=idx).values  # (member, lat, lon)
    return (da - 273.15).astype(np.float32)


def era5_window_data(year: int, target_month: int, target_day: int,
                     template: xr.DataArray) -> np.ndarray:
    """Return ERA5 daily Tmax in the ±WINDOW_DAYS window as (n_days, lat, lon) °C."""
    target  = pd.Timestamp(year=year, month=target_month, day=target_day)
    t_lo    = target - pd.Timedelta(days=WINDOW_DAYS)
    t_hi    = target + pd.Timedelta(days=WINDOW_DAYS)

    months_needed = set()
    for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
        d = target + pd.Timedelta(days=delta)
        months_needed.add((d.year, d.month))

    arrays = []
    for y, m in sorted(months_needed):
        if y < YEARS[0] or y > YEARS[-1]:
            continue
        p  = cache_era5_month(ERA5_ZARR, y, m, BBOX, template, CACHE_DIR)
        with xr.open_dataset(p) as ds_m:
            da_m   = ds_m["era5_daily_tmax_C"].load()
            times  = pd.DatetimeIndex(da_m.time.values)
            mask   = (times >= t_lo) & (times <= t_hi)
            if mask.any():
                arrays.append(da_m.isel(time=mask).values)

    if not arrays:
        return np.full((1, template.shape[-2], template.shape[-1]), np.nan, dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)   # (n_days, lat, lon)


def era5_at_target(year: int, target_month: int, target_day: int,
                   template: xr.DataArray) -> np.ndarray:
    """Return ERA5 daily Tmax at target date, (lat, lon) °C."""
    target = pd.Timestamp(year=year, month=target_month, day=target_day)
    p = cache_era5_month(ERA5_ZARR, year, target_month, BBOX, template, CACHE_DIR)
    with xr.open_dataset(p) as ds_m:
        da_m  = ds_m["era5_daily_tmax_C"].load()
        times = pd.DatetimeIndex(da_m.time.values)
        idx   = int(np.argmin(np.abs(times - target)))
        return da_m.isel(time=idx).values.astype(np.float32)


# ── LOO threshold computation ──────────────────────────────────────────────

def compute_era5_loo_thresholds(target_month: int, target_day: int,
                                template: xr.DataArray, force: bool = False) -> np.ndarray:
    """Compute LOO ERA5 90th-pct threshold for all years.

    Returns array of shape (n_years, n_lat, n_lon).
    Cached to THRESH_DIR/era5_loo_p90_{mmdd}.nc.
    """
    label    = f"{target_month:02d}{target_day:02d}"
    out_path = THRESH_DIR / f"era5_loo_p90_{label}.nc"

    if out_path.exists() and not force:
        with xr.open_dataset(out_path) as ds:
            return ds["threshold"].values   # (n_years, n_lat, n_lon)

    n_years = len(YEARS)
    n_lat   = template.sizes.get("lat", template.shape[-2])
    n_lon   = template.sizes.get("lon", template.shape[-1])
    thresh  = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    print(f"  Pre-loading ERA5 window data for {label} ...", flush=True)
    # era5_pool[year_idx] = (n_days_in_window, n_lat, n_lon)
    era5_pool = [
        era5_window_data(y, target_month, target_day, template)
        for y in tqdm(YEARS, desc=f"ERA5 cache {label}")
    ]

    print(f"  Computing LOO ERA5 thresholds (lat chunks) ...", flush=True)
    for lat0 in tqdm(range(0, n_lat, CHUNK_LAT), desc="lat chunks"):
        lat1 = min(lat0 + CHUNK_LAT, n_lat)
        for y_idx in range(n_years):
            other_chunks = [
                era5_pool[i][:, lat0:lat1, :]
                for i in range(n_years) if i != y_idx
            ]
            pooled = np.concatenate(other_chunks, axis=0)   # (samples, lat_chunk, lon)
            thresh[y_idx, lat0:lat1, :] = np.nanquantile(pooled, 0.90, axis=0)

    lat_vals = template["lat"].values if "lat" in template.coords else np.arange(n_lat)
    lon_vals = template["lon"].values if "lon" in template.coords else np.arange(n_lon)

    da = xr.DataArray(
        thresh,
        dims=["year", "lat", "lon"],
        coords={"year": YEARS, "lat": lat_vals, "lon": lon_vals},
        attrs={"units": "degC", "long_name": f"LOO ERA5 90th pct threshold ±{WINDOW_DAYS}d {label}"},
    )
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(da.rename("threshold").to_dataset(), out_path)
    print(f"  saved {out_path.name}", flush=True)
    return thresh


def compute_ace2_loo_thresholds(target_month: int, target_day: int,
                                template: xr.DataArray, force: bool = False) -> np.ndarray:
    """Compute LOO ACE2 90th-pct threshold for all years.

    Returns array of shape (n_years, n_lat, n_lon).
    Cached to THRESH_DIR/ace2_loo_p90_{mmdd}.nc.
    """
    label    = f"{target_month:02d}{target_day:02d}"
    out_path = THRESH_DIR / f"ace2_loo_p90_{label}.nc"

    if out_path.exists() and not force:
        with xr.open_dataset(out_path) as ds:
            return ds["threshold"].values

    n_years = len(YEARS)
    n_lat   = template.sizes.get("lat", template.shape[-2])
    n_lon   = template.sizes.get("lon", template.shape[-1])
    thresh  = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    print(f"  Pre-loading ACE2 window data for {label} ...", flush=True)
    # ace2_pool[year_idx] = (member*n_window, n_lat, n_lon)
    ace2_pool = [
        ace2_window_data(y, target_month, target_day)
        for y in tqdm(YEARS, desc=f"ACE2 load {label}")
    ]

    print(f"  Computing LOO ACE2 thresholds (lat chunks) ...", flush=True)
    for lat0 in tqdm(range(0, n_lat, CHUNK_LAT), desc="lat chunks"):
        lat1 = min(lat0 + CHUNK_LAT, n_lat)
        for y_idx in range(n_years):
            other_chunks = [
                ace2_pool[i][:, lat0:lat1, :]
                for i in range(n_years) if i != y_idx
            ]
            pooled = np.concatenate(other_chunks, axis=0)
            thresh[y_idx, lat0:lat1, :] = np.nanquantile(pooled, 0.90, axis=0)

    lat_vals = template["lat"].values if "lat" in template.coords else np.arange(n_lat)
    lon_vals = template["lon"].values if "lon" in template.coords else np.arange(n_lon)

    da = xr.DataArray(
        thresh,
        dims=["year", "lat", "lon"],
        coords={"year": YEARS, "lat": lat_vals, "lon": lon_vals},
        attrs={"units": "degC", "long_name": f"LOO ACE2 90th pct threshold ±{WINDOW_DAYS}d {label}"},
    )
    write_atomic(da.rename("threshold").to_dataset(), out_path)
    print(f"  saved {out_path.name}", flush=True)
    return thresh


# ── scoring ────────────────────────────────────────────────────────────────

def compute_events(target_month: int, target_day: int,
                   era5_thresh: np.ndarray, ace2_thresh: np.ndarray,
                   template: xr.DataArray):
    """Compute predicted probabilities and observed extremes for all years.

    Returns:
      pred_prob : (n_years, n_lat, n_lon)  fraction of ACE2 members exceeding ACE2 threshold
      obs_ext   : (n_years, n_lat, n_lon)  1 if ERA5 > ERA5 threshold else 0
    """
    n_years = len(YEARS)
    n_lat   = era5_thresh.shape[1]
    n_lon   = era5_thresh.shape[2]
    pred_prob = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)
    obs_ext   = np.full((n_years, n_lat, n_lon), np.nan, dtype=np.float32)

    for y_idx, year in enumerate(tqdm(YEARS, desc=f"events {target_month:02d}{target_day:02d}")):
        ace2 = ace2_at_target(year, target_month, target_day)   # (25, lat, lon) °C
        era5 = era5_at_target(year, target_month, target_day, template)  # (lat, lon) °C

        at = ace2_thresh[y_idx]   # (lat, lon)
        et = era5_thresh[y_idx]   # (lat, lon)

        pred_prob[y_idx] = (ace2 > at[np.newaxis, :, :]).mean(axis=0)
        obs_ext[y_idx]   = (era5 > et).astype(np.float32)

    return pred_prob, obs_ext


def brier_score(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Brier score per grid cell.  pred/obs: (events, lat, lon)."""
    return np.nanmean((pred - obs) ** 2, axis=0)


def brier_skill_score(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """BSS = 1 - BS / BS_clim per grid cell."""
    bs      = brier_score(pred, obs)
    clim    = np.nanmean(obs, axis=0, keepdims=True)
    bs_clim = np.nanmean((clim - obs) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bss = np.where(bs_clim > 0, 1.0 - bs / bs_clim, np.nan)
    return bss.astype(np.float32)


def kendall_tau_map(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Kendall tau per grid cell across events.  Returns (lat, lon)."""
    n_events, n_lat, n_lon = pred.shape
    tau_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    for i in range(n_lat):
        for j in range(n_lon):
            p = pred[:, i, j]
            o = obs[:, i, j]
            valid = np.isfinite(p) & np.isfinite(o)
            if valid.sum() < 10:
                continue
            if o[valid].std() == 0:
                continue
            tau, _ = kendalltau(p[valid], o[valid])
            tau_map[i, j] = tau

    return tau_map


def cos_lat_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """Cosine-latitude-weighted spatial mean of a 2D (lat, lon) field."""
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid   = np.isfinite(field)
    num     = np.nansum(field * weights * valid)
    den     = np.nansum(weights * valid)
    return float(num / den) if den > 0 else float("nan")


def cos_lat_mean_masked(field: np.ndarray, lat: np.ndarray,
                        mask: np.ndarray) -> float:
    """cos-lat weighted mean over grid cells where mask is True."""
    f = np.where(mask, field, np.nan)
    return cos_lat_mean(f, lat)


_LAND_MASK_CACHE: np.ndarray | None = None

def build_land_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return boolean (lat, lon) array, True = land. Cached after first call."""
    global _LAND_MASK_CACHE
    if _LAND_MASK_CACHE is not None:
        return _LAND_MASK_CACHE
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from shapely.geometry import Point

    land = prep(unary_union(
        list(shpreader.Reader(
            shpreader.natural_earth('110m', 'physical', 'land')
        ).geometries())
    ))
    lon_180 = np.where(lon > 180, lon - 360.0, lon)
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon_180):
            mask[i, j] = land.contains(Point(float(lo), float(la)))
    _LAND_MASK_CACHE = mask
    return mask


# ── plotting ───────────────────────────────────────────────────────────────

def _add_skill_stats(ax, field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     label: str, land_mask: np.ndarray | None):
    """Add overall / land / ocean stats text box in the bottom-right corner."""
    overall = cos_lat_mean(field, lat)
    lines   = [f"{label} (overall): {overall:.3f}"]
    if land_mask is not None:
        land_val  = cos_lat_mean_masked(field, lat, land_mask)
        ocean_val = cos_lat_mean_masked(field, lat, ~land_mask)
        lines.append(f"{label} (land):    {land_val:.3f}")
        lines.append(f"{label} (ocean):   {ocean_val:.3f}")
    ax.text(
        0.99, 0.03, "\n".join(lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )


def plot_tau_map(tau: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 title: str, out_path: Path, land_mask: np.ndarray | None = None):
    abs_tau = np.abs(tau)
    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw=dict(projection=_PLATE))
    mesh = ax.pcolormesh(lon, lat, abs_tau,
                         shading="auto", cmap=_HEAT_CMAP,
                         vmin=0.0, vmax=0.4,
                         transform=_PLATE)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="|Kendall τ|")
    ax.set_title(title, fontsize=11)
    _add_skill_stats(ax, abs_tau, lat, lon, "τ", land_mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def plot_bss_map(bss: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 title: str, out_path: Path, land_mask: np.ndarray | None = None):
    abs_bss = np.abs(bss)
    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw=dict(projection=_PLATE))
    mesh = ax.pcolormesh(lon, lat, abs_bss,
                         shading="auto", cmap=_HEAT_CMAP,
                         vmin=0.0, vmax=0.5,
                         transform=_PLATE)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="|BSS|")
    ax.set_title(title, fontsize=11)
    _add_skill_stats(ax, abs_bss, lat, lon, "BSS", land_mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def plot_skill_vs_lead(lead_days: list[int], bss_list: list[float],
                       tau_list: list[float], out_path: Path):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    color_bs  = "#1f77b4"
    color_tau = "#d62728"

    ax1.plot(lead_days, bss_list, "o-", color=color_bs, label="Brier Skill Score")
    ax1.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Lead time (days)")
    ax1.set_ylabel("Brier Skill Score", color=color_bs)
    ax1.tick_params(axis="y", labelcolor=color_bs)
    ax1.set_ylim(-0.1, 0.5)

    ax2 = ax1.twinx()
    ax2.plot(lead_days, tau_list, "s--", color=color_tau, label="Kendall τ (cos-lat wtd)")
    ax2.set_ylabel("Kendall τ", color=color_tau)
    ax2.tick_params(axis="y", labelcolor=color_tau)
    ax2.set_ylim(-0.1, 0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("ACE2 JJA heat-extreme skill vs lead time\n"
                  "1980–2016 · global cos-lat weighted", fontsize=10)
    ax1.set_xticks(lead_days)
    ax1.set_xticklabels([f"~{d}d" for d in lead_days])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Recompute cached thresholds")
    p.add_argument("--years", default="all", help="'all' or comma-separated years, e.g. 1980,1981")
    return p.parse_args()


def main():
    global YEARS
    args = parse_args()
    if args.years != "all":
        YEARS = [int(y) for y in args.years.split(",")]
        print(f"Running on years: {YEARS[0]}–{YEARS[-1]} ({len(YEARS)} years)", flush=True)
    for d in [CACHE_DIR, THRESH_DIR, FIGURES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    template  = get_template()
    lat       = template["lat"].values
    lon       = template["lon"].values
    print("Building land mask ...", flush=True)
    land_mask = build_land_mask(lat, lon)

    # ── pre-cache ERA5 monthly data (requester-free public zarr) ─────────
    print("Pre-caching ERA5 monthly data ...", flush=True)
    months_needed = set()
    for _, tm, td, _ in TARGETS:
        for year in YEARS:
            target = pd.Timestamp(year=year, month=tm, day=td)
            for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
                d = target + pd.Timedelta(days=delta)
                if YEARS[0] <= d.year <= YEARS[-1]:
                    months_needed.add((d.year, d.month))
    for y, m in tqdm(sorted(months_needed), desc="ERA5 month cache"):
        cache_era5_month(ERA5_ZARR, y, m, BBOX, template, CACHE_DIR)

    # ── per-target processing ─────────────────────────────────────────────
    all_tau_maps  = []
    all_pred_prob = []
    all_obs_ext   = []
    lead_days_list = []
    bss_per_lead   = []
    tau_per_lead   = []

    for label, tm, td, lead in TARGETS:
        print(f"\n{'='*60}\nTarget: {label} (lead ~{lead}d)\n{'='*60}", flush=True)

        era5_thresh = compute_era5_loo_thresholds(tm, td, template, force=args.force)
        ace2_thresh = compute_ace2_loo_thresholds(tm, td, template, force=args.force)

        pred_prob, obs_ext = compute_events(tm, td, era5_thresh, ace2_thresh, template)

        # Per-lead metrics
        bss_lead = brier_skill_score(pred_prob, obs_ext)
        tau_lead = kendall_tau_map(pred_prob, obs_ext)

        bss_scalar = cos_lat_mean(bss_lead, lat)
        tau_scalar = cos_lat_mean(tau_lead, lat)

        print(f"  {label}: BSS={bss_scalar:.3f}  τ={tau_scalar:.3f}", flush=True)

        lead_days_list.append(lead)
        bss_per_lead.append(bss_scalar)
        tau_per_lead.append(tau_scalar)

        # Per-lead tau and BSS maps
        plot_tau_map(
            tau_lead, lat, lon,
            f"|Kendall τ| — ACE2 JJA {label} | {YEARS[0]}–{YEARS[-1]}",
            FIGURES_DIR / f"tau_map_{label.lower()}.png",
            land_mask=land_mask,
        )
        plot_bss_map(
            bss_lead, lat, lon,
            f"|BSS| — ACE2 JJA {label} | {YEARS[0]}–{YEARS[-1]}",
            FIGURES_DIR / f"bss_map_{label.lower()}.png",
            land_mask=land_mask,
        )

        all_pred_prob.append(pred_prob)
        all_obs_ext.append(obs_ext)
        all_tau_maps.append(tau_lead)

        # Save pred_prob per target for sst_teleconnection_jja.py
        da_pp = xr.DataArray(
            pred_prob, dims=["year", "lat", "lon"],
            coords={"year": YEARS, "lat": lat, "lon": lon},
            attrs={"long_name": f"ACE2 predicted HHE probability {label}"},
        )
        write_atomic(da_pp.rename("pred_prob").to_dataset(),
                     METRICS_DIR / f"pred_prob_{label.lower()}.nc")

    # ── JJA pooled (all 3 target dates combined) ──────────────────────────
    print("\nComputing JJA-pooled metrics ...", flush=True)
    pred_jja = np.concatenate(all_pred_prob, axis=0)   # (111, lat, lon)
    obs_jja  = np.concatenate(all_obs_ext,   axis=0)

    bss_jja  = brier_skill_score(pred_jja, obs_jja)
    tau_jja  = kendall_tau_map(pred_jja, obs_jja)

    bss_jja_scalar = cos_lat_mean(bss_jja, lat)
    tau_jja_scalar = cos_lat_mean(tau_jja, lat)

    print(f"  JJA pooled: BSS={bss_jja_scalar:.3f}  τ={tau_jja_scalar:.3f}", flush=True)

    plot_tau_map(
        tau_jja, lat, lon,
        f"|Kendall τ| — ACE2 JJA pooled (Jun+Jul+Aug) | {YEARS[0]}–{YEARS[-1]}",
        FIGURES_DIR / "tau_map_jja.png",
        land_mask=land_mask,
    )
    plot_bss_map(
        bss_jja, lat, lon,
        f"|BSS| — ACE2 JJA pooled (Jun+Jul+Aug) | {YEARS[0]}–{YEARS[-1]}",
        FIGURES_DIR / "bss_map_jja.png",
        land_mask=land_mask,
    )

    # ── skill vs lead plot ────────────────────────────────────────────────
    plot_skill_vs_lead(lead_days_list, bss_per_lead, tau_per_lead,
                       FIGURES_DIR / "skill_vs_lead.png")

    # ── save scalar metrics ───────────────────────────────────────────────
    metrics = {
        "label":     [t[0] for t in TARGETS] + ["JJA"],
        "lead_days": lead_days_list + [None],
        "bss":       bss_per_lead + [bss_jja_scalar],
        "kendall_tau": tau_per_lead + [tau_jja_scalar],
    }
    import json
    (METRICS_DIR / "jja_skill_summary.json").write_text(json.dumps(metrics, indent=2))

    # Save tau maps as netCDF
    for (label, tm, td, lead), tau_map in zip(TARGETS, all_tau_maps):
        da = xr.DataArray(tau_map, dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon},
                          attrs={"lead_nominal_days": lead})
        write_atomic(da.rename("kendall_tau").to_dataset(),
                     METRICS_DIR / f"tau_map_{label.lower()}.nc")

    da_jja = xr.DataArray(tau_jja, dims=["lat", "lon"],
                          coords={"lat": lat, "lon": lon},
                          attrs={"n_events": 111})
    write_atomic(da_jja.rename("kendall_tau").to_dataset(),
                 METRICS_DIR / "tau_map_jja.nc")

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
