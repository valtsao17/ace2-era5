#!/usr/bin/env python3
"""DJF seasonal cold-extreme skill map (Pearson r), CONUS only.

For each init year: fraction of (member × DJF day) pairs where ACE2 Tmin < day-specific
10th-pct threshold (±7-day sliding window across all years). Same for ERA5. Pearson r
of those two time-series at each grid cell.

All heavy computation is restricted to the CONUS bounding box; thresholds use
np.partition (O(N)) rather than a full sort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import kendalltau, t as scipy_t
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMBINED_DIR = PROJECT_ROOT / "outputs/lag_nov/combined_djf"
ERA5_CACHE   = PROJECT_ROOT / "outputs/lag_nov/postprocess_djf/era5_cache"
FORCING_DIR  = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"

YEARS         = list(range(1981, 2016))
N_MEMBERS     = 25
COLD_PCT      = 10.0
THRESH_WINDOW = 7

DJF_SEQ: list[tuple[int, int]] = (
    [(12, d) for d in range(1, 32)]
  + [(1,  d) for d in range(1, 32)]
  + [(2,  d) for d in range(1, 29)]
)
_DJF_POS: dict[tuple[int, int], int] = {md: i for i, md in enumerate(DJF_SEQ)}
_DJF_POS[(2, 29)] = 89   # leap-year Feb 29 → same pool as Feb 28

CONUS_EXTENT = (-175.0, -50.0, 12.0, 77.0)   # lon_min, lon_max, lat_min, lat_max

_SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"],
    N=256,
)
_SKILL_CMAP.set_bad("white")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight"})


def abs_djf_months(init_year: int) -> list[tuple[int, int]]:
    return [(init_year, 12), (init_year + 1, 1), (init_year + 1, 2)]


# ── load helpers ──────────────────────────────────────────────────────────

def load_ace2_tmin(init_year: int) -> xr.DataArray:
    ds = xr.open_dataset(COMBINED_DIR / f"tmin_djf_{init_year}.nc")
    da = (ds["TMP2m"] - 273.15).astype(np.float32)
    ds.close()
    return da


def _fetch_era5_tmin_month(year: int, month: int, lat: np.ndarray, lon: np.ndarray):
    import gcsfs, zarr as zarr_lib
    ZARR_URL = ("gs://ai2cm-public-requester-pays/"
                "2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr")
    ERA5_REF = pd.Timestamp("1940-01-01T12:00:00")

    fs    = gcsfs.GCSFileSystem(project="ace2-ic-download", requester_pays=True,
                                token="google_default")
    store = zarr_lib.open(fs.get_mapper(ZARR_URL), mode="r")

    tmin_cands = ["mn2t", "mn2t24", "minimum_2m_air_temperature", "MN2m"]
    tmp_var    = next((v for v in ["TMP2m", "t2m", "2m_temperature"] if v in store), None)
    tmin_var   = next((v for v in tmin_cands if v in store), None)

    import calendar
    n_days = calendar.monthrange(year, month)[1]
    dates  = pd.date_range(f"{year}-{month:02d}-01",
                           f"{year}-{month:02d}-{n_days:02d}", freq="D")
    zarr_times = store["time"][:]
    zarr_dt    = ERA5_REF + pd.to_timedelta(zarr_times, unit="h")

    out_path   = ERA5_CACHE / f"era5_tmin_{year}{month:02d}.nc"
    daily_tmin = np.full((n_days, len(lat), len(lon)), np.nan, dtype=np.float32)

    for d_idx, date in enumerate(dates):
        day_mask = (zarr_dt >= date) & (zarr_dt < date + pd.Timedelta("1D"))
        tidxs    = np.where(day_mask)[0]
        if len(tidxs) == 0:
            continue
        if tmin_var:
            vals = store[tmin_var][tidxs, :, :].min(axis=0)
        elif tmp_var:
            vals = store[tmp_var][tidxs, :, :].min(axis=0)
        else:
            raise RuntimeError("Cannot find temperature variable in ERA5 zarr")
        daily_tmin[d_idx] = vals - 273.15

    da = xr.DataArray(daily_tmin, dims=["time", "lat", "lon"],
                      coords={"time": dates, "lat": lat, "lon": lon},
                      attrs={"units": "degC", "long_name": "ERA5 daily Tmin"})
    da.to_dataset(name="era5_daily_tmin_C").to_netcdf(out_path)
    print(f"  cached {out_path.name}", flush=True)


def ensure_era5_tmin_cached(lat: np.ndarray, lon: np.ndarray):
    needed = set()
    for year in YEARS:
        for cal_year, month in abs_djf_months(year):
            needed.add((cal_year, month))
    missing = [(y, m) for y, m in sorted(needed)
               if not (ERA5_CACHE / f"era5_tmin_{y}{m:02d}.nc").exists()]
    if not missing:
        print("All ERA5 Tmin months already cached.", flush=True)
        return
    print(f"Fetching {len(missing)} missing ERA5 Tmin months ...", flush=True)
    for y, m in tqdm(missing, desc="ERA5 Tmin cache"):
        _fetch_era5_tmin_month(y, m, lat, lon)


def load_era5_tmin_month(year: int, month: int) -> np.ndarray:
    path = ERA5_CACHE / f"era5_tmin_{year}{month:02d}.nc"
    with xr.open_dataset(path) as ds:
        return ds["era5_daily_tmin_C"].values.astype(np.float32)


# ── CONUS-subset loaders ───────────────────────────────────────────────────

def _load_all_era5_djf(lat_sel: np.ndarray, lon_sel: np.ndarray) -> np.ndarray:
    """(n_years, 90, nlat_c, nlon_c) — only CONUS pixels."""
    nlat_c, nlon_c = int(lat_sel.sum()), int(lon_sel.sum())
    arr = np.full((len(YEARS), 90, nlat_c, nlon_c), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ERA5 load")):
        for cal_year, month in abs_djf_months(year):
            data   = load_era5_tmin_month(cal_year, month)    # (n_days, nlat, nlon)
            data_c = data[:, lat_sel, :][:, :, lon_sel]       # (n_days, nlat_c, nlon_c)
            for d in range(data_c.shape[0]):
                pos = _DJF_POS.get((month, d + 1))
                if pos is None:
                    continue
                if np.isnan(arr[y_idx, pos]).all():
                    arr[y_idx, pos] = data_c[d]
                else:
                    arr[y_idx, pos] = np.fmin(arr[y_idx, pos], data_c[d])
    return arr


def _load_all_ace2_djf(lat_sel: np.ndarray, lon_sel: np.ndarray) -> np.ndarray:
    """(n_years, n_members, 90, nlat_c, nlon_c) — only CONUS pixels."""
    nlat_c, nlon_c = int(lat_sel.sum()), int(lon_sel.sum())
    arr = np.full((len(YEARS), N_MEMBERS, 90, nlat_c, nlon_c), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ACE2 load")):
        da    = load_ace2_tmin(year)                               # (member, time, lat, lon)
        times = pd.DatetimeIndex(da.time.values)
        vals  = da.values[:, :, lat_sel, :][:, :, :, lon_sel]     # (member, time, nlat_c, nlon_c)
        da.close()
        for t_idx, ts in enumerate(times):
            pos = _DJF_POS.get((ts.month, ts.day))
            if pos is None:
                continue
            arr[y_idx, :, pos, :, :] = vals[:, t_idx, :, :]
    return arr


# ── thresholds ────────────────────────────────────────────────────────────

def _thresh_one_day(flat: np.ndarray, d: int, n_days: int, window: int) -> np.ndarray:
    d0   = max(0, d - window)
    d1   = min(n_days, d + window + 1)
    pool = flat[:, d0:d1, :, :].reshape(-1, flat.shape[-2], flat.shape[-1])
    k    = max(0, int(np.floor(COLD_PCT / 100.0 * pool.shape[0])))
    return np.partition(pool, k, axis=0)[k].astype(np.float32)


def compute_daywise_thresholds(all_data: np.ndarray, window: int = THRESH_WINDOW) -> np.ndarray:
    """(90, nlat_c, nlon_c) — 10th-pct threshold per DJF day via ±window sliding pool."""
    n_days = all_data.shape[-3]
    flat   = all_data.reshape(-1, n_days, all_data.shape[-2], all_data.shape[-1])
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_thresh_one_day)(flat, d, n_days, window)
        for d in range(n_days)
    )
    return np.stack(results, axis=0)


# ── correlation ────────────────────────────────────────────────────────────

def _tau_one_row(pred_row: np.ndarray, obs_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nlon     = pred_row.shape[1]
    tau_row  = np.full(nlon, np.nan, dtype=np.float32)
    pval_row = np.full(nlon, np.nan, dtype=np.float32)
    for j in range(nlon):
        p = pred_row[:, j]; o = obs_row[:, j]
        if np.isfinite(p).all() and np.isfinite(o).all() and o.std() > 0 and p.std() > 0:
            t, pv = kendalltau(p, o)
            tau_row[j] = t; pval_row[j] = pv
    return tau_row, pval_row


def kendall_tau_map(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parallelised-by-lat-row Kendall τ. pred/obs: (n_years, nlat, nlon)."""
    n_lat   = pred.shape[1]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_tau_one_row)(pred[:, i, :], obs[:, i, :])
        for i in range(n_lat)
    )
    return (np.stack([r[0] for r in results], axis=0),
            np.stack([r[1] for r in results], axis=0))


def pearson_r_map(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Pearson r + two-tailed p-value. pred/obs: (n_years, nlat, nlon)."""
    n      = pred.shape[0]
    pred_m = pred - pred.mean(axis=0, keepdims=True)
    obs_m  = obs  - obs.mean(axis=0, keepdims=True)
    denom  = np.sqrt((pred_m**2).sum(axis=0)) * np.sqrt((obs_m**2).sum(axis=0))
    denom  = np.where(denom > 0, denom, np.nan)
    r      = (pred_m * obs_m).sum(axis=0) / denom
    t_stat = r * np.sqrt((n - 2) / np.maximum(1.0 - r**2, 1e-15))
    p_val  = (2 * scipy_t.sf(np.abs(t_stat), df=n - 2)).astype(np.float32)
    return r.astype(np.float32), p_val


def cos_lat_mean(field: np.ndarray, lat: np.ndarray,
                 mask: np.ndarray | None = None) -> float:
    w     = np.cos(np.deg2rad(lat))[:, np.newaxis]
    f     = field if mask is None else np.where(mask, field, np.nan)
    valid = np.isfinite(f)
    return float(np.nansum(f * w * valid) / np.nansum(w * valid))


# ── land mask & borders ────────────────────────────────────────────────────

def load_land_mask(lat: np.ndarray, lon_360: np.ndarray) -> np.ndarray | None:
    f = next(FORCING_DIR.glob("forcing_*.nc"), None)
    if f is None:
        return None
    with xr.open_dataset(f) as ds:
        lf    = ds["land_fraction"]
        lat_n = "latitude" if "latitude" in lf.dims else "lat"
        lon_n = "longitude" if "longitude" in lf.dims else "lon"
        lf2   = lf.interp({lat_n: lat, lon_n: lon_360}, method="nearest")
    return lf2.values > 0.5


_LAND_GEOMS  = None
_STATE_GEOMS = None


def _get_borders():
    global _LAND_GEOMS, _STATE_GEOMS
    if _LAND_GEOMS is None:
        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
        _STATE_GEOMS = list(shpreader.Reader(shp).geometries())
    return _LAND_GEOMS, _STATE_GEOMS


def _draw_borders(ax, xlim, ylim):
    from shapely.geometry import box
    vp = box(xlim[0], ylim[0], xlim[1], ylim[1])
    land_geoms, state_geoms = _get_borders()

    def _plot(geom, color, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot(g, color, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)

    for geom in land_geoms:
        try:
            _plot(geom.intersection(vp), "black", 0.5)
        except Exception:
            continue
    for geom in state_geoms:
        try:
            _plot(geom.intersection(vp), "0.4", 0.3)
        except Exception:
            continue


# ── figure ─────────────────────────────────────────────────────────────────

def plot_conus(r: np.ndarray, pval: np.ndarray,
               lat_c: np.ndarray, lon_c: np.ndarray,
               title: str, out_path: Path, land_mask_c: np.ndarray):
    """lat_c in °N, lon_c in °E (-180/180), both already CONUS-only."""
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    signed = "Kendall" in title

    stat_data = r if signed else np.abs(r)
    plot_data = np.where(land_mask_c, stat_data, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white")

    extent = [lon_c.min() - 0.5, lon_c.max() + 0.5,
              lat_c.min() - 0.5, lat_c.max() + 0.5]
    cmap = "RdBu_r" if signed else _SKILL_CMAP
    vmin, vmax = (-1.0, 1.0) if signed else (0.0, 0.6)
    im = ax.imshow(plot_data, extent=extent, origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest", zorder=1)

    _draw_borders(ax, (lon_min - 2, lon_max + 2), (lat_min - 2, lat_max + 2))

    LON2D, LAT2D = np.meshgrid(lon_c, lat_c)
    not_sig = (pval > 0.05) & land_mask_c & np.isfinite(r)
    ax.plot(LON2D[not_sig].ravel(), LAT2D[not_sig].ravel(),
            "k.", markersize=1.5, alpha=0.55, zorder=4)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    lon_ticks = np.arange(-160, -40, 20)
    lat_ticks = np.arange(20, 80, 10)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels([f"{abs(x)}°W" for x in lon_ticks], fontsize=8)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels([f"{y}°N" for y in lat_ticks], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    wavg = cos_lat_mean(stat_data, lat_c, mask=land_mask_c & np.isfinite(r))
    ax.text(0.98, 0.04, f"{wavg:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3), zorder=5)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.65, pad=0.07, aspect=30)
    metric_label = "Kendall τ" if "Kendall" in title else "Pearson r"
    cbar.set_label(metric_label, fontsize=9)
    ticks = np.linspace(-1.0, 1.0, 9) if signed else np.arange(0, 0.61, 0.06)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=7)

    ax.set_title(title, fontsize=10, pad=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    default_out = PROJECT_ROOT / f"outputs/lag_nov/seasonal_djf_sliding{THRESH_WINDOW}d"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(COMBINED_DIR / f"tmin_djf_{YEARS[0]}.nc") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values

    # CONUS bounding-box index masks — all heavy work stays inside this region
    lon_deg180 = np.where(lon > 180, lon - 360.0, lon)
    lat_sel    = (lat >= CONUS_EXTENT[2]) & (lat <= CONUS_EXTENT[3])
    lon_sel    = (lon_deg180 >= CONUS_EXTENT[0]) & (lon_deg180 <= CONUS_EXTENT[1])
    lat_c      = lat[lat_sel]
    lon_c      = lon_deg180[lon_sel]
    print(f"CONUS grid: {lat_c.size} lat × {lon_c.size} lon  (full grid: {lat.size}×{lon.size})",
          flush=True)

    land_mask   = load_land_mask(lat, lon)
    land_mask_c = (land_mask[lat_sel, :][:, lon_sel]
                   if land_mask is not None
                   else np.ones((lat_c.size, lon_c.size), dtype=bool))

    ERA5_CACHE.mkdir(parents=True, exist_ok=True)
    ensure_era5_tmin_cached(lat, lon)

    print("\nLoading all ERA5 DJF data (CONUS only) ...", flush=True)
    era5_all = _load_all_era5_djf(lat_sel, lon_sel)
    print(f"  era5_all: {era5_all.shape}  ({era5_all.nbytes/1e6:.0f} MB)", flush=True)

    print("Loading all ACE2 DJF data (CONUS only) ...", flush=True)
    ace2_all = _load_all_ace2_djf(lat_sel, lon_sel)
    print(f"  ace2_all: {ace2_all.shape}  ({ace2_all.nbytes/1e6:.0f} MB)", flush=True)

    print(f"\nComputing ±{THRESH_WINDOW}-day sliding thresholds ...", flush=True)
    era5_thresh = compute_daywise_thresholds(era5_all)   # (90, nlat_c, nlon_c)
    ace2_thresh = compute_daywise_thresholds(ace2_all)
    print("  done.", flush=True)

    # Vectorised frequency: no disk re-read, broadcast over member/day axes
    print("\nComputing seasonal frequencies ...", flush=True)
    ace2_freqs_list, era5_freqs_list = [], []
    for i, yr in enumerate(YEARS):
        af = (ace2_all[i] < ace2_thresh[np.newaxis]).astype(np.float32).mean(axis=(0, 1))
        ef = (era5_all[i] < era5_thresh).astype(np.float32).mean(axis=0)
        ace2_freqs_list.append(af)
        era5_freqs_list.append(ef)
        print(f"  {yr}: ACE2={af.mean():.3f}  ERA5={ef.mean():.3f}", flush=True)
    ace2_freqs = np.stack(ace2_freqs_list, axis=0)   # (n_years, nlat_c, nlon_c)
    era5_freqs = np.stack(era5_freqs_list, axis=0)

    print("\nComputing Pearson r ...", flush=True)
    r_map, r_pval = pearson_r_map(ace2_freqs, era5_freqs)

    print("Computing Kendall τ ...", flush=True)
    tau_map, tau_pval = kendall_tau_map(ace2_freqs, era5_freqs)

    da_r   = xr.DataArray(r_map,   dims=["lat", "lon"], coords={"lat": lat_c, "lon": lon_c})
    da_rp  = xr.DataArray(r_pval,  dims=["lat", "lon"], coords={"lat": lat_c, "lon": lon_c})
    da_tau = xr.DataArray(tau_map,  dims=["lat", "lon"], coords={"lat": lat_c, "lon": lon_c})
    da_tp  = xr.DataArray(tau_pval, dims=["lat", "lon"], coords={"lat": lat_c, "lon": lon_c})
    xr.Dataset({"pearson_r": da_r, "r_p_value": da_rp,
                "kendall_tau": da_tau, "tau_p_value": da_tp}).to_netcdf(
        out_dir / "skill_djf_cold_seasonal.nc")

    land_r   = cos_lat_mean(np.abs(r_map),  lat_c, mask=land_mask_c & np.isfinite(r_map))
    land_tau = cos_lat_mean(np.abs(tau_map), lat_c, mask=land_mask_c & np.isfinite(tau_map))
    print(f"\nLand-mean |r| = {land_r:.3f}", flush=True)
    print(f"Land-mean |τ| = {land_tau:.3f}", flush=True)

    yr_range     = f"{YEARS[0]}/{YEARS[0]+1}–{YEARS[-1]}/{YEARS[-1]+1}  (n={len(YEARS)} seasons)"
    thresh_label = f"±{THRESH_WINDOW}-day sliding window thresholds"

    plot_conus(r_map, r_pval, lat_c, lon_c,
               f"ACE2-ERA5  |  Skill in predicting winter (DJF) cold extreme frequency\n"
               f"Pearson r  |  {yr_range}  |  {thresh_label}",
               out_dir / "pearsonr_djf_cold_seasonal_conus.png", land_mask_c)

    plot_conus(tau_map, tau_pval, lat_c, lon_c,
               f"ACE2-ERA5  |  Skill in predicting winter (DJF) cold extreme frequency\n"
               f"Kendall τ  |  {yr_range}  |  {thresh_label}",
               out_dir / "tau_djf_cold_seasonal_conus.png", land_mask_c)

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
