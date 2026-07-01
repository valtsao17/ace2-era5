#!/usr/bin/env python3
"""JJA seasonal heat-extreme skill map — seasonal-frequency, ±7-day window.

For each init year Y:
  - Threshold: 90th pct of a ±THRESH_WINDOW sliding window, pooled across all
    years, computed separately for ERA5 and ACE2.
  - ACE2 seasonal freq: fraction of (member × JJA day) pairs where Tmax > ACE2 threshold.
  - ERA5 seasonal freq: fraction of JJA days where ERA5 Tmax > ERA5 threshold.
  Pearson r and Kendall τ of those two time-series across years at each grid cell.

Outputs → outputs/lag_may/seasonal_jja_sliding7d/
"""

from __future__ import annotations

from pathlib import Path
import warnings

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

COMBINED_DIR = PROJECT_ROOT / "outputs/lag_may/combined_jja"
ERA5_CACHE   = PROJECT_ROOT / "outputs/lag_may/postprocess_jja/era5_cache"
FORCING_DIR  = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"

YEARS         = list(range(1980, 2017))   # 37 init years
N_MEMBERS     = 25
HOT_PCT       = 90.0
THRESH_WINDOW = 7   # ±7-day sliding window

# 92-day JJA calendar: Jun 1-30, Jul 1-31, Aug 1-31
JJA_SEQ: list[tuple[int, int]] = (
    [(6, d) for d in range(1, 31)]
  + [(7, d) for d in range(1, 32)]
  + [(8, d) for d in range(1, 32)]
)
_JJA_POS: dict[tuple[int, int], int] = {md: i for i, md in enumerate(JJA_SEQ)}

_SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"],
    N=256,
)
_SKILL_CMAP.set_bad("white")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight"})

_COAST_GEOMS = None


def _get_coast():
    global _COAST_GEOMS
    if _COAST_GEOMS is None:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        _COAST_GEOMS = list(shpreader.Reader(shp).geometries())
    return _COAST_GEOMS


def _draw_coast(ax):
    from shapely.geometry import box
    xl, xr_ = ax.get_xlim()
    yb, yt  = ax.get_ylim()
    vp = box(xl, yb, xr_, yt)

    def _plot(geom, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot(g, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=lw, zorder=3)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color="black", linewidth=lw, zorder=3)

    for geom in _get_coast():
        try:
            _plot(geom.intersection(vp), 0.4)
        except Exception:
            continue


# ── loaders ───────────────────────────────────────────────────────────────

def load_era5_tmax_month(year: int, month: int) -> np.ndarray:
    path = ERA5_CACHE / f"era5_daily_tmax_C_y{year}_m{month:02d}_on_grid.nc"
    with xr.open_dataset(path) as ds:
        return ds["era5_daily_tmax_C"].values.astype(np.float32)


def _load_all_era5_jja(nlat: int, nlon: int) -> np.ndarray:
    """(n_years, 92, nlat, nlon) float32 ERA5 daily Tmax in °C."""
    arr = np.full((len(YEARS), 92, nlat, nlon), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ERA5 load")):
        for month in (6, 7, 8):
            data = load_era5_tmax_month(year, month)    # (n_days, nlat, nlon)
            for d in range(data.shape[0]):
                pos = _JJA_POS.get((month, d + 1))
                if pos is None:
                    continue
                arr[y_idx, pos] = data[d]
    return arr


def _load_all_ace2_jja(nlat: int, nlon: int) -> np.ndarray:
    """(n_years, n_members, 92, nlat, nlon) float32 ACE2 daily Tmax in °C."""
    arr = np.full((len(YEARS), N_MEMBERS, 92, nlat, nlon), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ACE2 load")):
        path = COMBINED_DIR / f"tmax_jja_{year}.nc"
        with xr.open_dataset(path) as ds:
            da    = (ds["TMP2m"] - 273.15).values.astype(np.float32)  # (member, time, lat, lon)
            times = pd.DatetimeIndex(ds["time"].values)
        for t_idx, ts in enumerate(times):
            pos = _JJA_POS.get((ts.month, ts.day))
            if pos is None:
                continue
            arr[y_idx, :, pos, :, :] = da[:, t_idx, :, :]
    return arr


# ── Sliding-window thresholds ─────────────────────────────────────────────

def _thresh_one_day(day_data: np.ndarray, pct: float) -> np.ndarray:
    """Full-climatology pct threshold for one JJA day position (no LOYO).

    A single threshold is computed from ALL years' samples and applied to every
    year. Leave-one-year-out was removed per methodology: the predictand is a
    seasonal frequency scored against a fixed climatological threshold.

    day_data: (n_years, n_samp_per_year, nlat, nlon)
    Returns  : (n_years, nlat, nlon)  (one threshold broadcast over years)
    """
    n_years, n_samp, nlat, nlon = day_data.shape
    pool = day_data.reshape(-1, nlat, nlon)          # all years pooled (no leave-out)
    # Missing member/day samples are possible when older May rollouts do not
    # cover the full JJA season. Ignore them when defining the percentile
    # threshold; otherwise missing values can turn the top-decile frequency into
    # an artifact of rollout length.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        thr = np.nanpercentile(pool, pct, axis=0).astype(np.float32)
    return np.broadcast_to(thr, (n_years, nlat, nlon)).astype(np.float32)


def compute_daywise_thresholds(all_data: np.ndarray,
                               window: int = THRESH_WINDOW,
                               pct: float = HOT_PCT) -> np.ndarray:
    """(n_years, 92, nlat, nlon) full-climatology thresholds via ±window pool.

    No LOYO: each day position's threshold is the pct of the ±window pool over
    ALL years, applied to every year (broadcast). all_data: (n_years,
    [n_members,] 92, nlat, nlon).
    """
    n_years = all_data.shape[0]
    n_days  = all_data.shape[-3]
    nlat    = all_data.shape[-2]
    nlon    = all_data.shape[-1]

    # Reshape to (n_years, n_samp_per_year, n_days, nlat, nlon)
    flat = all_data.reshape(n_years, -1, n_days, nlat, nlon)

    def _one_day(d):
        d0 = max(0, d - window)
        d1 = min(n_days, d + window + 1)
        # (n_years, n_samp*(d1-d0), nlat, nlon) — copy to make contiguous before masking
        day_data = np.ascontiguousarray(
            flat[:, :, d0:d1, :, :]
        ).reshape(n_years, -1, nlat, nlon)
        return _thresh_one_day(day_data, pct)   # (n_years, nlat, nlon)

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_day)(d) for d in range(n_days)
    )
    # results[d]: (n_years, nlat, nlon)  →  stack along day axis
    return np.stack(results, axis=1)   # (n_years, n_days, nlat, nlon)


# Back-compat alias: behavior is now full-climatology (no LOYO) despite the name.
compute_daywise_thresholds_loo = compute_daywise_thresholds


def exceedance_frequency(samples: np.ndarray, thresh: np.ndarray,
                         axes: tuple[int, ...] | int) -> np.ndarray:
    """Fraction of finite samples exceeding a finite threshold."""
    valid = np.isfinite(samples) & np.isfinite(thresh)
    hits = (samples > thresh) & valid
    denom = valid.sum(axis=axes)
    num = hits.sum(axis=axes)
    out = np.full(denom.shape, np.nan, dtype=np.float32)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


# ── correlation maps ──────────────────────────────────────────────────────

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


def cos_lat_mean(field: np.ndarray, lat: np.ndarray,
                 mask: np.ndarray | None = None) -> float:
    w     = np.cos(np.deg2rad(lat))[:, np.newaxis]
    f     = field if mask is None else np.where(mask, field, np.nan)
    valid = np.isfinite(f)
    return float(np.nansum(f * w * valid) / np.nansum(w * valid))


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


def domain_scores_label(title: str, field: np.ndarray, lat: np.ndarray,
                        land_mask: np.ndarray | None, fmt: str = "{:.3f}") -> str:
    """Single-line corner-annotation string: the cos-lat-weighted score over the
    whole domain, with land-only and ocean-only sub-domain values appended in a
    parenthetical, e.g.:

        mean τ = 0.482  (land 0.515, sea 0.447)

    Keeps the original compact one-line legend look. Land/ocean means honour any
    NaNs already in `field` (e.g. HHE occurrence masking), since cos_lat_mean
    ignores non-finite cells.
    """
    a = cos_lat_mean(field, lat)
    if land_mask is None:
        return f"{title} = {fmt.format(a)}"
    lnd = cos_lat_mean(field, lat, mask=land_mask)
    sea = cos_lat_mean(field, lat, mask=~land_mask)
    return f"{title} = {fmt.format(a)}  (land {fmt.format(lnd)}, sea {fmt.format(sea)})"


# ── plotting ───────────────────────────────────────────────────────────────

def _roll_to_180(field, lon):
    """Shift 0-360 grid to -180/180."""
    split = np.searchsorted(lon, 180.0)
    lon_r = np.concatenate([lon[split:] - 360.0, lon[:split]])
    n = len(lon)
    field_r = np.roll(field, n - split, axis=-1)
    return field_r, lon_r


def plot_global(field: np.ndarray, pval: np.ndarray,
                lat: np.ndarray, lon: np.ndarray,
                title: str, out_path: Path, metric_label: str,
                land_mask: np.ndarray | None = None, signed: bool = False):
    """signed=True plots the raw (signed) field on a blue-white-red diverging
    scale instead of |field| on the sequential white-red scale — used for
    rank correlation (tau), where sign is informative and shouldn't be
    discarded."""
    plot_field   = field if signed else np.abs(field)
    plot_r, lon_r = _roll_to_180(plot_field, lon)
    pval_r, _    = _roll_to_180(pval, lon)
    field_r, _   = _roll_to_180(field, lon)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#d0e8f0")

    LON2D, LAT2D = np.meshgrid(lon_r, lat)
    if signed:
        vmax = 1.0
        cmap, vmin = "RdBu_r", -vmax
    else:
        cmap, vmin, vmax = _SKILL_CMAP, 0.0, 0.6
    mesh = ax.pcolormesh(LON2D, LAT2D, plot_r,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="nearest", zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    _draw_coast(ax)

    not_sig   = (pval_r > 0.05) & np.isfinite(field_r)
    n_stipple = int(not_sig.sum())
    if n_stipple > 40000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_stipple, size=40000, replace=False)
        xs = LON2D[not_sig].ravel()[idx]
        ys = LAT2D[not_sig].ravel()[idx]
    else:
        xs = LON2D[not_sig].ravel()
        ys = LAT2D[not_sig].ravel()
    ax.plot(xs, ys, "k.", markersize=0.8, alpha=0.4, zorder=4, linewidth=0)

    label = metric_label if signed else f"|{metric_label}|"
    fig.colorbar(mesh, ax=ax, shrink=0.7, label=label)
    full_mean = cos_lat_mean(np.abs(field), lat)
    if land_mask is not None:
        land_mean  = cos_lat_mean(np.abs(field), lat, mask=land_mask)
        ocean_mean = cos_lat_mean(np.abs(field), lat, mask=~land_mask)
        txt = (f"cos-lat mean |{metric_label}|   "
              f"full={full_mean:.3f}   land={land_mean:.3f}   ocean={ocean_mean:.3f}")
    else:
        txt = f"cos-lat mean |{metric_label}| = {full_mean:.3f}"
    ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

    ax.set_xticks(range(-180, 181, 60))
    ax.set_xticklabels(["180°", "120°W", "60°W", "0°", "60°E", "120°E", "180°"], fontsize=8)
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--replot", action="store_true",
                   help="Skip recomputation; reload skill_jja_seasonal.nc and redraw plots only")
    args = p.parse_args()

    default_out = PROJECT_ROOT / f"outputs/lag_may/seasonal_jja_sliding{THRESH_WINDOW}d"
    out_dir = Path(args.out_dir) if args.out_dir else default_out

    if args.replot:
        print(f"--replot: loading {out_dir / 'skill_jja_seasonal.nc'}", flush=True)
        with xr.open_dataset(out_dir / "skill_jja_seasonal.nc") as ds:
            lat      = ds["lat"].values
            lon      = ds["lon"].values
            r_map    = ds["pearson_r"].values
            r_pval   = ds["r_p_value"].values
            tau_map  = ds["kendall_tau"].values
            tau_pval = ds["tau_p_value"].values

        global_r   = cos_lat_mean(np.abs(r_map), lat)
        global_tau = cos_lat_mean(np.abs(tau_map), lat)
        print(f"Global cos-lat |r|   = {global_r:.3f}", flush=True)
        print(f"Global cos-lat |τ|   = {global_tau:.3f}", flush=True)

        print("Loading land mask ...", flush=True)
        land_mask = load_land_mask(lat, lon)
        if land_mask is None:
            print("WARNING: no forcing file found — land/ocean split unavailable", flush=True)

        yr_range     = f"{YEARS[0]}–{YEARS[-1]}  (n={len(YEARS)} seasons)"
        thresh_label = f"±{THRESH_WINDOW}-day sliding window thresholds"

        plot_global(
            r_map, r_pval, lat, lon,
            f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Pearson r\n"
            f"{yr_range}  |  {thresh_label}",
            out_dir / "pearsonr_jja_seasonal_global.png",
            metric_label="Pearson r",
            land_mask=land_mask,
        )
        plot_global(
            tau_map, tau_pval, lat, lon,
            f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Kendall τ\n"
            f"{yr_range}  |  {thresh_label}",
            out_dir / "tau_jja_seasonal_global.png",
            metric_label="Kendall τ",
            land_mask=land_mask,
            signed=True,
        )
        print("All done.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(COMBINED_DIR / f"tmax_jja_{YEARS[0]}.nc") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
    nlat, nlon = len(lat), len(lon)
    print(f"Grid: {nlat} lat × {nlon} lon (global)", flush=True)

    missing = [(yr, m) for yr in YEARS for m in (6, 7, 8)
               if not (ERA5_CACHE / f"era5_daily_tmax_C_y{yr}_m{m:02d}_on_grid.nc").exists()]
    if missing:
        print(f"WARNING: {len(missing)} ERA5 cache files missing — run postprocess_jja_lag.py first",
              flush=True)
        return
    print("All ERA5 Tmax months cached.", flush=True)

    print("\nLoading all ERA5 JJA data ...", flush=True)
    era5_all = _load_all_era5_jja(nlat, nlon)
    print(f"  era5_all: {era5_all.shape}  ({era5_all.nbytes/1e6:.0f} MB)", flush=True)

    print("Loading all ACE2 JJA data ...", flush=True)
    ace2_all = _load_all_ace2_jja(nlat, nlon)
    print(f"  ace2_all: {ace2_all.shape}  ({ace2_all.nbytes/1e9:.1f} GB)", flush=True)

    print(f"\nComputing ERA5 ±{THRESH_WINDOW}-day climatological thresholds (no LOYO) ...", flush=True)
    era5_thresh = compute_daywise_thresholds(era5_all)   # (n_years, 92, nlat, nlon)
    print("  ERA5 done.", flush=True)

    print(f"Computing ACE2 ±{THRESH_WINDOW}-day climatological thresholds (no LOYO) ...", flush=True)
    ace2_thresh = compute_daywise_thresholds(ace2_all)
    print("  ACE2 done.", flush=True)

    # Vectorised seasonal frequency against fixed full-climatology thresholds.
    print("\nComputing seasonal frequencies ...", flush=True)
    ace2_freqs_list, era5_freqs_list = [], []
    for i, yr in enumerate(YEARS):
        # ace2_all[i]: (25, 92, nlat, nlon),  ace2_thresh[i]: (92, nlat, nlon)
        af = exceedance_frequency(ace2_all[i], ace2_thresh[i][np.newaxis], axes=(0, 1))
        ef = exceedance_frequency(era5_all[i], era5_thresh[i], axes=0)
        ace2_freqs_list.append(af)
        era5_freqs_list.append(ef)
        print(f"  {yr}: ACE2={np.nanmean(af):.3f}  ERA5={np.nanmean(ef):.3f}", flush=True)
    ace2_freqs = np.stack(ace2_freqs_list, axis=0)   # (n_years, nlat, nlon)
    era5_freqs = np.stack(era5_freqs_list, axis=0)

    print("\nComputing Pearson r ...", flush=True)
    r_map, r_pval = pearson_r_map(ace2_freqs, era5_freqs)

    print("Computing Kendall τ ...", flush=True)
    tau_map, tau_pval = kendall_tau_map(ace2_freqs, era5_freqs)

    # Save
    da_r   = xr.DataArray(r_map,   dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    da_rp  = xr.DataArray(r_pval,  dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    da_tau = xr.DataArray(tau_map,  dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    da_tp  = xr.DataArray(tau_pval, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    xr.Dataset({"pearson_r": da_r, "r_p_value": da_rp,
                "kendall_tau": da_tau, "tau_p_value": da_tp}).to_netcdf(
        out_dir / "skill_jja_seasonal.nc")

    # Save seasonal frequency arrays for downstream SST / cluster scripts
    da_af = xr.DataArray(ace2_freqs, dims=["year", "lat", "lon"],
                         coords={"year": YEARS, "lat": lat, "lon": lon},
                         attrs={"long_name": "ACE2 JJA raw TMP2m heat-extreme seasonal frequency (no-LOYO +/-7d)"})
    da_ef = xr.DataArray(era5_freqs, dims=["year", "lat", "lon"],
                         coords={"year": YEARS, "lat": lat, "lon": lon},
                         attrs={"long_name": "ERA5 JJA raw TMP2m heat-extreme seasonal frequency (no-LOYO +/-7d)"})
    xr.Dataset({"ace2_freq": da_af, "era5_freq": da_ef}).to_netcdf(
        out_dir / "jja_seasonal_freqs.nc")
    print(f"wrote: {out_dir / 'jja_seasonal_freqs.nc'}", flush=True)

    global_r   = cos_lat_mean(np.abs(r_map), lat)
    global_tau = cos_lat_mean(np.abs(tau_map), lat)
    print(f"\nGlobal cos-lat |r|   = {global_r:.3f}", flush=True)
    print(f"Global cos-lat |τ|   = {global_tau:.3f}", flush=True)

    print("Loading land mask ...", flush=True)
    land_mask = load_land_mask(lat, lon)
    if land_mask is None:
        print("WARNING: no forcing file found — land/ocean split unavailable", flush=True)

    yr_range     = f"{YEARS[0]}–{YEARS[-1]}  (n={len(YEARS)} seasons)"
    thresh_label = f"±{THRESH_WINDOW}-day sliding window thresholds"

    plot_global(
        r_map, r_pval, lat, lon,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Pearson r\n"
        f"{yr_range}  |  {thresh_label}",
        out_dir / "pearsonr_jja_seasonal_global.png",
        metric_label="Pearson r",
        land_mask=land_mask,
    )
    plot_global(
        tau_map, tau_pval, lat, lon,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Kendall τ\n"
        f"{yr_range}  |  {thresh_label}",
        out_dir / "tau_jja_seasonal_global.png",
        metric_label="Kendall τ",
        land_mask=land_mask,
        signed=True,
    )

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
