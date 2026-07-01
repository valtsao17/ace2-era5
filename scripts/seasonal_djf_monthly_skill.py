#!/usr/bin/env python3
"""DJF monthly cold-extreme skill maps at 3 lead times (Dec/Jan/Feb).

Same methodology as seasonal_djf_skill.py but computes monthly cold-extreme
frequency SEPARATELY for Dec, Jan, Feb instead of aggregating all 90 DJF days.
Gives Pearson r AND Kendall τ maps for each month (~30 / ~60 / ~90 day lead
from the Nov-1 initialisation).

Thresholds: ±7-day sliding window, 10th pct, pooled across ALL 35 years (no LOO).
Separate thresholds for ACE2 and ERA5.

Outputs → outputs/lag_nov/seasonal_djf_monthly_sliding7d/
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

# 90-day DJF sequence
DJF_SEQ: list[tuple[int, int]] = (
    [(12, d) for d in range(1, 32)]
  + [(1,  d) for d in range(1, 32)]
  + [(2,  d) for d in range(1, 29)]
)
_DJF_POS: dict[tuple[int, int], int] = {md: i for i, md in enumerate(DJF_SEQ)}
_DJF_POS[(2, 29)] = 89

# Month windows within the 90-position DJF array and nominal lead from Nov 1
MONTHS = [
    ("Dec", 0,  31, "~30d lead"),   # Dec 1-31  → positions  0-30
    ("Jan", 31, 62, "~60d lead"),   # Jan 1-31  → positions 31-61
    ("Feb", 62, 90, "~90d lead"),   # Feb 1-28  → positions 62-89
]

CONUS_EXTENT = (-175.0, -50.0, 12.0, 77.0)

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


# ── loaders ───────────────────────────────────────────────────────────────

def load_era5_tmin_month(year: int, month: int) -> np.ndarray:
    path = ERA5_CACHE / f"era5_tmin_{year}{month:02d}.nc"
    with xr.open_dataset(path) as ds:
        return ds["era5_daily_tmin_C"].values.astype(np.float32)


def _load_all_era5_djf(lat_sel: np.ndarray, lon_sel: np.ndarray) -> np.ndarray:
    nlat_c, nlon_c = int(lat_sel.sum()), int(lon_sel.sum())
    arr = np.full((len(YEARS), 90, nlat_c, nlon_c), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ERA5 load")):
        for cal_year, month in abs_djf_months(year):
            data   = load_era5_tmin_month(cal_year, month)
            data_c = data[:, lat_sel, :][:, :, lon_sel]
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
    nlat_c, nlon_c = int(lat_sel.sum()), int(lon_sel.sum())
    arr = np.full((len(YEARS), N_MEMBERS, 90, nlat_c, nlon_c), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(YEARS, desc="ACE2 load")):
        ds    = xr.open_dataset(COMBINED_DIR / f"tmin_djf_{year}.nc")
        da    = (ds["TMP2m"] - 273.15).values.astype(np.float32)
        times = pd.DatetimeIndex(ds["time"].values)
        ds.close()
        vals  = da[:, :, lat_sel, :][:, :, :, lon_sel]
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


def compute_daywise_thresholds(all_data: np.ndarray,
                                window: int = THRESH_WINDOW) -> np.ndarray:
    """(90, nlat_c, nlon_c) 10th-pct threshold per DJF day."""
    n_days = all_data.shape[-3]
    flat   = all_data.reshape(-1, n_days, all_data.shape[-2], all_data.shape[-1])
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_thresh_one_day)(flat, d, n_days, window)
        for d in range(n_days)
    )
    return np.stack(results, axis=0)


# ── monthly frequency ─────────────────────────────────────────────────────

def monthly_freqs(all_data: np.ndarray, thresh: np.ndarray,
                  pos_start: int, pos_end: int) -> np.ndarray:
    """Cold-extreme frequency for one month across all years.

    all_data : (n_years, [n_members,] 90, nlat, nlon)
    thresh   : (90, nlat, nlon)
    Returns  : (n_years, nlat, nlon)
    """
    t_slice = thresh[pos_start:pos_end]   # (n_days_month, nlat, nlon)
    if all_data.ndim == 4:
        # ERA5: (n_years, 90, nlat, nlon)
        d_slice = all_data[:, pos_start:pos_end, :, :]    # (n_years, n_days, nlat, nlon)
        return (d_slice < t_slice[np.newaxis]).astype(np.float32).mean(axis=1)
    else:
        # ACE2: (n_years, n_members, 90, nlat, nlon)
        d_slice = all_data[:, :, pos_start:pos_end, :, :]  # (n_years, n_mem, n_days, nlat, nlon)
        return (d_slice < t_slice[np.newaxis, np.newaxis]).astype(np.float32).mean(axis=(1, 2))


# ── correlation maps ──────────────────────────────────────────────────────

def pearson_r_map(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        try: _plot(geom.intersection(vp), "black", 0.5)
        except Exception: continue
    for geom in state_geoms:
        try: _plot(geom.intersection(vp), "0.4", 0.3)
        except Exception: continue


# ── figure ─────────────────────────────────────────────────────────────────

def plot_conus(field: np.ndarray, pval: np.ndarray,
               lat_c: np.ndarray, lon_c: np.ndarray,
               title: str, out_path: Path,
               land_mask_c: np.ndarray, metric_label: str):
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    signed = "Kendall" in metric_label
    stat_data = field if signed else np.abs(field)
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
    not_sig = (pval > 0.05) & land_mask_c & np.isfinite(field)
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

    wavg = cos_lat_mean(stat_data, lat_c,
                        mask=land_mask_c & np.isfinite(field))
    ax.text(0.98, 0.04, f"{wavg:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3), zorder=5)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.65, pad=0.07, aspect=30)
    cbar.set_label(metric_label, fontsize=9)
    ticks = np.linspace(-1.0, 1.0, 9) if signed else np.arange(0, 0.61, 0.06)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=7)

    ax.set_title(title, fontsize=10, pad=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def plot_skill_vs_lead(month_labels: list[str], r_vals: list[float],
                       tau_vals: list[float], out_path: Path):
    x = np.arange(len(month_labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, r_vals,   "o-", color="#1f77b4", label="Pearson r")
    ax.plot(x, tau_vals, "s--", color="#d62728", label="Kendall τ")
    ax.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[0]} ({m[3]})" for m in MONTHS])
    ax.set_ylabel("Land-mean |skill|")
    ax.set_ylim(0, 0.6)
    ax.legend()
    ax.set_title("ACE2 DJF cold-extreme skill vs lead\n"
                 f"±{THRESH_WINDOW}-day sliding window thresholds  |  "
                 f"{YEARS[0]}–{YEARS[-1]}", fontsize=10)
    fig.tight_layout()
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

    default_out = PROJECT_ROOT / f"outputs/lag_nov/seasonal_djf_monthly_sliding{THRESH_WINDOW}d"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(COMBINED_DIR / f"tmin_djf_{YEARS[0]}.nc") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values

    lon_deg180 = np.where(lon > 180, lon - 360.0, lon)
    lat_sel    = (lat >= CONUS_EXTENT[2]) & (lat <= CONUS_EXTENT[3])
    lon_sel    = (lon_deg180 >= CONUS_EXTENT[0]) & (lon_deg180 <= CONUS_EXTENT[1])
    lat_c      = lat[lat_sel]
    lon_c      = lon_deg180[lon_sel]
    print(f"CONUS grid: {lat_c.size} lat × {lon_c.size} lon", flush=True)

    land_mask   = load_land_mask(lat, lon)
    land_mask_c = (land_mask[lat_sel, :][:, lon_sel]
                   if land_mask is not None
                   else np.ones((lat_c.size, lon_c.size), dtype=bool))

    print("\nLoading all ERA5 DJF data (CONUS) ...", flush=True)
    era5_all = _load_all_era5_djf(lat_sel, lon_sel)
    print(f"  {era5_all.shape}  ({era5_all.nbytes/1e6:.0f} MB)", flush=True)

    print("Loading all ACE2 DJF data (CONUS) ...", flush=True)
    ace2_all = _load_all_ace2_djf(lat_sel, lon_sel)
    print(f"  {ace2_all.shape}  ({ace2_all.nbytes/1e6:.0f} MB)", flush=True)

    print(f"\nComputing ±{THRESH_WINDOW}-day sliding thresholds ...", flush=True)
    era5_thresh = compute_daywise_thresholds(era5_all)
    ace2_thresh = compute_daywise_thresholds(ace2_all)
    print("  done.", flush=True)

    yr_range     = f"{YEARS[0]}/{YEARS[0]+1}–{YEARS[-1]}/{YEARS[-1]+1}  (n={len(YEARS)})"
    thresh_label = f"±{THRESH_WINDOW}-day sliding window"

    r_scores   = []
    tau_scores = []
    ds_vars    = {}

    for month_name, pos0, pos1, lead_label in MONTHS:
        print(f"\n── {month_name} ({lead_label}) ──", flush=True)

        ace2_freq = monthly_freqs(ace2_all, ace2_thresh, pos0, pos1)  # (n_years, nlat_c, nlon_c)
        era5_freq = monthly_freqs(era5_all, era5_thresh, pos0, pos1)

        for yr, af, ef in zip(YEARS, ace2_freq, era5_freq):
            print(f"  {yr}: ACE2={af.mean():.3f}  ERA5={ef.mean():.3f}", flush=True)

        r_map,   r_pval   = pearson_r_map(ace2_freq, era5_freq)
        tau_map, tau_pval = kendall_tau_map(ace2_freq, era5_freq)

        land_r   = cos_lat_mean(np.abs(r_map),   lat_c, mask=land_mask_c & np.isfinite(r_map))
        land_tau = cos_lat_mean(np.abs(tau_map),  lat_c, mask=land_mask_c & np.isfinite(tau_map))
        print(f"  land |r|={land_r:.3f}  land |τ|={land_tau:.3f}", flush=True)
        r_scores.append(land_r)
        tau_scores.append(land_tau)

        mn = month_name.lower()
        ds_vars[f"pearson_r_{mn}"]    = xr.DataArray(r_map,   dims=["lat","lon"],
                                                       coords={"lat":lat_c,"lon":lon_c})
        ds_vars[f"r_pval_{mn}"]       = xr.DataArray(r_pval,  dims=["lat","lon"],
                                                       coords={"lat":lat_c,"lon":lon_c})
        ds_vars[f"kendall_tau_{mn}"]  = xr.DataArray(tau_map,  dims=["lat","lon"],
                                                       coords={"lat":lat_c,"lon":lon_c})
        ds_vars[f"tau_pval_{mn}"]     = xr.DataArray(tau_pval, dims=["lat","lon"],
                                                       coords={"lat":lat_c,"lon":lon_c})

        plot_conus(
            r_map, r_pval, lat_c, lon_c,
            f"ACE2-ERA5  |  DJF cold-extreme skill  {month_name} ({lead_label})\n"
            f"Pearson r  |  {yr_range}  |  {thresh_label}",
            out_dir / f"pearsonr_djf_{mn}_conus.png",
            land_mask_c, "Pearson r",
        )
        plot_conus(
            tau_map, tau_pval, lat_c, lon_c,
            f"ACE2-ERA5  |  DJF cold-extreme skill  {month_name} ({lead_label})\n"
            f"Kendall τ  |  {yr_range}  |  {thresh_label}",
            out_dir / f"tau_djf_{mn}_conus.png",
            land_mask_c, "Kendall τ",
        )

    xr.Dataset(ds_vars).to_netcdf(out_dir / "skill_djf_monthly.nc")
    print(f"\nSaved: {out_dir / 'skill_djf_monthly.nc'}", flush=True)

    plot_skill_vs_lead(
        [m[0] for m in MONTHS], r_scores, tau_scores,
        out_dir / "skill_vs_lead.png",
    )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
