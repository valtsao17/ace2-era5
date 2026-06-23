#!/usr/bin/env python3
"""Compute skill maps over CONUS for the 10-year lag experiment.

Metrics computed at each grid cell across years:
  - Spearman rank correlation (predicted prob vs observed binary extreme)
  - Kendall tau rank correlation (same)
  - Brier score (mean squared error of probabilistic forecast)

For each of 6 combinations (3 leads × 2 extreme types):
  - ERA5 climatology (1940-2022) for the target month -> 90th/10th percentile threshold
  - For each year: predicted probability + observed binary extreme
  - Metrics computed across years at each CONUS grid cell

Area-weighted domain averages use cos(lat) weighting (spherical area element).

Outputs:
  outputs/lag_10yr/rank_corr/spearman_{heat|cold}_lead{030|060|090}d.nc
  outputs/lag_10yr/rank_corr/kendall_{heat|cold}_lead{030|060|090}d.nc
  outputs/lag_10yr/rank_corr/brier_{heat|cold}_lead{030|060|090}d.nc
  outputs/lag_10yr/figures/{rank_corr|kendall|brier}_CONUS_{heat|cold}_lead{030|060|090}d.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.io import subset_bbox

COMBINED_DIR = PROJECT_ROOT / "outputs/lag_10yr/combined"
OUT_ROOT     = PROJECT_ROOT / "outputs/lag_10yr"
RANK_DIR     = OUT_ROOT / "rank_corr"
FIG_DIR      = OUT_ROOT / "figures"
CACHE_DIR    = OUT_ROOT / "era5_cache_rankcorr"

for d in [RANK_DIR, FIG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

YEARS = list(range(1980, 1990))
LEADS = {"030d": 30, "060d": 60, "090d": 90}

# North America bounding box (lat: S, N; lon: W, E in 0-360)
# Covers Alaska, Canada, CONUS, Mexico — matching reference figure extent
CONUS_BBOX = (18.0, 72.0, 195.0, 305.0)

# Baseline years for climatological thresholds
BASELINE_YEARS = list(range(1940, 2023))

# Percentile thresholds
HEAT_PCT = 90
COLD_PCT = 10


# ── ERA5 helpers ──────────────────────────────────────────────────────────────

def target_date(year: int, lead_days: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, 11, 1)) + pd.Timedelta(days=lead_days)


def _fetch_era5_once(month: int, year: int, stat: str, template: xr.DataArray) -> xr.DataArray:
    """Single attempt to fetch and cache ERA5 daily data for one month/year."""
    import socket
    socket.setdefaulttimeout(120)   # 2-min socket timeout for GCS reads

    ds = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={})
    vname = next(v for v in ["2m_temperature", "t2m", "TMP2m"] if v in ds)
    start = pd.Timestamp(year=year, month=month, day=1)
    end   = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    da    = ds[vname].sel(time=slice(str(start), str(end)))

    lat_name = next(c for c in da.coords if c in ("latitude", "lat"))
    lon_name = next(c for c in da.coords if c in ("longitude", "lon"))
    tlat     = next(c for c in template.coords if c in ("lat", "latitude"))
    tlon     = next(c for c in template.coords if c in ("lon", "longitude"))
    da = da.sel({lat_name: template[tlat].values,
                 lon_name: template[tlon].values}, method="nearest")
    da = da.assign_coords({lat_name: template[tlat].values,
                           lon_name: template[tlon].values})

    if stat == "tmax":
        daily = (da - 273.15).resample(time="1D").max()
    else:
        daily = (da - 273.15).resample(time="1D").min()

    daily = daily.sel(time=daily.time.dt.month == month).astype("float32").load()
    daily = daily.rename({lat_name: "lat", lon_name: "lon"})
    daily.name = "TMP2m"
    ds.close()
    return daily


def cache_era5_daily(month: int, year: int, stat: str, template: xr.DataArray,
                     force: bool = False) -> xr.DataArray:
    """Cache ERA5 daily Tmax or Tmin for a single month/year on the template grid."""
    path = CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{year:04d}.nc"
    if path.exists() and not force:
        return xr.open_dataset(path)["TMP2m"]

    import time as _time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            daily = _fetch_era5_once(month, year, stat, template)
            daily.to_dataset().to_netcdf(path)
            return daily
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"    RETRY {year}-{month:02d} attempt {attempt+1}/{max_retries} ({e}) — wait {wait}s",
                  flush=True)
            _time.sleep(wait)

    raise RuntimeError(f"Failed to fetch ERA5 {stat} {year}-{month:02d} after {max_retries} retries")


def compute_threshold(month: int, stat: str, pct: float,
                      template: xr.DataArray) -> xr.DataArray:
    """Compute climatological percentile threshold from 1940-2022."""
    cache_path = CACHE_DIR / f"threshold_{stat}_m{month:02d}_pct{int(pct):02d}_1940_2022.nc"
    if cache_path.exists():
        return xr.open_dataset(cache_path)["threshold"]

    print(f"  Building {stat} {pct}th-pct threshold for month {month} ...", flush=True)
    arrays = []
    for year in BASELINE_YEARS:
        try:
            da = cache_era5_daily(month, year, stat, template)
            arrays.append(da)
        except Exception as e:
            print(f"    WARN {year}-{month:02d}: {e}", flush=True)

    all_days = xr.concat(arrays, dim="time").sortby("time")
    threshold = all_days.quantile(pct / 100.0, dim="time").drop_vars("quantile")
    threshold = threshold.rename("threshold").astype("float32")
    threshold.to_dataset().to_netcdf(cache_path)
    return threshold


def get_era5_observed(tgt: pd.Timestamp, stat: str, template: xr.DataArray) -> xr.DataArray:
    """Get ERA5 daily Tmax or Tmin on a specific date."""
    cache_path = CACHE_DIR / f"era5_obs_{stat}_{tgt.strftime('%Y%m%d')}.nc"
    if cache_path.exists():
        return xr.open_dataset(cache_path)["TMP2m"]
    da = cache_era5_daily(tgt.month, tgt.year, stat, template)
    obs = da.sel(time=str(tgt.date())).astype("float32")
    obs.to_dataset(name="TMP2m").to_netcdf(cache_path)
    return obs


# ── Main analysis ──────────────────────────────────────────────────────────────

def load_combined(year: int, lead_label: str, varname: str) -> xr.DataArray:
    path = COMBINED_DIR / f"{varname}_{year}_{lead_label}.nc"
    return xr.open_dataset(path)["TMP2m"]   # (sample=25, lat, lon)


def run_lead(lead_label: str, lead_days: int):
    print(f"\n=== Lead {lead_label} ===", flush=True)

    # Load a template (first year, first member) for grid info
    template_full = load_combined(YEARS[0], lead_label, "tmax").isel(sample=0)
    template_conus = subset_bbox(template_full, CONUS_BBOX)

    # Determine target month (Dec for +30/+60, Jan for +90)
    sample_tgt = target_date(YEARS[0], lead_days)
    tgt_month  = sample_tgt.month  # 12 or 1

    for extreme_type in ["heat", "cold"]:
        stat = "tmax" if extreme_type == "heat" else "tmin"
        pct  = HEAT_PCT if extreme_type == "heat" else COLD_PCT
        varname = stat

        print(f"  {extreme_type} ({stat}, {pct}th pct) ...", flush=True)
        threshold = compute_threshold(tgt_month, stat, pct, template_conus)
        threshold = subset_bbox(threshold, CONUS_BBOX)

        pred_probs = []  # list of (lat, lon) arrays, one per year
        obs_extremes = []

        for year in YEARS:
            tgt = target_date(year, lead_days)

            # Predicted probability = fraction of ensemble members exceeding threshold
            preds = load_combined(year, lead_label, varname)   # (25, lat, lon) in K
            preds_conus = subset_bbox(preds, CONUS_BBOX) - 273.15  # convert K → °C

            if extreme_type == "heat":
                exceeded = (preds_conus > threshold).astype(float)
            else:
                exceeded = (preds_conus < threshold).astype(float)

            pred_prob = exceeded.mean(dim="sample").values  # (lat, lon)
            pred_probs.append(pred_prob)

            # ERA5 observed extreme
            obs = get_era5_observed(tgt, stat, template_conus)
            obs_conus = subset_bbox(obs, CONUS_BBOX)

            if extreme_type == "heat":
                obs_extreme = (obs_conus > threshold).astype(float).values
            else:
                obs_extreme = (obs_conus < threshold).astype(float).values

            obs_extremes.append(obs_extreme)
            print(f"    {year} tgt={tgt.date()} "
                  f"obs_frac={obs_extreme.mean():.3f} "
                  f"pred_prob_mean={pred_prob.mean():.3f}", flush=True)

        pred_arr = np.stack(pred_probs,  axis=0)   # (n_years, lat, lon)
        obs_arr  = np.stack(obs_extremes, axis=0)  # (n_years, lat, lon)

        lat_conus = template_conus["lat"].values
        lon_conus = template_conus["lon"].values
        nlat, nlon = pred_arr.shape[1], pred_arr.shape[2]
        coords = {"lat": lat_conus, "lon": lon_conus}

        rho   = np.full((nlat, nlon), np.nan, dtype=np.float32)
        pval  = np.full((nlat, nlon), np.nan, dtype=np.float32)
        tau   = np.full((nlat, nlon), np.nan, dtype=np.float32)
        tau_p = np.full((nlat, nlon), np.nan, dtype=np.float32)
        brier = np.full((nlat, nlon), np.nan, dtype=np.float32)

        for i in range(nlat):
            for j in range(nlon):
                p = pred_arr[:, i, j]
                o = obs_arr[:,  i, j]
                # Brier score is always defined (mean squared error of prob forecast)
                brier[i, j] = np.mean((p - o) ** 2)
                # Rank correlations only defined when observed series is not constant
                if o.std() > 0:
                    r, pv = spearmanr(p, o)
                    rho[i, j]  = r
                    pval[i, j] = pv
                    t, tp = kendalltau(p, o)
                    tau[i, j]  = t
                    tau_p[i, j] = tp

        # Climatological Brier score: p_clim*(1-p_clim)
        # For 90th-pct heat: P(exceed) = 0.10, bs_clim = 0.09
        # For 10th-pct cold: P(below)  = 0.10, bs_clim = 0.09
        # Formula (pct/100)*(1-pct/100) gives 0.09 in both cases.
        bs_clim = (pct / 100.0) * (1.0 - pct / 100.0)

        # ── Spearman ──
        out_nc = RANK_DIR / f"spearman_{extreme_type}_{lead_label}.nc"
        xr.Dataset({
            "spearman_rho": xr.DataArray(
                rho, dims=["lat", "lon"], coords=coords,
                attrs={"long_name": f"Spearman rank correlation ({extreme_type}, lead={lead_label})",
                       "n_years": len(YEARS), "threshold_pct": pct}),
            "p_value": xr.DataArray(pval, dims=["lat", "lon"], coords=coords),
        }).to_netcdf(out_nc)
        print(f"  wrote: {out_nc}", flush=True)
        make_figure(rho, pval > 0.05, lat_conus, lon_conus,
                    extreme_type, lead_label, pct, metric="spearman")

        # ── Kendall tau ──
        out_nc = RANK_DIR / f"kendall_{extreme_type}_{lead_label}.nc"
        xr.Dataset({
            "kendall_tau": xr.DataArray(
                tau, dims=["lat", "lon"], coords=coords,
                attrs={"long_name": f"Kendall tau ({extreme_type}, lead={lead_label})",
                       "n_years": len(YEARS), "threshold_pct": pct}),
            "p_value": xr.DataArray(tau_p, dims=["lat", "lon"], coords=coords),
        }).to_netcdf(out_nc)
        print(f"  wrote: {out_nc}", flush=True)
        make_figure(tau, tau_p > 0.05, lat_conus, lon_conus,
                    extreme_type, lead_label, pct, metric="kendall")

        # ── Brier score ──
        out_nc = RANK_DIR / f"brier_{extreme_type}_{lead_label}.nc"
        xr.Dataset({
            "brier_score": xr.DataArray(
                brier, dims=["lat", "lon"], coords=coords,
                attrs={"long_name": f"Brier score ({extreme_type}, lead={lead_label})",
                       "n_years": len(YEARS), "threshold_pct": pct,
                       "bs_climatology": float(bs_clim)}),
        }).to_netcdf(out_nc)
        print(f"  wrote: {out_nc}", flush=True)
        # Stipple cells where model is worse than climatological reference
        make_figure(brier, brier > bs_clim, lat_conus, lon_conus,
                    extreme_type, lead_label, pct, metric="brier")


def make_figure(values, sig_mask, lat, lon,
                extreme_type: str, lead_label: str, pct: int,
                metric: str = "spearman"):
    """
    Plot a spatial skill/score map over the CONUS domain.

    values   : 2-D array (lat, lon)
    sig_mask : boolean array — True = stipple (p > 0.05 for rank corr;
               BS > BS_clim for Brier score)
    metric   : "spearman" | "kendall" | "brier"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # ── metric-specific configuration ──
        if metric == "spearman":
            cmap_colors  = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax   = 0.0, 0.6
            cbar_label   = "Rank correlation (Spearman rho)"
            ticks        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            clip_neg     = True   # negative → white (no positive skill)
            fig_path     = FIG_DIR / f"rank_corr_CONUS_{extreme_type}_{lead_label}.png"
            avg_prefix   = "rho"
            metric_title = "Spearman rank correlation"
        elif metric == "kendall":
            cmap_colors  = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax   = 0.0, 0.5
            cbar_label   = "Rank correlation (Kendall tau)"
            ticks        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            clip_neg     = True
            fig_path     = FIG_DIR / f"kendall_CONUS_{extreme_type}_{lead_label}.png"
            avg_prefix   = "tau"
            metric_title = "Kendall tau rank correlation"
        elif metric == "brier":
            cmap_colors  = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax   = 0.0, 0.25
            cbar_label   = "Brier score  (lower = better)"
            ticks        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
            clip_neg     = False
            fig_path     = FIG_DIR / f"brier_CONUS_{extreme_type}_{lead_label}.png"
            avg_prefix   = "BS"
            metric_title = "Brier score"
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        # Convert 0-360 lon to -180/180 for cartopy
        lon_plot = lon.copy()
        lon_plot[lon_plot > 180] -= 360

        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": proj})
        ax.set_facecolor("#d0d8e0")

        cmap = mcolors.LinearSegmentedColormap.from_list("skill", cmap_colors, N=256)
        cmap.set_bad("white")
        cmap.set_under("white")

        vals_plot = values.copy().astype(float)
        if clip_neg:
            vals_plot[vals_plot < 0] = np.nan

        im = ax.pcolormesh(lon_plot, lat, vals_plot, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto", transform=proj, zorder=1)

        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=2)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="white", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="0.35", zorder=3)

        # Stipple: insignificant cells (rank corr) or worse-than-climatology cells (Brier)
        LON2D, LAT2D = np.meshgrid(lon_plot, lat)
        stipple = np.asarray(sig_mask, dtype=bool) | np.isnan(np.asarray(sig_mask, dtype=float))
        ax.plot(LON2D[stipple].ravel(), LAT2D[stipple].ravel(),
                "k.", markersize=1.2, alpha=0.55, transform=proj, zorder=4)

        ax.set_extent([lon_plot.min(), lon_plot.max(),
                       lat.min(), lat.max()], crs=proj)

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False

        # Spherically area-weighted domain average: weight proportional to cos(lat)
        # accounts for smaller grid-cell areas at higher latitudes on a sphere
        lat_w = np.cos(np.radians(lat))
        w2d   = np.broadcast_to(lat_w[:, None], values.shape)
        valid = ~np.isnan(values)
        avg   = (np.nansum(values[valid] * w2d[valid]) / np.nansum(w2d[valid])) if valid.any() else np.nan
        ax.text(0.98, 0.04, f"{avg_prefix}={avg:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2), zorder=5)

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                            shrink=0.6, pad=0.05, aspect=28, extend="neither")
        cbar.set_label(cbar_label, fontsize=9)
        cbar.set_ticks(ticks)

        n_years = len(YEARS)
        yr_range = f"{min(YEARS)}/{min(YEARS)+1}–{max(YEARS)}/{max(YEARS)+1}"
        extreme_label = (f"heat extreme  (DJF Tmax > {pct}th pct)"
                         if extreme_type == "heat"
                         else f"cold extreme  (DJF Tmin < {pct}th pct)")
        ax.set_title(
            f"{metric_title} — winter {extreme_label}\n"
            f"ACE2-ERA5 lag ensemble  |  Lead {lead_label} from Nov 1  |  {yr_range}",
            fontsize=10,
        )

        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote figure: {fig_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"  WARN: figure failed ({metric}): {e}", flush=True)
        traceback.print_exc()


# ── Entry point ───────────────────────────────────────────────────────────────

from datetime import datetime   # noqa: E402 (needed for target_date)

def figures_only():
    """Regenerate all figures from existing saved .nc files (no ERA5 fetch)."""
    global YEARS
    for lead_label in LEADS:
        for extreme_type in ["heat", "cold"]:
            # Spearman
            nc = RANK_DIR / f"spearman_{extreme_type}_{lead_label}.nc"
            if nc.exists():
                ds  = xr.open_dataset(nc)
                rho  = ds["spearman_rho"].values
                pval = ds["p_value"].values
                lat, lon = ds["lat"].values, ds["lon"].values
                pct = int(ds["spearman_rho"].attrs.get("threshold_pct",
                          90 if extreme_type == "heat" else 10))
                n = int(ds["spearman_rho"].attrs.get("n_years", len(YEARS)))
                if n != len(YEARS):
                    YEARS = list(range(1980, 1980 + n))
                print(f"  spearman {extreme_type} {lead_label} ...", flush=True)
                make_figure(rho, pval > 0.05, lat, lon,
                            extreme_type, lead_label, pct, metric="spearman")
            else:
                print(f"  SKIP (missing): {nc.name}", flush=True)

            # Kendall tau
            nc = RANK_DIR / f"kendall_{extreme_type}_{lead_label}.nc"
            if nc.exists():
                ds    = xr.open_dataset(nc)
                tau   = ds["kendall_tau"].values
                tau_p = ds["p_value"].values
                lat, lon = ds["lat"].values, ds["lon"].values
                pct = int(ds["kendall_tau"].attrs.get("threshold_pct",
                          90 if extreme_type == "heat" else 10))
                print(f"  kendall {extreme_type} {lead_label} ...", flush=True)
                make_figure(tau, tau_p > 0.05, lat, lon,
                            extreme_type, lead_label, pct, metric="kendall")
            else:
                print(f"  SKIP (missing): {nc.name}", flush=True)

            # Brier score
            nc = RANK_DIR / f"brier_{extreme_type}_{lead_label}.nc"
            if nc.exists():
                ds    = xr.open_dataset(nc)
                brier = ds["brier_score"].values
                lat, lon = ds["lat"].values, ds["lon"].values
                pct = int(ds["brier_score"].attrs.get("threshold_pct",
                          90 if extreme_type == "heat" else 10))
                bs_clim = (pct / 100.0) * (1.0 - pct / 100.0)
                print(f"  brier {extreme_type} {lead_label} ...", flush=True)
                make_figure(brier, brier > bs_clim, lat, lon,
                            extreme_type, lead_label, pct, metric="brier")
            else:
                print(f"  SKIP (missing): {nc.name}", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated, e.g. '1980,1981,1982'")
    p.add_argument("--figures-only", action="store_true",
                   help="Regenerate figures from existing .nc files, skip ERA5 fetch")
    args = p.parse_args()

    if args.figures_only:
        figures_only()
        print("\nFigures regenerated.", flush=True)
        return

    global YEARS
    if args.years != "all":
        YEARS = [int(y) for y in args.years.split(",")]

    for lead_label, lead_days in LEADS.items():
        run_lead(lead_label, lead_days)
    print("\nRank correlation analysis complete.", flush=True)


if __name__ == "__main__":
    main()
