#!/usr/bin/env python3
"""DJF-aggregated skill maps for the ACE2-ERA5 10-year lag experiment.

For each year (1990-2000, n=11), at each CONUS grid cell:
  Predicted:  mean fraction of DJF days across 25 ensemble members where
              Tmax (heat) or Tmin (cold) exceeded the monthly climatological
              threshold (separate thresholds for Dec, Jan, Feb from 1940-2022).
  Observed:   same fraction computed from ERA5.

Rank correlations (Spearman, Kendall tau) computed across 11 years at each
grid cell. Brier score is MSE of predicted vs observed seasonal extreme frequency.

Requires: outputs/lag_10yr/combined_djf/tmax_djf_{year}.nc (from combine_lag_djf.py)

Outputs — _djf suffix, existing lead-specific files untouched:
  outputs/lag_10yr/rank_corr/spearman_{heat|cold}_djf.nc
  outputs/lag_10yr/rank_corr/kendall_{heat|cold}_djf.nc
  outputs/lag_10yr/rank_corr/brier_{heat|cold}_djf.nc
  outputs/lag_10yr/figures/rank_corr_CONUS_{heat|cold}_djf.png
  outputs/lag_10yr/figures/kendall_CONUS_{heat|cold}_djf.png
  outputs/lag_10yr/figures/brier_CONUS_{heat|cold}_djf.png
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

COMBINED_DJF = PROJECT_ROOT / "outputs/lag_10yr/combined_djf"
OUT_ROOT     = PROJECT_ROOT / "outputs/lag_10yr"
RANK_DIR     = OUT_ROOT / "rank_corr"
FIG_DIR      = OUT_ROOT / "figures"
CACHE_DIR    = OUT_ROOT / "era5_cache_rankcorr"
FORCING_DATA = Path("/home/jovyan/ace2-era5/data/lag_data/forcing_data_ace2era5")

for d in [RANK_DIR, FIG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

YEARS          = list(range(1990, 2001))   # 11 years with raw DJF outputs
DJF_MONTHS     = [12, 1, 2]
BASELINE_YEARS = list(range(1940, 2023))
HEAT_PCT       = 90
COLD_PCT       = 10
CONUS_BBOX     = (18.0, 72.0, 195.0, 305.0)
ERA5_ZARR      = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


# ── ERA5 helpers (reuse existing cache from rank_corr_analysis.py) ────────────

def load_land_mask(lat, lon):
    forcing_file = next(FORCING_DATA.glob("forcing_*.nc"), None)
    if forcing_file is None:
        return None
    ds = xr.open_dataset(forcing_file)
    lf = ds["land_fraction"]
    rename = {}
    if "latitude" in lf.dims:
        rename["latitude"] = "lat"
    if "longitude" in lf.dims:
        rename["longitude"] = "lon"
    if rename:
        lf = lf.rename(rename)
    lf_interp = lf.interp(lat=lat, lon=lon, method="nearest")
    ds.close()
    return (lf_interp.values > 0.5)


def _fetch_era5_once(month, year, stat, template):
    import socket
    socket.setdefaulttimeout(120)
    ds    = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={})
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
    daily = (da - 273.15).resample(time="1D").max() if stat == "tmax" \
            else (da - 273.15).resample(time="1D").min()
    daily = daily.sel(time=daily.time.dt.month == month).astype("float32").load()
    daily = daily.rename({lat_name: "lat", lon_name: "lon"})
    daily.name = "TMP2m"
    ds.close()
    return daily


def cache_era5_daily(month, year, stat, template, force=False):
    import time as _time
    path = CACHE_DIR / f"era5_daily_{stat}_m{month:02d}_y{year:04d}.nc"
    if path.exists() and not force:
        return xr.open_dataset(path)["TMP2m"]
    for attempt in range(5):
        try:
            daily = _fetch_era5_once(month, year, stat, template)
            daily.to_dataset().to_netcdf(path)
            return daily
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"    RETRY {year}-{month:02d} attempt {attempt+1}/5 ({e}) — wait {wait}s",
                  flush=True)
            _time.sleep(wait)
    raise RuntimeError(f"Failed ERA5 {stat} {year}-{month:02d} after 5 retries")


def compute_threshold(month, stat, pct, template):
    cache_path = CACHE_DIR / f"threshold_{stat}_m{month:02d}_pct{int(pct):02d}_1940_2022.nc"
    if cache_path.exists():
        return xr.open_dataset(cache_path)["threshold"]
    print(f"  Building {stat} {pct}th-pct threshold month {month} ...", flush=True)
    arrays = []
    for year in BASELINE_YEARS:
        try:
            arrays.append(cache_era5_daily(month, year, stat, template))
        except Exception as e:
            print(f"    WARN {year}-{month:02d}: {e}", flush=True)
    all_days  = xr.concat(arrays, dim="time").sortby("time")
    threshold = all_days.quantile(pct / 100.0, dim="time").drop_vars("quantile")
    threshold = threshold.rename("threshold").astype("float32")
    threshold.to_dataset().to_netcdf(cache_path)
    return threshold


# ── DJF extreme frequency ──────────────────────────────────────────────────────

def model_djf_freq(init_year, stat, thresholds):
    """Mean-member fraction of DJF days exceeding monthly threshold.

    thresholds: dict {month: (lat, lon) ndarray in °C}
    Returns (lat, lon) ndarray.
    """
    varname = "tmax" if stat == "tmax" else "tmin"
    ds      = xr.open_dataset(COMBINED_DJF / f"{varname}_djf_{init_year}.nc")
    data    = ds["TMP2m"] - 273.15   # (member, time, lat, lon) → °C
    times   = pd.DatetimeIndex(data["time"].values)
    ds.close()

    extreme_sum = None
    total_days  = 0
    for month in DJF_MONTHS:
        thr  = thresholds[month]                          # (lat, lon)
        mask = times.month == month
        monthly = data.isel(time=mask).values             # (member, n_days, lat, lon)
        if monthly.shape[1] == 0:
            continue
        exc = (monthly > thr) if stat == "tmax" else (monthly < thr)
        # mean over members first, then sum over days
        count = exc.astype(np.float32).mean(axis=0).sum(axis=0)   # (lat, lon)
        extreme_sum = count if extreme_sum is None else extreme_sum + count
        total_days += monthly.shape[1]

    return extreme_sum / total_days   # (lat, lon)


def obs_djf_freq(init_year, stat, thresholds, template):
    """Fraction of ERA5 DJF days exceeding monthly threshold.

    DJF for init_year: Dec of init_year, Jan/Feb of init_year+1.
    Returns (lat, lon) ndarray.
    """
    month_years = [(12, init_year), (1, init_year + 1), (2, init_year + 1)]
    extreme_sum = None
    total_days  = 0
    for month, year in month_years:
        obs = cache_era5_daily(month, year, stat, template)   # (days, lat, lon) °C
        thr = thresholds[month]
        exc = (obs.values > thr) if stat == "tmax" else (obs.values < thr)
        count = exc.astype(np.float32).sum(axis=0)
        extreme_sum = count if extreme_sum is None else extreme_sum + count
        total_days += obs.sizes["time"]
    return extreme_sum / total_days   # (lat, lon)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(values, sig_mask, lat, lon, extreme_type, pct, metric, land_mask=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if metric == "spearman":
            cmap_colors = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax  = 0.0, 0.6
            cbar_label  = "Rank correlation (Spearman rho)"
            ticks       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            clip_neg    = True
            fig_path    = FIG_DIR / f"rank_corr_CONUS_{extreme_type}_djf.png"
            avg_prefix  = "rho"
            title_metric = "Spearman rank correlation"
        elif metric == "kendall":
            cmap_colors = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax  = 0.0, 0.5
            cbar_label  = "Rank correlation (Kendall tau)"
            ticks       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            clip_neg    = True
            fig_path    = FIG_DIR / f"kendall_CONUS_{extreme_type}_djf.png"
            avg_prefix  = "tau"
            title_metric = "Kendall tau rank correlation"
        elif metric == "brier":
            cmap_colors = ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"]
            vmin, vmax  = 0.0, 0.05
            cbar_label  = "MSE of DJF extreme frequency  (lower = better)"
            ticks       = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
            clip_neg    = False
            fig_path    = FIG_DIR / f"brier_CONUS_{extreme_type}_djf.png"
            avg_prefix  = "MSE"
            title_metric = "MSE (DJF freq)"
        else:
            raise ValueError(metric)

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
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),     facecolor="white",  zorder=2)
        ax.add_feature(cfeature.LAKES.with_scale("50m"),     facecolor="white",  zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7,
                       edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),   linewidth=0.6,
                       edgecolor="black", zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"),    linewidth=0.3,
                       edgecolor="0.35",  zorder=3)

        LON2D, LAT2D = np.meshgrid(lon_plot, lat)
        stipple = np.asarray(sig_mask, dtype=bool) | np.isnan(np.asarray(sig_mask, dtype=float))
        ax.plot(LON2D[stipple].ravel(), LAT2D[stipple].ravel(),
                "k.", markersize=1.2, alpha=0.55, transform=proj, zorder=4)
        ax.set_extent([lon_plot.min(), lon_plot.max(), lat.min(), lat.max()], crs=proj)

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--", x_inline=False, y_inline=False)
        gl.top_labels   = False
        gl.right_labels = False

        lat_w = np.cos(np.radians(lat))
        w2d   = np.broadcast_to(lat_w[:, None], values.shape)
        valid = ~np.isnan(values)

        def wavg(mask):
            m = valid & mask if mask is not None else valid
            return (np.nansum(values[m] * w2d[m]) / np.nansum(w2d[m])) if m.any() else np.nan

        label_str = (f"land {avg_prefix}={wavg(land_mask):.3f}  ocean {avg_prefix}={wavg(~land_mask):.3f}"
                     if land_mask is not None else f"{avg_prefix}={wavg(None):.3f}")
        ax.text(0.98, 0.04, label_str, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2), zorder=5)

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                            shrink=0.6, pad=0.05, aspect=28, extend="neither")
        cbar.set_label(cbar_label, fontsize=9)
        cbar.set_ticks(ticks)

        extreme_label = (f"heat extreme  (DJF Tmax > {pct}th pct)"
                         if extreme_type == "heat"
                         else f"cold extreme  (DJF Tmin < {pct}th pct)")
        ax.set_title(
            f"{title_metric} — DJF-aggregated {extreme_label}\n"
            f"ACE2-ERA5 lag ensemble  |  Full DJF season  "
            f"|  1980/81–2016/17  (n={len(YEARS)})",
            fontsize=10,
        )

        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote figure: {fig_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"  WARN figure failed ({metric}): {e}", flush=True)
        traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_extreme(extreme_type):
    print(f"\n=== {extreme_type} ===", flush=True)
    stat = "tmax" if extreme_type == "heat" else "tmin"
    pct  = HEAT_PCT if extreme_type == "heat" else COLD_PCT

    # Template: first DJF file (already CONUS-subset)
    ds_tmpl  = xr.open_dataset(COMBINED_DJF / f"{stat}_djf_{YEARS[0]}.nc")
    template = ds_tmpl["TMP2m"].isel(member=0, time=0)
    lat_conus = template["lat"].values
    lon_conus = template["lon"].values
    ds_tmpl.close()

    # Monthly thresholds for Dec, Jan, Feb (in °C, on CONUS grid)
    thresholds = {}
    for month in DJF_MONTHS:
        thr = compute_threshold(month, stat, pct, template)
        thresholds[month] = subset_bbox(thr, CONUS_BBOX).values   # (lat, lon) ndarray

    pred_freqs = []
    obs_freqs  = []

    for year in YEARS:
        print(f"  {year} ...", flush=True)
        pred = model_djf_freq(year, stat, thresholds)
        obs  = obs_djf_freq(year, stat, thresholds, template)
        pred_freqs.append(pred)
        obs_freqs.append(obs)
        print(f"    obs_freq={obs.mean():.4f}  pred_freq={pred.mean():.4f}", flush=True)

    pred_arr = np.stack(pred_freqs, axis=0)   # (11, lat, lon)
    obs_arr  = np.stack(obs_freqs,  axis=0)

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
            brier[i, j] = float(np.mean((p - o) ** 2))
            if np.isfinite(p).all() and np.isfinite(o).all() and o.std() > 0:
                r, pv = spearmanr(p, o)
                rho[i, j]  = r
                pval[i, j] = pv
                t, tp = kendalltau(p, o)
                tau[i, j]  = t
                tau_p[i, j] = tp

    # Climatological MSE: always predict mean obs_freq across years
    bs_clim   = float(np.nanmean(np.var(obs_arr, axis=0)))
    land_mask = load_land_mask(lat_conus, lon_conus)

    for metric, data, p_mask in [
        ("spearman", rho,   pval  > 0.05),
        ("kendall",  tau,   tau_p > 0.05),
        ("brier",    brier, brier > bs_clim),
    ]:
        suffix = "spearman_rho" if metric == "spearman" else \
                 "kendall_tau"  if metric == "kendall"  else "brier_score"
        long   = {"spearman": f"Spearman rho DJF-aggregated ({extreme_type})",
                  "kendall":  f"Kendall tau DJF-aggregated ({extreme_type})",
                  "brier":    f"MSE DJF extreme frequency ({extreme_type})"}[metric]
        out_nc = RANK_DIR / f"{metric}_{extreme_type}_djf.nc"
        xr.Dataset({
            suffix: xr.DataArray(data, dims=["lat", "lon"], coords=coords,
                                 attrs={"long_name": long, "n_years": len(YEARS),
                                        "threshold_pct": pct}),
            **({} if metric == "brier" else
               {"p_value": xr.DataArray(p_mask.astype(np.float32),
                                        dims=["lat", "lon"], coords=coords)}),
        }).to_netcdf(out_nc)
        print(f"  wrote: {out_nc}", flush=True)
        make_figure(data, p_mask, lat_conus, lon_conus,
                    extreme_type, pct, metric=metric, land_mask=land_mask)


def main():
    for extreme_type in ["heat", "cold"]:
        run_extreme(extreme_type)
    print("\nDJF rank correlation analysis complete.", flush=True)


if __name__ == "__main__":
    main()
