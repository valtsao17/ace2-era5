#!/usr/bin/env python3
"""SST–HHE teleconnection analysis for JJA 1980–2016.

Reproduces the Jia et al. 2023-style analysis:
  1. ERA5 JJA HHE frequency in the Southeastern US (SE-US):
       For every JJA calendar day D and year Y (LOO):
         threshold(D,Y) = 90th pct of ERA5 daily Tmax within ±WINDOW_DAYS of D,
                          across all years ≠ Y  →  same rolling-window LOO approach
                          as postprocess_jja_lag.py
       HHE_freq(Y) = fraction of JJA days in year Y where ERA5 Tmax > threshold,
                     averaged over the SE-US bounding box.

  2. ERA5 JJA mean SST field  (lat × lon × year)

  3. Both linearly detrended over 1980–2016.

  4. Pearson correlation of detrended SST(lat,lon,year) with detrended HHE_freq(year)
     → global map of r.

  Optionally also computes the ACE2 version: replace ERA5 HHE freq with
  the mean ACE2 predicted probability (from the 3 target dates Jun/Jul/Aug 1)
  averaged over the SE-US.

Outputs (in outputs/lag_may/postprocess_jja/sst_teleconnection/):
  era5_jja_hhe_freq_seus.nc        – (year,)   ERA5 HHE frequency SE-US
  era5_jja_sst_mean.nc             – (year, lat, lon)  JJA mean SST (K)
  sst_hhe_corr_era5.nc             – (lat, lon)  Pearson r  ERA5 × ERA5
  figures/sst_hhe_corr_era5.png    – global correlation map (ERA5)
  figures/sst_hhe_corr_ace2.png    – global correlation map (ACE2 prob, if available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.era5 import cache_era5_month
from hiro_ace_pipeline.io import write_atomic

# ── paths ──────────────────────────────────────────────────────────────────
POSTPROC_DIR  = PROJECT_ROOT / "outputs/lag_may/postprocess_jja"
CACHE_DIR     = POSTPROC_DIR / "era5_cache"
OUT_DIR       = POSTPROC_DIR / "sst_teleconnection"
FIGURES_DIR   = OUT_DIR / "figures"
METRICS_DIR   = OUT_DIR / "metrics"   # per-target predicted prob saved here by postprocess script
COMBINED_DIR  = PROJECT_ROOT / "outputs/lag_may/combined_jja"

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ── constants ──────────────────────────────────────────────────────────────
YEARS       = list(range(1980, 2017))
WINDOW_DAYS = 15       # ±15 days rolling LOO window (same as postprocess_jja_lag.py)
CHUNK_LAT   = 10

# SE-US bounding box  (lat S,N ; lon W,E in 0-360)
SEUS_BBOX = (25.0, 37.0, 260.0, 285.0)

# Global bbox for SST correlation map (ocean only via NaN masking)
GLOBAL_BBOX = (-90.0, 90.0, 0.0, 360.0)

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
    path = COMBINED_DIR / f"tmax_jja_{YEARS[0]}.nc"
    with xr.open_dataset(path) as ds:
        return ds["TMP2m"].isel(member=0, time=0).load()


def jja_calendar_days() -> list[tuple[int, int]]:
    """All (month, day) pairs in JJA (non-leap reference)."""
    days = []
    for m, nd in [(6, 30), (7, 31), (8, 31)]:
        for d in range(1, nd + 1):
            days.append((m, d))
    return days


def subset_seus(da: xr.DataArray) -> xr.DataArray:
    """Subset DataArray to SE-US bbox."""
    lat_s, lat_n, lon_w, lon_e = SEUS_BBOX
    lat_name = next(c for c in da.coords if c in ("lat", "latitude"))
    lon_name = next(c for c in da.coords if c in ("lon", "longitude"))
    lat_vals = da[lat_name].values
    lat_slice = slice(lat_n, lat_s) if lat_vals[0] > lat_vals[-1] else slice(lat_s, lat_n)
    return da.sel({lat_name: lat_slice, lon_name: slice(lon_w, lon_e)})


def cos_lat_mean_2d(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid   = np.isfinite(field)
    num     = np.nansum(field * weights * valid)
    den     = np.nansum(weights * valid)
    return float(num / den) if den > 0 else float("nan")


def detrend_1d(x: np.ndarray) -> np.ndarray:
    """Linear detrend a 1-D array, returning residuals."""
    return signal.detrend(x, type="linear").astype(np.float32)


def detrend_along_year(arr: np.ndarray) -> np.ndarray:
    """Linear detrend along axis-0 (years) for a (year, lat, lon) array."""
    out = np.empty_like(arr)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            col = arr[:, i, j]
            if np.all(np.isnan(col)):
                out[:, i, j] = np.nan
            else:
                out[:, i, j] = signal.detrend(col, type="linear")
    return out.astype(np.float32)


# ── ERA5 HHE frequency ─────────────────────────────────────────────────────

def compute_era5_hhe_frequency(template: xr.DataArray, force: bool = False) -> np.ndarray:
    """Compute ERA5 JJA HHE frequency in SE-US for each year (LOO).

    Returns (n_years,) array.  Cached to OUT_DIR/era5_jja_hhe_freq_seus.nc.
    """
    out_path = OUT_DIR / "era5_jja_hhe_freq_seus.nc"
    if out_path.exists() and not force:
        with xr.open_dataset(out_path) as ds:
            return ds["hhe_freq"].values

    print("Computing ERA5 JJA HHE frequency in SE-US (LOO) ...", flush=True)

    # Pre-cache all needed months (May–Sep for ±15d windows around Jun–Aug)
    months_needed = set()
    for year in YEARS:
        for month, day in jja_calendar_days():
            target = pd.Timestamp(year=year, month=month, day=day)
            for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
                d = target + pd.Timedelta(days=delta)
                if YEARS[0] <= d.year <= YEARS[-1]:
                    months_needed.add((d.year, d.month))
    print(f"  Caching {len(months_needed)} ERA5 month files ...", flush=True)
    for y, m in tqdm(sorted(months_needed), desc="ERA5 month cache"):
        cache_era5_month(ERA5_ZARR, y, m, GLOBAL_BBOX, template, CACHE_DIR)

    # For efficiency: preload all cached monthly ERA5 data into memory
    # grouped by (year, month)
    print("  Loading cached ERA5 months into memory ...", flush=True)
    era5_monthly: dict[tuple[int, int], np.ndarray] = {}
    era5_monthly_times: dict[tuple[int, int], pd.DatetimeIndex] = {}
    for y, m in tqdm(sorted(months_needed), desc="ERA5 load"):
        p = cache_era5_month(ERA5_ZARR, y, m, GLOBAL_BBOX, template, CACHE_DIR)
        with xr.open_dataset(p) as ds:
            da = ds["era5_daily_tmax_C"].load()
            era5_monthly[(y, m)] = da.values        # (n_days, lat, lon)
            era5_monthly_times[(y, m)] = pd.DatetimeIndex(da.time.values)

    n_lat = template.sizes.get("lat", template.shape[-2])
    n_lon = template.sizes.get("lon", template.shape[-1])

    hhe_freq = np.full(len(YEARS), np.nan, dtype=np.float32)

    for y_idx, year in enumerate(tqdm(YEARS, desc="LOO HHE frequency")):
        other_years = [y for y in YEARS if y != year]
        jja_days = jja_calendar_days()
        n_jja = len(jja_days)

        # Compute threshold and extreme flag for each JJA calendar day
        extreme_count   = 0
        day_count_valid = 0

        # We process the threshold computation in lat chunks to save memory
        # but since we only need the SE-US average, subset first
        lat_vals = template["lat"].values if "lat" in template.coords else np.arange(n_lat)
        lon_vals = template["lon"].values if "lon" in template.coords else np.arange(n_lon)

        lat_s, lat_n, lon_w, lon_e = SEUS_BBOX
        lat_idx_lo = int(np.searchsorted(np.sort(lat_vals), lat_s))
        lat_idx_hi = int(np.searchsorted(np.sort(lat_vals), lat_n))
        if lat_vals[0] > lat_vals[-1]:  # descending lat
            lat_idx_lo = int(np.searchsorted(-lat_vals[::-1], -lat_n))
            lat_idx_hi = int(np.searchsorted(-lat_vals[::-1], -lat_s))
            lat_slice  = slice(n_lat - lat_idx_hi, n_lat - lat_idx_lo)
        else:
            lat_slice = slice(lat_idx_lo, lat_idx_hi)

        lon_idx_lo = int(np.searchsorted(lon_vals, lon_w))
        lon_idx_hi = int(np.searchsorted(lon_vals, lon_e))
        lon_slice  = slice(lon_idx_lo, lon_idx_hi)

        seus_lat = lat_vals[lat_slice]
        cos_w    = np.cos(np.deg2rad(seus_lat))[:, np.newaxis]

        for month, day in jja_days:
            try:
                target    = pd.Timestamp(year=year, month=month, day=day)
                target_oy = [pd.Timestamp(year=oy, month=month, day=day) for oy in other_years]
            except ValueError:
                continue   # Feb 29 in non-leap years; skip

            t_lo = target - pd.Timedelta(days=WINDOW_DAYS)
            t_hi = target + pd.Timedelta(days=WINDOW_DAYS)

            # Pool other-year window data in SE-US only
            window_chunks = []
            for oy, oy_target in zip(other_years, target_oy):
                oy_lo = oy_target - pd.Timedelta(days=WINDOW_DAYS)
                oy_hi = oy_target + pd.Timedelta(days=WINDOW_DAYS)
                months_needed_oy = set()
                for delta in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
                    d = oy_target + pd.Timedelta(days=delta)
                    months_needed_oy.add((d.year, d.month))
                for key in months_needed_oy:
                    if key not in era5_monthly:
                        continue
                    times = era5_monthly_times[key]
                    mask  = (times >= oy_lo) & (times <= oy_hi)
                    if mask.any():
                        chunk = era5_monthly[key][np.where(mask)[0], :, :]
                        window_chunks.append(chunk[:, lat_slice, lon_slice])

            if not window_chunks:
                continue

            pooled    = np.concatenate(window_chunks, axis=0)   # (N_samples, seus_lat, seus_lon)
            threshold = np.nanquantile(pooled, 0.90, axis=0)    # (seus_lat, seus_lon)

            # ERA5 tmax at (year, target date) in SE-US
            key = (year, month)
            if key not in era5_monthly:
                continue
            times = era5_monthly_times[key]
            tidx  = int(np.argmin(np.abs(times - target)))
            era5_today = era5_monthly[key][tidx, lat_slice, lon_slice]  # (seus_lat, seus_lon)

            extreme = (era5_today > threshold).astype(np.float32)
            # cos-lat weighted mean over SE-US
            extreme_mean = float(np.nansum(extreme * cos_w) / np.nansum(cos_w * np.isfinite(extreme)))
            extreme_count   += extreme_mean
            day_count_valid += 1

        if day_count_valid > 0:
            hhe_freq[y_idx] = extreme_count / day_count_valid

    # Save
    da = xr.DataArray(
        hhe_freq,
        dims=["year"],
        coords={"year": YEARS},
        attrs={"units": "fraction", "long_name": "ERA5 JJA HHE frequency SE-US (LOO)"},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(da.rename("hhe_freq").to_dataset(), out_path)
    print(f"  saved {out_path.name}", flush=True)
    return hhe_freq


# ── ERA5 JJA mean SST ──────────────────────────────────────────────────────

def compute_era5_jja_sst(force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return JJA mean SST (n_years, lat, lon) in K, plus lat and lon arrays.

    Cached to OUT_DIR/era5_jja_sst_mean.nc.
    """
    out_path = OUT_DIR / "era5_jja_sst_mean.nc"
    if out_path.exists() and not force:
        with xr.open_dataset(out_path) as ds:
            return ds["sst"].values, ds["lat"].values, ds["lon"].values

    print("Computing ERA5 JJA mean SST from ARCO-ERA5 zarr ...", flush=True)
    ds_era5 = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={"time": 24})

    # Find SST variable name
    sst_candidates = ("sea_surface_temperature", "sst", "SST", "tos")
    sst_var = next((v for v in sst_candidates if v in ds_era5.data_vars), None)
    if sst_var is None:
        raise KeyError(f"Cannot find SST variable in ERA5 zarr. Available: {list(ds_era5.data_vars)[:20]}")

    sst_data = ds_era5[sst_var]

    sst_yearly = []
    for year in tqdm(YEARS, desc="ERA5 JJA SST"):
        start = f"{year}-06-01"
        end   = f"{year}-08-31"
        da    = sst_data.sel(time=slice(start, end))
        mean  = da.mean("time").compute().values.astype(np.float32)
        sst_yearly.append(mean)

    sst_arr = np.stack(sst_yearly, axis=0)   # (n_years, lat, lon)

    lat_name = next(c for c in ds_era5.coords if c in ("lat", "latitude"))
    lon_name = next(c for c in ds_era5.coords if c in ("lon", "longitude"))
    lat_vals = ds_era5[lat_name].values.astype(np.float32)
    lon_vals = ds_era5[lon_name].values.astype(np.float32)
    ds_era5.close()

    da_out = xr.DataArray(
        sst_arr,
        dims=["year", "lat", "lon"],
        coords={"year": YEARS, "lat": lat_vals, "lon": lon_vals},
        attrs={"units": "K", "long_name": "ERA5 JJA mean SST"},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(da_out.rename("sst").to_dataset(), out_path)
    print(f"  saved {out_path.name}", flush=True)
    return sst_arr, lat_vals, lon_vals


# ── correlation ────────────────────────────────────────────────────────────

def pearson_corr_map(sst: np.ndarray, hhe: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r and p-value between detrended SST(year,lat,lon) and detrended HHE(year).

    Returns (corr, pval), each (lat, lon).  NaN where SST is all-NaN (land).
    """
    n_years, n_lat, n_lon = sst.shape
    corr = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    pval = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    hhe_dt = detrend_1d(hhe)

    for i in range(n_lat):
        for j in range(n_lon):
            col = sst[:, i, j]
            if np.all(np.isnan(col)):
                continue
            col_dt = detrend_1d(col)
            mask   = np.isfinite(col_dt) & np.isfinite(hhe_dt)
            if mask.sum() < 10:
                continue
            r, p   = stats.pearsonr(col_dt[mask], hhe_dt[mask])
            corr[i, j] = r
            pval[i, j] = p

    return corr, pval


# ── ACE2 HHE proxy ─────────────────────────────────────────────────────────

def load_ace2_seus_prob(template: xr.DataArray) -> np.ndarray | None:
    """Load ACE2 mean predicted probability averaged over SE-US.

    Reads tau_map_*.nc files (which have lat/lon) from the postprocessing output
    to reconstruct the per-target predicted probs, if available.
    Falls back to None if data not present.

    This uses the 3 target dates (Jun1, Jul1, Aug1) as a proxy for JJA HHE freq.
    """
    targets = [("Jun1", 6, 1), ("Jul1", 7, 1), ("Aug1", 8, 1)]
    probs_all = []

    for label, tm, td in targets:
        # Try to find cached predicted-prob files from postprocess_jja_lag output
        prob_path = POSTPROC_DIR / "metrics" / f"pred_prob_{label.lower()}.nc"
        if not prob_path.exists():
            return None
        with xr.open_dataset(prob_path) as ds:
            if "pred_prob" not in ds:
                return None
            da = ds["pred_prob"]   # (year, lat, lon)
            # Subset to SE-US
            lat_name = next(c for c in da.coords if c in ("lat", "latitude"))
            lon_name = next(c for c in da.coords if c in ("lon", "longitude"))
            lat_s, lat_n, lon_w, lon_e = SEUS_BBOX
            lat_vals = da[lat_name].values
            if lat_vals[0] > lat_vals[-1]:
                lat_sel = slice(lat_n, lat_s)
            else:
                lat_sel = slice(lat_s, lat_n)
            da_seus = da.sel({lat_name: lat_sel, lon_name: slice(lon_w, lon_e)})
            cos_w   = np.cos(np.deg2rad(da_seus[lat_name].values))
            # Weighted mean over lat/lon → (year,)
            weighted = (da_seus.values * cos_w[np.newaxis, :, np.newaxis])
            seus_mean = np.nansum(weighted, axis=(1, 2)) / np.nansum(
                cos_w[:, np.newaxis] * np.isfinite(da_seus.values), axis=(1, 2)
            )
            probs_all.append(seus_mean)

    if not probs_all:
        return None
    return np.nanmean(np.stack(probs_all, axis=0), axis=0)  # (n_years,)


# ── plotting ───────────────────────────────────────────────────────────────

# Domain matching the reference sst.png (Pacific–Atlantic basin)
_SST_EXTENT = [155, 315, -23, 68]   # lon W, lon E, lat S, lat N  (0-360)


def _draw_corr_panel(ax, corr: np.ndarray, pval: np.ndarray,
                     lat: np.ndarray, lon: np.ndarray, panel_label: str):
    """Draw one correlation panel with stippling and SE-US box."""
    mesh = ax.pcolormesh(lon, lat, corr,
                         shading="auto", cmap="RdBu_r",
                         vmin=-1.0, vmax=1.0,
                         transform=_PLATE)
    ax.set_extent(_SST_EXTENT, crs=_PLATE)
    ax.add_feature(cfeature.LAND,      facecolor="0.88", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,    zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3,    zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels   = False
    gl.right_labels = False

    # Stipple where p < 0.05
    sig = np.isfinite(pval) & (pval < 0.05)
    lon_g, lat_g = np.meshgrid(lon, lat)
    ax.scatter(lon_g[sig], lat_g[sig], s=0.8, c="k", alpha=0.5,
               transform=_PLATE, zorder=4, linewidths=0)

    # SE-US box
    lat_s, lat_n, lon_w, lon_e = SEUS_BBOX
    import matplotlib.patches as mpatches
    rect = mpatches.Rectangle(
        (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
        linewidth=1.5, edgecolor="black", facecolor="none",
        linestyle="--", transform=_PLATE, zorder=5,
    )
    ax.add_patch(rect)

    ax.text(0.01, 0.97, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            verticalalignment="top", horizontalalignment="left")

    return mesh


def plot_corr_panels(
    corr_era5: np.ndarray, pval_era5: np.ndarray,
    corr_ace2: np.ndarray | None, pval_ace2: np.ndarray | None,
    lat: np.ndarray, lon: np.ndarray,
    out_path: Path,
    years: list[int],
):
    """Two-panel (a) ERA5, (b) Hindcast correlation map — matches sst.png layout."""
    panels = [("(a) ERA5", corr_era5, pval_era5)]
    if corr_ace2 is not None:
        panels.append(("(b) Hindcast", corr_ace2, pval_ace2))

    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, figsize=(8.5, 4.2 * n),
        subplot_kw=dict(projection=_PLATE),
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for ax, (plabel, corr, pval) in zip(axes, panels):
        mesh = _draw_corr_panel(ax, corr, pval, lat, lon, plabel)

    # Shared colorbar
    fig.colorbar(mesh, ax=axes, orientation="vertical",
                 shrink=0.6, pad=0.02, label="Pearson r")

    fig.suptitle(
        f"Relationship between SSTs and JJA HHE frequency in SE-US\n"
        f"{years[0]}–{years[-1]}  |  ±15-day threshold  |  stippling: p < 0.05",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--years", default="all", help="'all' or comma-separated years")
    return p.parse_args()


def main():
    global YEARS
    args = parse_args()
    if args.years != "all":
        YEARS = [int(y) for y in args.years.split(",")]
        print(f"Running on years: {YEARS[0]}–{YEARS[-1]} ({len(YEARS)} years)", flush=True)
    for d in [OUT_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    template = get_template()

    # 1. ERA5 HHE frequency in SE-US
    hhe_freq = compute_era5_hhe_frequency(template, force=args.force)
    print(f"\nERA5 SE-US HHE freq (mean): {np.nanmean(hhe_freq):.3f}", flush=True)

    # 2. ERA5 JJA mean SST
    sst_arr, sst_lat, sst_lon = compute_era5_jja_sst(force=args.force)
    print(f"SST array shape: {sst_arr.shape}", flush=True)

    # 3. Correlation: detrended SST × detrended ERA5 HHE freq
    print("\nComputing SST–HHE correlation map (ERA5) ...", flush=True)
    sst_dt    = detrend_along_year(sst_arr)
    corr_era5, pval_era5 = pearson_corr_map(sst_dt, hhe_freq)

    da_corr = xr.DataArray(
        corr_era5, dims=["lat", "lon"],
        coords={"lat": sst_lat, "lon": sst_lon},
        attrs={"long_name": "Pearson r: detrended JJA SST vs detrended ERA5 JJA HHE freq SE-US"},
    )
    write_atomic(da_corr.rename("r").to_dataset(),
                 OUT_DIR / "sst_hhe_corr_era5.nc")

    # 4. ACE2 version (if predicted-prob files exist from postprocess_jja_lag.py)
    print("\nAttempting ACE2 SE-US HHE probability ...", flush=True)
    ace2_prob = load_ace2_seus_prob(template)
    corr_ace2, pval_ace2 = None, None
    if ace2_prob is not None:
        corr_ace2, pval_ace2 = pearson_corr_map(sst_dt, ace2_prob)
        da_corr_ace2 = xr.DataArray(
            corr_ace2, dims=["lat", "lon"],
            coords={"lat": sst_lat, "lon": sst_lon},
            attrs={"long_name": "Pearson r: detrended JJA SST vs detrended ACE2 JJA HHE prob SE-US"},
        )
        write_atomic(da_corr_ace2.rename("r").to_dataset(),
                     OUT_DIR / "sst_hhe_corr_ace2.nc")
    else:
        print("  ACE2 predicted-prob files not found — will plot ERA5 panel only.")

    # 5. Two-panel figure  (a) ERA5, (b) ACE2 — matching sst.png layout
    plot_corr_panels(
        corr_era5, pval_era5,
        corr_ace2, pval_ace2,
        sst_lat, sst_lon,
        FIGURES_DIR / "sst_hhe_corr_panels.png",
        years=YEARS,
    )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
