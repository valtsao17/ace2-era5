#!/usr/bin/env python3
"""SST–HHE teleconnection using ±7-day LOO seasonal-frequency approach.

Parallel to sst_teleconnection_jja.py but using seasonal frequency from
seasonal_jja_skill.py instead of the per-date ±15-day LOO from postprocess_jja_lag.py.

ERA5 HHE freq and ACE2 HHE freq are both read from jja_seasonal_freqs.nc
(produced by seasonal_jja_skill.py). For the SST the existing cached
era5_jja_sst_mean.nc is reused if available, else computed from ARCO zarr.

Outputs → outputs/lag_may/sst_teleconnection_sliding7d/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import signal, stats
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
SLIDING_DIR   = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
OLD_SST_CACHE = PROJECT_ROOT / "outputs/lag_may/postprocess_jja/sst_teleconnection/era5_jja_sst_mean.nc"
OUT_DIR       = PROJECT_ROOT / "outputs/lag_may/sst_teleconnection_sliding7d"
FIGURES_DIR   = OUT_DIR / "figures"
COMBINED_DIR  = PROJECT_ROOT / "outputs/lag_may/combined_jja"

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

YEARS     = list(range(1980, 2017))
SEUS_BBOX = (25.0, 37.0, 260.0, 285.0)   # lat_s, lat_n, lon_w_360, lon_e_360
TNA_BBOX  = (5.5, 23.5, 302.5, 345.0)    # Tropical North Atlantic (Enfield et al. 1999)
CONUS_EXTENT = (-175.0, -50.0, 12.0, 77.0)   # lon_min, lon_max, lat_min, lat_max

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight"})

_COAST_GEOMS = None


def _get_coast():
    global _COAST_GEOMS
    if _COAST_GEOMS is None:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        _COAST_GEOMS = list(shpreader.Reader(shp).geometries())
    return _COAST_GEOMS


def _draw_coast_and_borders(ax, xlim, ylim):
    from shapely.geometry import box
    vp = box(xlim[0], ylim[0], xlim[1], ylim[1])

    def _plot(geom, color, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot(g, color, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=4)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, linewidth=lw, zorder=4)

    for geom in _get_coast():
        try:
            _plot(geom.intersection(vp), "black", 0.5)
        except Exception:
            continue


def bbox_mean(arr, lat, lon_360, bbox):
    """Cos-lat weighted mean over a bbox. arr: (year, lat, lon) in 0-360 lon."""
    lat_s, lat_n, lon_w, lon_e = bbox
    lat_sel = (lat >= lat_s) & (lat <= lat_n)
    lon_sel = (lon_360 >= lon_w) & (lon_360 <= lon_e)
    sub = arr[:, lat_sel, :][:, :, lon_sel]
    cos_w = np.cos(np.deg2rad(lat[lat_sel]))[:, np.newaxis]
    valid = np.isfinite(sub)
    num   = np.nansum(sub * cos_w[np.newaxis] * valid, axis=(1, 2))
    den   = np.nansum(cos_w[np.newaxis] * valid, axis=(1, 2))
    return np.where(den > 0, num / den, np.nan).astype(np.float32)


def detrend_1d(x):
    return signal.detrend(x, type="linear").astype(np.float32)


def detrend_along_year(arr):
    out = np.empty_like(arr)
    nlat, nlon = arr.shape[1], arr.shape[2]
    for i in range(nlat):
        for j in range(nlon):
            col = arr[:, i, j]
            if np.all(np.isnan(col)):
                out[:, i, j] = np.nan
            else:
                out[:, i, j] = signal.detrend(col, type="linear")
    return out.astype(np.float32)


def pearson_corr_map(sst, hhe):
    """Pearson r and p-value: (n_years, lat, lon) SST vs (n_years,) HHE series."""
    n_years, nlat, nlon = sst.shape
    corr = np.full((nlat, nlon), np.nan, dtype=np.float32)
    pval = np.full((nlat, nlon), np.nan, dtype=np.float32)
    hhe_dt = detrend_1d(hhe)
    for i in range(nlat):
        for j in range(nlon):
            col = sst[:, i, j]
            if np.all(np.isnan(col)):
                continue
            col_dt = detrend_1d(col)
            mask = np.isfinite(col_dt) & np.isfinite(hhe_dt)
            if mask.sum() < 10:
                continue
            r, p = stats.pearsonr(col_dt[mask], hhe_dt[mask])
            corr[i, j] = r
            pval[i, j] = p
    return corr, pval


def regression_map(field_dt, predictor_dt):
    """Per-grid-cell OLS regression of a detrended field (n_years, lat, lon)
    onto a detrended scalar predictor series. Returns slope and two-tailed
    p-value maps, shape (lat, lon)."""
    n_years, nlat, nlon = field_dt.shape
    slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
    pval  = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            col = field_dt[:, i, j]
            if np.all(np.isnan(col)):
                continue
            mask = np.isfinite(col) & np.isfinite(predictor_dt)
            if mask.sum() < 10:
                continue
            res = stats.linregress(predictor_dt[mask], col[mask])
            slope[i, j] = res.slope
            pval[i, j]  = res.pvalue
    return slope, pval


def load_or_compute_sst():
    """Load SST from old cache if available, else fetch from ARCO zarr."""
    if OLD_SST_CACHE.exists():
        print(f"Reusing SST cache from {OLD_SST_CACHE}", flush=True)
        with xr.open_dataset(OLD_SST_CACHE) as ds:
            return ds["sst"].values, ds["lat"].values, ds["lon"].values

    print("Computing ERA5 JJA mean SST from ARCO zarr ...", flush=True)
    import gcsfs
    ds_era5 = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={"time": 24},
                               storage_options={"token": "anon"})
    sst_candidates = ("sea_surface_temperature", "sst", "SST", "tos")
    sst_var = next((v for v in sst_candidates if v in ds_era5.data_vars), None)
    if sst_var is None:
        raise KeyError(f"SST variable not found in zarr. Available: {list(ds_era5.data_vars)[:20]}")

    sst_data = ds_era5[sst_var]
    sst_yearly = []
    for year in tqdm(YEARS, desc="ERA5 JJA SST"):
        da = sst_data.sel(time=slice(f"{year}-06-01", f"{year}-08-31"))
        sst_yearly.append(da.mean("time").compute().values.astype(np.float32))
    sst_arr = np.stack(sst_yearly, axis=0)

    lat_name = next(c for c in ds_era5.coords if c in ("lat", "latitude"))
    lon_name = next(c for c in ds_era5.coords if c in ("lon", "longitude"))
    lat_vals = ds_era5[lat_name].values.astype(np.float32)
    lon_vals = ds_era5[lon_name].values.astype(np.float32)
    ds_era5.close()

    out = OUT_DIR / "era5_jja_sst_mean.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"sst": xr.DataArray(sst_arr, dims=["year", "lat", "lon"],
                coords={"year": YEARS, "lat": lat_vals, "lon": lon_vals})}).to_netcdf(out)
    print(f"wrote: {out}", flush=True)
    return sst_arr, lat_vals, lon_vals


def _roll_to_180(field, lon_360):
    split = np.searchsorted(lon_360, 180.0)
    n = len(lon_360)
    lon_r = np.concatenate([lon_360[split:] - 360.0, lon_360[:split]])
    if field.ndim == 2:
        field_r = np.roll(field, n - split, axis=-1)
    else:
        field_r = np.roll(field, n - split, axis=-1)
    return field_r, lon_r


def _thin_mask(mask, max_points=8000, seed=0):
    """Random subsample of a boolean mask so dense significant regions read
    as a stipple texture rather than the regular SST grid lattice (every
    0.25deg point plotted makes the dot rows/columns visible as a grid)."""
    idx = np.flatnonzero(mask)
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    thinned = np.zeros(mask.shape, dtype=bool)
    thinned.flat[idx] = True
    return thinned


def _render_corr_ax(ax, corr, pval, lat, lon_360, title):
    """Draw one SST-correlation panel (map + stipple) on a given axes."""
    corr_r, lon_r = _roll_to_180(corr, lon_360)
    pval_r, _     = _roll_to_180(pval, lon_360)

    lon_min, lon_max = -170.0, -40.0   # match Atlantic/Pacific SST domain
    lat_min, lat_max = -25.0, 70.0

    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_r >= lon_min) & (lon_r <= lon_max)

    corr_sub = corr_r[lat_sel, :][:, lon_sel]
    pval_sub = pval_r[lat_sel, :][:, lon_sel]
    lat_sub  = lat[lat_sel]
    lon_sub  = lon_r[lon_sel]

    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)

    ax.set_facecolor("#d0e8f0")
    mesh = ax.pcolormesh(LON2D, LAT2D, corr_sub,
                         cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                         shading="nearest", zorder=1)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    _draw_coast_and_borders(ax, (lon_min, lon_max), (lat_min, lat_max))

    sig = np.isfinite(pval_sub) & (pval_sub < 0.05)
    sig = _thin_mask(sig)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.0, c="k", alpha=0.5, zorder=5, linewidths=0)

    ax.set_xticks(range(-160, -30, 20))
    ax.set_xticklabels([f"{abs(x)}°W" for x in range(-160, -30, 20)], fontsize=8)
    ax.set_yticks(range(-20, 71, 20))
    ax.set_yticklabels([f"{abs(y)}°S" if y < 0 else f"{y}°N" for y in range(-20, 71, 20)], fontsize=8)
    ax.set_title(title, fontsize=10)
    return mesh


def plot_corr_panel(corr, pval, lat, lon_360, title, out_path):
    """Plot a single SST correlation map on a plain matplotlib axes (no cartopy)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = _render_corr_ax(ax, corr, pval, lat, lon_360, title)
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="Pearson r", orientation="vertical")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def plot_corr_comparison(corr_era5, pval_era5, corr_ace2, pval_ace2,
                         lat, lon_360, title_era5, title_ace2, out_path):
    """ERA5 vs ACE2 SST-correlation maps side by side for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    _render_corr_ax(axes[0], corr_era5, pval_era5, lat, lon_360, title_era5)
    mesh = _render_corr_ax(axes[1], corr_ace2, pval_ace2, lat, lon_360, title_ace2)
    fig.colorbar(mesh, ax=axes, shrink=0.8, label="Pearson r", orientation="vertical")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


_CONUS_LAND_GEOMS  = None
_CONUS_STATE_GEOMS = None


def _get_conus_borders():
    global _CONUS_LAND_GEOMS, _CONUS_STATE_GEOMS
    if _CONUS_LAND_GEOMS is None:
        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _CONUS_LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
        _CONUS_STATE_GEOMS = list(shpreader.Reader(shp).geometries())
    return _CONUS_LAND_GEOMS, _CONUS_STATE_GEOMS


def _draw_conus_borders(ax, xlim, ylim):
    from shapely.geometry import box
    vp = box(xlim[0], ylim[0], xlim[1], ylim[1])
    land_geoms, state_geoms = _get_conus_borders()

    def _plot(geom, color, lw):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                _plot(g, color, lw)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=4)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color=color, linewidth=lw, zorder=4)

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


def _render_regression_ax(ax, slope_sub, pval_sub, lon_sub, lat_sub, vmax, title):
    """Draw one HHE-freq-on-TNA-SST regression panel (CONUS map + stipple)."""
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT

    LON2D, LAT2D = np.meshgrid(lon_sub, lat_sub)
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(LON2D, LAT2D, slope_sub,
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="nearest", zorder=1)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0)
    _draw_conus_borders(ax, (lon_min, lon_max), (lat_min, lat_max))

    sig = np.isfinite(pval_sub) & (pval_sub < 0.05)
    sig = _thin_mask(sig)
    ax.scatter(LON2D[sig], LAT2D[sig], s=2.5, c="k", alpha=0.55, zorder=5, linewidths=0)

    ax.set_xticks(range(-160, -40, 20))
    ax.set_xticklabels([f"{abs(x)}°W" for x in range(-160, -40, 20)], fontsize=8)
    ax.set_yticks(range(20, 80, 10))
    ax.set_yticklabels([f"{y}°N" for y in range(20, 80, 10)], fontsize=8)
    ax.set_title(title, fontsize=10)
    return mesh


def plot_regression_comparison(slope_era5, pval_era5, slope_ace2, pval_ace2,
                               lat, lon_360, title_era5, title_ace2, out_path,
                               cbar_label="Regression coeff.  (Δ HHE freq per °C TNA SST)"):
    """Regression of (detrended) gridded JJA HHE frequency onto the detrended
    TNA-mean SST index, ERA5 vs ACE2 side by side, over CONUS."""
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)

    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_180 >= lon_min) & (lon_180 <= lon_max)
    lat_sub = lat[lat_sel]
    lon_sub = lon_180[lon_sel]

    def _sub(a):
        return a[lat_sel, :][:, lon_sel]

    s_era5, p_era5 = _sub(slope_era5), _sub(pval_era5)
    s_ace2, p_ace2 = _sub(slope_ace2), _sub(pval_ace2)

    finite = np.concatenate([s_era5[np.isfinite(s_era5)], s_ace2[np.isfinite(s_ace2)]])
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 0.01
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _render_regression_ax(axes[0], s_era5, p_era5, lon_sub, lat_sub, vmax, title_era5)
    mesh = _render_regression_ax(axes[1], s_ace2, p_ace2, lon_sub, lat_sub, vmax, title_ace2)
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                label=cbar_label)
    fig.suptitle("JJA HHE frequency regressed onto detrended TNA-mean SST index", fontsize=12, y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def main():
    freq_nc = SLIDING_DIR / "jja_seasonal_freqs.nc"
    if not freq_nc.exists():
        print(f"ERROR: {freq_nc} not found. Run seasonal_jja_skill.py first.", flush=True)
        return

    print(f"Loading seasonal frequencies from {freq_nc}", flush=True)
    ds = xr.open_dataset(freq_nc)
    lat_f    = ds["lat"].values
    lon_f    = ds["lon"].values    # 0-360
    ace2_arr = ds["ace2_freq"].values.astype(np.float32)   # (37, 180, 360)
    era5_arr = ds["era5_freq"].values.astype(np.float32)
    ds.close()
    print(f"  Loaded: {ace2_arr.shape}", flush=True)

    # SE-US scalar time series
    era5_seus = bbox_mean(era5_arr, lat_f, lon_f, SEUS_BBOX)
    ace2_seus = bbox_mean(ace2_arr, lat_f, lon_f, SEUS_BBOX)
    print(f"ERA5 SE-US HHE freq: mean={np.nanmean(era5_seus):.3f}", flush=True)
    print(f"ACE2 SE-US HHE freq: mean={np.nanmean(ace2_seus):.3f}", flush=True)

    # Save SE-US freq series
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset({
        "era5_seus_freq": xr.DataArray(era5_seus, dims=["year"], coords={"year": YEARS}),
        "ace2_seus_freq": xr.DataArray(ace2_seus, dims=["year"], coords={"year": YEARS}),
    }).to_netcdf(OUT_DIR / "jja_seus_hhe_freq.nc")

    # Load / compute SST
    sst_arr, sst_lat, sst_lon = load_or_compute_sst()
    print(f"SST: {sst_arr.shape}  lat {sst_lat[0]:.1f}–{sst_lat[-1]:.1f}", flush=True)

    # Detrend SST along year axis
    print("Detrending SST ...", flush=True)
    sst_dt = detrend_along_year(sst_arr)

    # ERA5 correlation
    print("Computing SST × ERA5-HHE correlation ...", flush=True)
    corr_era5, pval_era5 = pearson_corr_map(sst_dt, era5_seus)
    xr.Dataset({"r": xr.DataArray(corr_era5, dims=["lat", "lon"],
                coords={"lat": sst_lat, "lon": sst_lon})}).to_netcdf(
        OUT_DIR / "sst_hhe_corr_era5.nc")

    # ACE2 correlation
    print("Computing SST × ACE2-HHE correlation ...", flush=True)
    corr_ace2, pval_ace2 = pearson_corr_map(sst_dt, ace2_seus)
    xr.Dataset({"r": xr.DataArray(corr_ace2, dims=["lat", "lon"],
                coords={"lat": sst_lat, "lon": sst_lon})}).to_netcdf(
        OUT_DIR / "sst_hhe_corr_ace2.nc")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    yr_range = f"{YEARS[0]}–{YEARS[-1]}"
    thresh_label = "±7-day sliding window thresholds, LOO"

    plot_corr_panel(
        corr_era5, pval_era5, sst_lat, sst_lon,
        f"SST × ERA5 JJA HHE freq (SE-US)  |  Pearson r\n"
        f"{yr_range}  |  {thresh_label}  |  stippling: p < 0.05",
        FIGURES_DIR / "sst_hhe_corr_era5.png",
    )
    plot_corr_panel(
        corr_ace2, pval_ace2, sst_lat, sst_lon,
        f"SST × ACE2 JJA HHE freq (SE-US)  |  Pearson r\n"
        f"{yr_range}  |  {thresh_label}  |  stippling: p < 0.05",
        FIGURES_DIR / "sst_hhe_corr_ace2.png",
    )
    plot_corr_comparison(
        corr_era5, pval_era5, corr_ace2, pval_ace2, sst_lat, sst_lon,
        f"ERA5  |  {yr_range}  |  stippling: p < 0.05",
        f"ACE2  |  {yr_range}  |  stippling: p < 0.05",
        FIGURES_DIR / "sst_hhe_corr_era5_vs_ace2.png",
    )

    # ── TNA-mean SST index → regression onto gridded HHE frequency ──────────
    print("Computing TNA-mean SST index ...", flush=True)
    tna_sst    = bbox_mean(sst_arr, sst_lat, sst_lon, TNA_BBOX)
    tna_sst_dt = detrend_1d(tna_sst)
    print(f"  TNA SST index: mean={np.nanmean(tna_sst):.3f} K  "
          f"detrended std={np.nanstd(tna_sst_dt):.3f}", flush=True)

    print("Detrending gridded HHE frequency fields ...", flush=True)
    era5_freq_dt = detrend_along_year(era5_arr)
    ace2_freq_dt = detrend_along_year(ace2_arr)

    print("Computing regression of HHE freq on TNA SST index ...", flush=True)
    slope_era5, pval_era5_reg = regression_map(era5_freq_dt, tna_sst_dt)
    slope_ace2, pval_ace2_reg = regression_map(ace2_freq_dt, tna_sst_dt)

    xr.Dataset({"slope": xr.DataArray(slope_era5, dims=["lat", "lon"],
                coords={"lat": lat_f, "lon": lon_f}),
                "p_value": xr.DataArray(pval_era5_reg, dims=["lat", "lon"],
                coords={"lat": lat_f, "lon": lon_f})}).to_netcdf(
        OUT_DIR / "hhe_freq_on_tna_sst_regression_era5.nc")
    xr.Dataset({"slope": xr.DataArray(slope_ace2, dims=["lat", "lon"],
                coords={"lat": lat_f, "lon": lon_f}),
                "p_value": xr.DataArray(pval_ace2_reg, dims=["lat", "lon"],
                coords={"lat": lat_f, "lon": lon_f})}).to_netcdf(
        OUT_DIR / "hhe_freq_on_tna_sst_regression_ace2.nc")

    plot_regression_comparison(
        slope_era5, pval_era5_reg, slope_ace2, pval_ace2_reg, lat_f, lon_f,
        f"ERA5  |  {yr_range}  |  stippling: p < 0.05",
        f"ACE2  |  {yr_range}  |  stippling: p < 0.05",
        FIGURES_DIR / "hhe_freq_on_tna_sst_regression_era5_vs_ace2.png",
    )

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
