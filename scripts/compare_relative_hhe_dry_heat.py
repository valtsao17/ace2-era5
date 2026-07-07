#!/usr/bin/env python3
"""Compare relative humid-heat extremes with relative dry-heat extremes.

The array-based diagnostics use the seasonal-frequency NetCDFs created by:

  scripts/relative_hhe_jja_skill.py
  scripts/seasonal_jja_skill.py

and, when available, the SST-correlation NetCDFs created by:

  scripts/relative_hhe_sst_rankcorr_box_pipeline.py
  scripts/sst_raw_rankcorr_box_pipeline.py

If those NetCDFs are not present, ``--png-only`` compares the two already
rendered SST-correlation figures as an image-level sanity check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPL_DIR = PROJECT_ROOT / "tmp" / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import numpy as np
import xarray as xr
from PIL import Image
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import sst_raw_rankcorr_box_pipeline as raw_pipe  # noqa: E402
import relative_hhe_sst_rankcorr_box_pipeline as hhe_pipe  # noqa: E402
from seasonal_jja_skill import load_land_mask  # noqa: E402
from sst_teleconnection_jja_sliding7d import bbox_mean  # noqa: E402

DEFAULT_RAW_FREQ = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
DEFAULT_RAW_SKILL = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/skill_jja_seasonal.nc"
DEFAULT_HHE_FREQ = PROJECT_ROOT / "outputs/lag_may/relative_hhe_jja_sliding7d/jja_seasonal_freqs.nc"
DEFAULT_HHE_SKILL = PROJECT_ROOT / "outputs/lag_may/relative_hhe_jja_sliding7d/skill_jja_seasonal.nc"
DEFAULT_RAW_SST = PROJECT_ROOT / "outputs/lag_may/sst_raw_rankcorr_box/sst_raw_jja_rankcorrbox_corr.nc"
DEFAULT_HHE_SST = PROJECT_ROOT / "outputs/lag_may/relative_hhe_sst_rankcorr_box/sst_relhhe_jja_rankcorrbox_corr.nc"
DEFAULT_OUT = PROJECT_ROOT / "outputs/lag_may/hhe_vs_dry_heat_diagnostics"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
    "font.size": 10,
})

_COAST_GEOMS = None


def parse_years(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def zscore_detrended(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if mask.sum() < 3:
        return out
    vals = signal.detrend(x[mask], type="linear")
    sd = np.nanstd(vals, ddof=1)
    if not np.isfinite(sd) or sd <= 0.0:
        return out
    out[mask] = (vals - np.nanmean(vals)) / sd
    return out


def corr_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    out: dict[str, float | int] = {"n": int(mask.sum())}
    if mask.sum() < 3:
        return out
    xx = x[mask]
    yy = y[mask]
    pear = stats.pearsonr(xx, yy)
    spear = stats.spearmanr(xx, yy)
    kend = stats.kendalltau(xx, yy)
    out.update({
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "kendall_tau": float(kend.statistic),
        "kendall_p": float(kend.pvalue),
        "z_rmse": float(np.sqrt(np.nanmean((xx - yy) ** 2))),
    })
    return out


def _get_coast():
    global _COAST_GEOMS
    if _COAST_GEOMS is None:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        _COAST_GEOMS = list(shpreader.Reader(shp).geometries())
    return _COAST_GEOMS


def _draw_coast(ax, xlim, ylim):
    from shapely.geometry import box

    vp = box(xlim[0], ylim[0], xlim[1], ylim[1])

    def plot_geom(geom):
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                plot_geom(g)
        elif hasattr(geom, "exterior"):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color="0.15", lw=0.45, zorder=5)
            for ring in geom.interiors:
                xs, ys = ring.xy
                ax.plot(xs, ys, color="0.15", lw=0.45, zorder=5)

    for geom in _get_coast():
        try:
            plot_geom(geom.intersection(vp))
        except Exception:
            continue


def lon_to_180(lon_360: np.ndarray) -> np.ndarray:
    return np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)


def roll_to_180(field: np.ndarray, lon_360: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split = int(np.searchsorted(lon_360, 180.0))
    lon_r = np.concatenate([lon_360[split:] - 360.0, lon_360[:split]])
    field_r = np.roll(field, len(lon_360) - split, axis=-1)
    return field_r, lon_r


def map_panel(ax, field, lat, lon, title, *, vmin=-1.0, vmax=1.0, cmap="RdBu_r", extent=None):
    field_r, lon_r = roll_to_180(np.asarray(field), np.asarray(lon))
    if extent is None:
        extent = (float(lon_r[0]), float(lon_r[-1]), float(lat[0]), float(lat[-1]))
    lon_min, lon_max, lat_min, lat_max = extent
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_r >= lon_min) & (lon_r <= lon_max)
    xx, yy = np.meshgrid(lon_r[lon_sel], lat[lat_sel])
    mesh = ax.pcolormesh(
        xx, yy, field_r[np.ix_(lat_sel, lon_sel)],
        shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    _draw_coast(ax, (lon_min, lon_max), (lat_min, lat_max))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(np.ceil(lon_min / 30) * 30, lon_max + 1, 30))
    ax.set_yticks(np.arange(np.ceil(lat_min / 20) * 20, lat_max + 1, 20))
    ax.tick_params(labelsize=8)
    return mesh


def area_masked_index(freq: np.ndarray, lat, lon, box, mask) -> np.ndarray:
    arr = np.where(mask[np.newaxis, :, :], freq, np.nan)
    return bbox_mean(arr.astype(np.float32), lat, lon, box).astype(np.float64)


def intersection_box(a, b):
    lat0 = max(a[0], b[0])
    lat1 = min(a[1], b[1])
    lon0 = max(a[2], b[2])
    lon1 = min(a[3], b[3])
    if lat1 > lat0 and lon1 > lon0:
        return (lat0, lat1, lon0, lon1)
    return None


def union_box(a, b):
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def box_json(box) -> dict[str, list[float]]:
    lat0, lat1, lon0, lon1 = box
    return {
        "latN": [round(float(lat0), 2), round(float(lat1), 2)],
        "lonW": [round(float(360.0 - lon1), 2), round(float(360.0 - lon0), 2)],
    }


def select_rankcorr_boxes(raw_skill, hhe_skill, lat, lon, land):
    lat_c = lat[raw_pipe.CONUS_LAT_SLICE]
    lon_c = lon[raw_pipe.CONUS_LON_SLICE]
    land_c = land[raw_pipe.CONUS_LAT_SLICE, raw_pipe.CONUS_LON_SLICE]

    try:
        allowed = raw_pipe._country_mask(lat, lon, raw_pipe.ALLOWED_RAW_BOX_COUNTRIES) & land
    except Exception as exc:
        print(f"WARNING: US/Canada country mask failed ({exc}); falling back to all land.", flush=True)
        allowed = land
    allowed_c = allowed[raw_pipe.CONUS_LAT_SLICE, raw_pipe.CONUS_LON_SLICE]

    raw_tau = raw_skill["kendall_tau"].values.astype(np.float32)
    raw_p = raw_skill["tau_p_value"].values.astype(np.float32)
    hhe_tau = hhe_skill["kendall_tau"].values.astype(np.float32)
    hhe_p = hhe_skill["tau_p_value"].values.astype(np.float32)

    raw_box_info = raw_pipe.find_rankcorr_box(
        raw_tau[raw_pipe.CONUS_LAT_SLICE, raw_pipe.CONUS_LON_SLICE],
        raw_p[raw_pipe.CONUS_LAT_SLICE, raw_pipe.CONUS_LON_SLICE],
        lat_c, lon_c, land_c, allowed_c,
    )
    hhe_box_info = hhe_pipe.find_rankcorr_box(
        hhe_tau[hhe_pipe.CONUS_LAT_SLICE, hhe_pipe.CONUS_LON_SLICE],
        hhe_p[hhe_pipe.CONUS_LAT_SLICE, hhe_pipe.CONUS_LON_SLICE],
        lat_c, lon_c, land_c, allowed_c,
    )

    raw_selected = raw_box_info["bbox"]
    hhe_selected = hhe_box_info["bbox"]
    raw_box = raw_pipe.expand_box_on_grid(raw_selected, lat_c, lon_c)
    hhe_box = hhe_pipe.expand_box_on_grid(hhe_selected, lat_c, lon_c)
    common_box = intersection_box(raw_box, hhe_box) or union_box(raw_box, hhe_box)
    return {
        "raw_selected": raw_selected,
        "hhe_selected": hhe_selected,
        "raw_box": raw_box,
        "hhe_box": hhe_box,
        "common_box": common_box,
        "common_box_type": "intersection" if intersection_box(raw_box, hhe_box) else "union",
        "allowed_land": allowed,
    }


def temporal_corr_map(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_year, nlat, nlon = a.shape
    r = np.full((nlat, nlon), np.nan, dtype=np.float32)
    p = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            x = zscore_detrended(a[:, i, j])
            y = zscore_detrended(b[:, i, j])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < min(10, n_year):
                continue
            res = stats.pearsonr(x[mask], y[mask])
            r[i, j] = res.statistic
            p[i, j] = res.pvalue
    return r, p


def plot_index_timeseries(years, indices, stats_summary, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("era5_common", "ERA5, common box"),
        ("ace2_common", "ACE2, common box"),
        ("era5_own", "ERA5, own boxes"),
        ("ace2_own", "ACE2, own boxes"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        raw_z = zscore_detrended(indices[key]["dry"])
        hhe_z = zscore_detrended(indices[key]["hhe"])
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.plot(years, raw_z, color="#1f77b4", lw=1.8, marker="o", ms=3.5, label="dry heat")
        ax.plot(years, hhe_z, color="#d62728", lw=1.8, marker="s", ms=3.2, label="relative HHE")
        s = stats_summary[key]
        label = f"r={s.get('pearson_r', np.nan):.2f}, tau={s.get('kendall_tau', np.nan):.2f}"
        ax.text(0.02, 0.92, label, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9, pad=3))
        ax.set_title(title)
        ax.set_ylabel("detrended z-score")
        ax.grid(True, lw=0.35, alpha=0.35)
    axes[0, 0].legend(loc="lower left", fontsize=9, frameon=True)
    axes[-1, 0].set_xlabel("year")
    axes[-1, 1].set_xlabel("year")
    fig.suptitle("Annual regional frequency indices: dry heat vs relative HHE", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_index_scatter(indices, stats_summary, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    panels = [
        ("era5_common", "ERA5, common box"),
        ("ace2_common", "ACE2, common box"),
        ("era5_own", "ERA5, own boxes"),
        ("ace2_own", "ACE2, own boxes"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        x = zscore_detrended(indices[key]["dry"])
        y = zscore_detrended(indices[key]["hhe"])
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=42, color="#444444", alpha=0.82)
        if mask.sum() >= 2:
            lim = np.nanmax(np.abs(np.concatenate([x[mask], y[mask]])))
            lim = max(float(lim), 1.0)
            ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.0, ls="--")
            reg = stats.linregress(x[mask], y[mask])
            xx = np.linspace(-lim, lim, 50)
            ax.plot(xx, reg.intercept + reg.slope * xx, color="#d62728", lw=1.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        s = stats_summary[key]
        ax.text(0.04, 0.92, f"r={s.get('pearson_r', np.nan):.2f}\nRMSE={s.get('z_rmse', np.nan):.2f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9, pad=3))
        ax.set_title(title)
        ax.set_xlabel("dry heat detrended z-score")
        ax.set_ylabel("relative HHE detrended z-score")
        ax.grid(True, lw=0.35, alpha=0.35)
    fig.suptitle("One-to-one test of the annual regional indices", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gridcell_frequency_corr(lat, lon, era5_r, ace2_r, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), constrained_layout=True)
    mesh = map_panel(
        axes[0], era5_r, lat, lon,
        "ERA5: corr(year-to-year relative HHE freq, dry freq)",
        extent=(-170, -40, 10, 75),
    )
    mesh = map_panel(
        axes[1], ace2_r, lat, lon,
        "ACE2: corr(year-to-year relative HHE freq, dry freq)",
        extent=(-170, -40, 10, 75),
    )
    fig.colorbar(mesh, ax=axes, shrink=0.75, label="Pearson r after detrending")
    fig.suptitle("Grid-cell temporal similarity of the two extreme definitions", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def spatial_stats(a, b) -> dict[str, float | int]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    out: dict[str, float | int] = {"n_cells": int(mask.sum())}
    if mask.sum() < 3:
        return out
    aa = a[mask]
    bb = b[mask]
    out.update({
        "spatial_r": float(stats.pearsonr(aa, bb).statistic),
        "mean_abs_difference": float(np.nanmean(np.abs(aa - bb))),
        "median_abs_difference": float(np.nanmedian(np.abs(aa - bb))),
        "sign_agreement_fraction": float(np.mean(np.sign(aa) == np.sign(bb))),
    })
    return out


def compare_sst_corr_maps(raw_sst_path: Path, hhe_sst_path: Path, out_dir: Path) -> dict:
    if not raw_sst_path.exists() or not hhe_sst_path.exists():
        return {"available": False, "missing": [str(p) for p in (raw_sst_path, hhe_sst_path) if not p.exists()]}

    with xr.open_dataset(raw_sst_path) as raw_ds, xr.open_dataset(hhe_sst_path) as hhe_ds:
        lat = raw_ds["lat"].values.astype(np.float32)
        lon = raw_ds["lon"].values.astype(np.float32)
        if not (np.allclose(lat, hhe_ds["lat"].values) and np.allclose(lon, hhe_ds["lon"].values)):
            raise ValueError("SST-correlation grids differ; regrid before comparing.")
        fields = {
            "era5": (raw_ds["corr_era5"].values, hhe_ds["corr_era5"].values),
            "ace2": (raw_ds["corr_ace2"].values, hhe_ds["corr_ace2"].values),
        }

    summary = {"available": True}
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    for row, (label, (raw_corr, hhe_corr)) in enumerate(fields.items()):
        diff = hhe_corr - raw_corr
        summary[label] = spatial_stats(raw_corr, hhe_corr)
        title = f"{label.upper()}: relative HHE minus dry-heat SST-correlation"
        mesh = map_panel(axes[row, 0], diff, lat, lon, title, vmin=-0.5, vmax=0.5, extent=(-170, -40, -25, 70))
        ax = axes[row, 1]
        mask = np.isfinite(raw_corr) & np.isfinite(hhe_corr)
        x = raw_corr[mask]
        y = hhe_corr[mask]
        if x.size > 20000:
            rng = np.random.default_rng(0)
            idx = rng.choice(x.size, size=20000, replace=False)
            x = x[idx]
            y = y[idx]
        ax.scatter(x, y, s=5, alpha=0.18, color="#333333", linewidths=0)
        ax.plot([-1, 1], [-1, 1], color="0.55", lw=1.0, ls="--")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("dry-heat SST correlation r")
        ax.set_ylabel("relative-HHE SST correlation r")
        s = summary[label]
        ax.text(0.04, 0.92, f"spatial r={s.get('spatial_r', np.nan):.2f}\nMAD={s.get('mean_abs_difference', np.nan):.2f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9, pad=3))
        ax.set_title(f"{label.upper()}: pixelwise map correspondence")
        ax.grid(True, lw=0.35, alpha=0.35)
    fig.colorbar(mesh, ax=axes[:, 0], shrink=0.8, label="Delta r (relative HHE - dry heat)")
    fig.suptitle("Do the SST teleconnection maps change when dry heat is replaced by relative HHE?", fontsize=13)
    out = out_dir / "hhe_vs_dry_sstcorr_map_difference.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = str(out)
    return summary


def run_array_diagnostics(args) -> dict:
    required = [args.raw_freq, args.raw_skill, args.hhe_freq, args.hhe_skill]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required NetCDF inputs:\n  " + "\n  ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.raw_freq) as raw_freq_ds, xr.open_dataset(args.hhe_freq) as hhe_freq_ds:
        raw_years = [int(y) for y in raw_freq_ds["year"].values]
        hhe_years = [int(y) for y in hhe_freq_ds["year"].values]
        wanted = parse_years(args.years)
        years = sorted(set(raw_years) & set(hhe_years))
        if wanted is not None:
            years = [y for y in wanted if y in years]
        if len(years) < 3:
            raise ValueError(f"Need at least three overlapping years; found {years}")
        raw_idx = [raw_years.index(y) for y in years]
        hhe_idx = [hhe_years.index(y) for y in years]

        lat = raw_freq_ds["lat"].values.astype(np.float32)
        lon = raw_freq_ds["lon"].values.astype(np.float32)
        if not (np.allclose(lat, hhe_freq_ds["lat"].values) and np.allclose(lon, hhe_freq_ds["lon"].values)):
            raise ValueError("Frequency grids differ; regrid before comparing.")
        raw_era5 = raw_freq_ds["era5_freq"].isel(year=raw_idx).values.astype(np.float32)
        raw_ace2 = raw_freq_ds["ace2_freq"].isel(year=raw_idx).values.astype(np.float32)
        hhe_era5 = hhe_freq_ds["era5_freq"].isel(year=hhe_idx).values.astype(np.float32)
        hhe_ace2 = hhe_freq_ds["ace2_freq"].isel(year=hhe_idx).values.astype(np.float32)

    with xr.open_dataset(args.raw_skill) as raw_skill, xr.open_dataset(args.hhe_skill) as hhe_skill:
        land = load_land_mask(lat, lon)
        if land is None:
            land = np.isfinite(np.nanmean(raw_era5, axis=0))
        boxes = select_rankcorr_boxes(raw_skill, hhe_skill, lat, lon, land)

    mask = boxes["allowed_land"]
    common = boxes["common_box"]
    raw_box = boxes["raw_box"]
    hhe_box = boxes["hhe_box"]

    indices = {
        "era5_common": {
            "dry": area_masked_index(raw_era5, lat, lon, common, mask),
            "hhe": area_masked_index(hhe_era5, lat, lon, common, mask),
        },
        "ace2_common": {
            "dry": area_masked_index(raw_ace2, lat, lon, common, mask),
            "hhe": area_masked_index(hhe_ace2, lat, lon, common, mask),
        },
        "era5_own": {
            "dry": area_masked_index(raw_era5, lat, lon, raw_box, mask),
            "hhe": area_masked_index(hhe_era5, lat, lon, hhe_box, mask),
        },
        "ace2_own": {
            "dry": area_masked_index(raw_ace2, lat, lon, raw_box, mask),
            "hhe": area_masked_index(hhe_ace2, lat, lon, hhe_box, mask),
        },
    }

    stats_summary = {
        key: corr_stats(zscore_detrended(vals["dry"]), zscore_detrended(vals["hhe"]))
        for key, vals in indices.items()
    }

    fig_ts = args.out_dir / "hhe_vs_dry_index_timeseries.png"
    fig_scatter = args.out_dir / "hhe_vs_dry_index_scatter.png"
    plot_index_timeseries(np.asarray(years), indices, stats_summary, fig_ts)
    plot_index_scatter(indices, stats_summary, fig_scatter)

    era5_grid_r, era5_grid_p = temporal_corr_map(hhe_era5, raw_era5)
    ace2_grid_r, ace2_grid_p = temporal_corr_map(hhe_ace2, raw_ace2)
    fig_grid = args.out_dir / "hhe_vs_dry_gridcell_temporal_corr.png"
    plot_gridcell_frequency_corr(lat, lon, era5_grid_r, ace2_grid_r, fig_grid)

    grid_summary = {
        "era5_median_land_r": float(np.nanmedian(np.where(mask, era5_grid_r, np.nan))),
        "ace2_median_land_r": float(np.nanmedian(np.where(mask, ace2_grid_r, np.nan))),
        "era5_sig_land_fraction": float(np.nanmean(np.where(mask & np.isfinite(era5_grid_p), era5_grid_p < 0.05, np.nan))),
        "ace2_sig_land_fraction": float(np.nanmean(np.where(mask & np.isfinite(ace2_grid_p), ace2_grid_p < 0.05, np.nan))),
    }

    sst_summary = compare_sst_corr_maps(args.raw_sst_corr, args.hhe_sst_corr, args.out_dir)

    summary = {
        "years": [int(years[0]), int(years[-1])],
        "n_years": len(years),
        "boxes": {
            "dry_heat_box": box_json(raw_box),
            "relative_hhe_box": box_json(hhe_box),
            "common_box": box_json(common),
            "common_box_type": boxes["common_box_type"],
        },
        "regional_index_correlations": stats_summary,
        "gridcell_frequency_correlations": grid_summary,
        "sst_correlation_map_comparison": sst_summary,
        "figures": {
            "index_timeseries": str(fig_ts),
            "index_scatter": str(fig_scatter),
            "gridcell_temporal_corr": str(fig_grid),
        },
    }
    out_json = args.out_dir / "hhe_vs_dry_heat_diagnostic_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {fig_ts}", flush=True)
    print(f"wrote {fig_scatter}", flush=True)
    print(f"wrote {fig_grid}", flush=True)
    return summary


def _crop_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "era5": (int(0.030 * width), int(0.105 * height), int(0.462 * width), int(0.963 * height)),
        "ace2": (int(0.500 * width), int(0.105 * height), int(0.935 * width), int(0.963 * height)),
    }


def color_signal(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    # Approximate the diverging map coordinate: red positive, blue negative.
    return arr[..., 0] - arr[..., 2]


def png_panel_stats(hhe_rgb: np.ndarray, raw_rgb: np.ndarray) -> dict[str, float | int]:
    hsig = color_signal(hhe_rgb)
    rsig = color_signal(raw_rgb)
    mean_h = hhe_rgb.mean(axis=-1)
    mean_r = raw_rgb.mean(axis=-1)
    dark = (mean_h < 45.0) | (mean_r < 45.0)
    alpha = np.isfinite(hsig) & np.isfinite(rsig) & ~dark
    x = rsig[alpha].ravel().astype(np.float64)
    y = hsig[alpha].ravel().astype(np.float64)
    if x.size < 3:
        return {"n_pixels": int(x.size)}
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    r = float(np.sum(xm * ym) / denom) if denom > 0 else np.nan
    return {
        "n_pixels": int(x.size),
        "color_signal_r": r,
        "mean_abs_color_signal_difference": float(np.mean(np.abs(y - x))),
        "rgb_mean_abs_difference_0_255": float(np.mean(np.abs(hhe_rgb[alpha].astype(float) - raw_rgb[alpha].astype(float)))),
    }


def run_png_diagnostics(args) -> dict:
    if args.hhe_png is None or args.dry_png is None:
        raise ValueError("--png-only requires --hhe-png and --dry-png")
    if not args.hhe_png.exists():
        raise FileNotFoundError(args.hhe_png)
    if not args.dry_png.exists():
        raise FileNotFoundError(args.dry_png)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hhe_img = Image.open(args.hhe_png).convert("RGB")
    raw_img = Image.open(args.dry_png).convert("RGB")
    if hhe_img.size != raw_img.size:
        raw_img = raw_img.resize(hhe_img.size, Image.Resampling.BICUBIC)
    w, h = hhe_img.size
    boxes = _crop_boxes(w, h)
    summary = {
        "available": True,
        "hhe_png": str(args.hhe_png),
        "dry_png": str(args.dry_png),
        "image_size": [w, h],
        "panels": {},
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
    for row, label in enumerate(("era5", "ace2")):
        crop = boxes[label]
        hhe_crop = np.asarray(hhe_img.crop(crop))
        raw_crop = np.asarray(raw_img.crop(crop))
        stats_one = png_panel_stats(hhe_crop, raw_crop)
        summary["panels"][label] = stats_one

        axes[row, 0].imshow(raw_crop)
        axes[row, 0].set_title(f"{label.upper()} dry-heat figure crop")
        axes[row, 1].imshow(hhe_crop)
        axes[row, 1].set_title(f"{label.upper()} relative-HHE figure crop")
        diff = np.mean(np.abs(hhe_crop.astype(float) - raw_crop.astype(float)), axis=-1)
        im = axes[row, 2].imshow(diff, cmap="magma", vmin=0, vmax=np.percentile(diff, 99))
        axes[row, 2].set_title(
            f"{label.upper()} abs RGB difference\n"
            f"color-signal r={stats_one.get('color_signal_r', np.nan):.3f}"
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, 2], shrink=0.8, label="mean abs RGB difference")
    fig.suptitle("Image-only comparison of the two SST-correlation figures", fontsize=13)
    out = args.out_dir / "png_sstcorr_figure_similarity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = str(out)

    out_json = args.out_dir / "png_sstcorr_figure_similarity_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}", flush=True)
    print(f"wrote {out_json}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-freq", type=Path, default=DEFAULT_RAW_FREQ)
    p.add_argument("--raw-skill", type=Path, default=DEFAULT_RAW_SKILL)
    p.add_argument("--hhe-freq", type=Path, default=DEFAULT_HHE_FREQ)
    p.add_argument("--hhe-skill", type=Path, default=DEFAULT_HHE_SKILL)
    p.add_argument("--raw-sst-corr", type=Path, default=DEFAULT_RAW_SST)
    p.add_argument("--hhe-sst-corr", type=Path, default=DEFAULT_HHE_SST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--years", default=None, help="'START:END' or comma-separated years; default is overlap.")
    p.add_argument("--png-only", action="store_true", help="Only compare the already rendered PNG figures.")
    p.add_argument("--also-png", action="store_true", help="Run PNG comparison after array diagnostics.")
    p.add_argument("--hhe-png", type=Path, default=None)
    p.add_argument("--dry-png", type=Path, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.png_only:
        run_png_diagnostics(args)
        return
    run_array_diagnostics(args)
    if args.also_png:
        run_png_diagnostics(args)


if __name__ == "__main__":
    main()
