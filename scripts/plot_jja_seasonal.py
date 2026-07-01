#!/usr/bin/env python3
"""Generate JJA seasonal skill PNG maps from existing skill_jja_seasonal.nc.

Uses plain matplotlib axes + shapely coastlines (same as DJF plots) to avoid
cartopy GeoAxes segfault on this system.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NC_PATH = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/skill_jja_seasonal.nc"
OUT_DIR = NC_PATH.parent

_SKILL_CMAP = LinearSegmentedColormap.from_list(
    "skill",
    ["#ffffff", "#fff2b0", "#ffcc55", "#ff8800", "#cc2200", "#780000"],
    N=256,
)
_SKILL_CMAP.set_bad("white")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "lightblue",
                     "font.size": 10, "savefig.dpi": 180, "savefig.bbox": "tight"})

# Cached border geometries
_COAST_GEOMS = None


def _get_coast():
    global _COAST_GEOMS
    if _COAST_GEOMS is None:
        shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        _COAST_GEOMS = list(shpreader.Reader(shp).geometries())
    return _COAST_GEOMS


def _draw_coast(ax):
    """Draw land outlines clipped to current xlim/ylim using shapely."""
    from shapely.geometry import box
    xl, xr_ = ax.get_xlim()
    yb, yt = ax.get_ylim()
    vp = box(xl, yb, xr_, yt)
    geoms = _get_coast()

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

    for geom in geoms:
        try:
            _plot(geom.intersection(vp), 0.4)
        except Exception:
            continue


def cos_lat_mean(field: np.ndarray, lat: np.ndarray) -> float:
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(field)
    return float(np.nansum(field * w * valid) / np.nansum(w * valid))


def _roll_to_180(field, lon):
    """Re-center data from 0-360 to -180/180."""
    # lon: [0.5, 1.5, ..., 179.5, 180.5, ..., 359.5]
    n = len(lon)
    split = np.searchsorted(lon, 180.0)          # first index where lon >= 180
    lon_shifted = np.concatenate([lon[split:] - 360.0, lon[:split]])
    if field.ndim == 2:
        field_shifted = np.roll(field, n - split, axis=-1)
    else:
        field_shifted = np.roll(field, n - split, axis=-1)
    return field_shifted, lon_shifted


def plot_global(field, pval, lat, lon, title, out_path, metric_label):
    signed = "Kendall" in metric_label
    plot_field = field if signed else np.abs(field)
    print(f"Plotting {out_path.name} ...", flush=True)

    # Roll 0-360 → -180/180
    plot_field_r, lon_r = _roll_to_180(plot_field, lon)
    pval_r, _          = _roll_to_180(pval, lon)
    field_r, _         = _roll_to_180(field, lon)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#d0e8f0")   # ocean color

    # pcolormesh on plain axes
    LON2D, LAT2D = np.meshgrid(lon_r, lat)
    cmap = "RdBu_r" if signed else _SKILL_CMAP
    vmin, vmax = (-1.0, 1.0) if signed else (0.0, 0.6)
    mesh = ax.pcolormesh(LON2D, LAT2D, plot_field_r,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="nearest", zorder=1)
    print("  pcolormesh done", flush=True)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    print("  drawing coastlines ...", flush=True)
    _draw_coast(ax)
    print("  coastlines done", flush=True)

    # Stipple: not significant
    not_sig = (pval_r > 0.05) & np.isfinite(field_r)
    n_stipple = int(not_sig.sum())
    print(f"  stippling {n_stipple} points ...", flush=True)
    if n_stipple > 40000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_stipple, size=40000, replace=False)
        xs = LON2D[not_sig].ravel()[idx]
        ys = LAT2D[not_sig].ravel()[idx]
    else:
        xs = LON2D[not_sig].ravel()
        ys = LAT2D[not_sig].ravel()
    ax.plot(xs, ys, "k.", markersize=0.8, alpha=0.4, zorder=4, linewidth=0)
    print("  stipple done", flush=True)

    # Colorbar, annotations
    cbar_label = metric_label if signed else f"|{metric_label}|"
    fig.colorbar(mesh, ax=ax, shrink=0.7, label=cbar_label)
    wavg = cos_lat_mean(plot_field, lat)
    mean_label = f"cos-lat mean {cbar_label}"
    ax.text(0.01, 0.03, f"{mean_label} = {wavg:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

    # Tick labels
    ax.set_xticks(range(-180, 181, 60))
    ax.set_xticklabels(["180°", "120°W", "60°W", "0°", "60°E", "120°E", "180°"], fontsize=8)
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("  saving ...", flush=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def main():
    print(f"Loading {NC_PATH}", flush=True)
    ds = xr.open_dataset(NC_PATH)
    lat = ds["lat"].values
    lon = ds["lon"].values
    r_map    = ds["pearson_r"].values
    r_pval   = ds["r_p_value"].values
    tau_map  = ds["kendall_tau"].values
    tau_pval = ds["tau_p_value"].values
    ds.close()
    print(f"Grid: {len(lat)} lat × {len(lon)} lon", flush=True)

    yr_range     = "1980–2016  (n=37 seasons)"
    thresh_label = "±7-day sliding window thresholds"

    plot_global(
        r_map, r_pval, lat, lon,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Pearson r\n"
        f"{yr_range}  |  {thresh_label}",
        OUT_DIR / "pearsonr_jja_seasonal_global.png",
        metric_label="Pearson r",
    )
    plot_global(
        tau_map, tau_pval, lat, lon,
        f"ACE2-ERA5  |  JJA heat-extreme frequency skill  |  Kendall τ\n"
        f"{yr_range}  |  {thresh_label}",
        OUT_DIR / "tau_jja_seasonal_global.png",
        metric_label="Kendall τ",
    )

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
