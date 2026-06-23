#!/usr/bin/env python3
"""Two ERA5-vs-ACE2 seasonal extreme-frequency figures over CONUS+Mexico+Canada.

Figure 1 — RAW heat extreme: fraction of JJA days above the day-of-year 90th
           percentile (no leave-one-year-out).   src jja_seasonal_freqs.nc
Figure 2 — HHE: fraction of JJA days with NOAA Heat Index >= 105 F (paper,
           absolute threshold).                  src jja_hi_freq_{era5,ace2}.nc

Both are whole-season frequencies (extreme days / JJA days), climatology = mean
over the 37 years.  Left panel ERA5, right panel ACE2, common colour scale per
figure.  A bounding box around the region of highest ERA5 extreme concentration
is drawn ON THE ERA5 PANEL ONLY (per-figure, ERA5-defined).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
DRY_NC  = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
OUT_DIR = HHE_DIR
CMAP    = LinearSegmentedColormap.from_list(   # pure white at 0 -> dark red at max
    "wred", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#a50f15"])
BOX_LAT_SPAN = 8.0     # degrees — compact hotspot box
BOX_LON_SPAN = 12.0


def find_box(clim, lat, lon):
    """Slide a fixed BOX_LAT_SPAN×BOX_LON_SPAN box; return the position maximizing
    mean climatological extreme frequency (ERA5)."""
    best = None
    for la0 in lat:
        la1 = la0 + BOX_LAT_SPAN
        if la1 > lat[-1]:
            continue
        sel_la = (lat >= la0) & (lat <= la1)
        for lo0 in lon:
            lo1 = lo0 + BOX_LON_SPAN
            if lo1 > lon[-1]:
                continue
            sel_lo = (lon >= lo0) & (lon <= lo1)
            sub = clim[np.ix_(sel_la, sel_lo)]
            if not np.isfinite(sub).any():
                continue
            score = float(np.nanmean(sub))
            if best is None or score > best[0]:
                best = (score, (float(la0), float(la1), float(lo0), float(lo1)))
    return best[1], best[0]


def _panel(ax, field_pct, lat, lon, title, vmax, box=None):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    ax.set_facecolor("#eef1f4")
    im = ax.imshow(field_pct, origin="lower", extent=extent, aspect="equal",
                   vmin=0.0, vmax=vmax, cmap=CMAP, interpolation="nearest", zorder=1)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    if box is not None:
        la0, la1, lo0, lo1 = box
        x0 = lo0 - 360.0 if lo0 > 180 else lo0
        x1 = lo1 - 360.0 if lo1 > 180 else lo1
        ax.add_patch(Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="#00b050", linewidth=2.6, zorder=5))
    ax.set_title(title, fontsize=11)
    return im


def make_figure(era5, ace2, lat, lon, title, cbar_label, out_png, draw_box=False):
    e_clim = np.nanmean(era5, axis=0) * 100.0   # -> percent of JJA days
    a_clim = np.nanmean(ace2, axis=0) * 100.0
    e = e_clim[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    a = a_clim[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    latc = lat[CONUS_LAT_SLICE]
    lonc = lon[CONUS_LON_SLICE]

    box = None
    if draw_box:
        box, box_score = find_box(e, latc, lonc)
        la0, la1, lo0, lo1 = box
        print(f"  {out_png.name}: ERA5 box lat {la0:.1f}-{la1:.1f} "
              f"lon {lo0:.1f}-{lo1:.1f}E ({360-lo1:.0f}-{360-lo0:.0f}°W)  "
              f"mean={box_score:.2f}% (domain mean {np.nanmean(e):.2f}%)", flush=True)

    vmax = float(np.nanpercentile(np.concatenate([e.ravel(), a.ravel()]), 99.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    _panel(axes[0], e, latc, lonc, "ERA5", vmax, box=box)
    im = _panel(axes[1], a, latc, lonc, "ACE2", vmax, box=None)
    fig.suptitle(title, fontsize=13, y=0.99)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.12, shrink=0.5, aspect=40)
    cbar.set_label(cbar_label)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def main():
    with xr.open_dataset(DRY_NC) as d:
        lat = d["lat"].values
        lon = d["lon"].values
        era5_raw = d["era5_freq"].values.astype(np.float32)
        ace2_raw = d["ace2_freq"].values.astype(np.float32)
    make_figure(
        era5_raw, ace2_raw, lat, lon,
        "Raw heat extreme (JJA days > 90th-pct TMP2m), seasonal frequency",
        "% of JJA days above 90th percentile",
        OUT_DIR / "fig1_raw_extreme_freq_panels.png",
    )

    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values
        lon = d["lon"].values
        era5_hhe = d["era5_hi_freq"].values.astype(np.float32)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2_hhe = d["ace2_hi_freq"].values.astype(np.float32)
    make_figure(
        era5_hhe, ace2_hhe, lat, lon,
        "Humid heat extreme (JJA days with Heat Index ≥ 105°F), seasonal frequency",
        "% of JJA days with HI ≥ 105°F",
        OUT_DIR / "fig2_hhe_freq_panels.png",
        draw_box=True,
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
