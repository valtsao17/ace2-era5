#!/usr/bin/env python3
"""Plain climatological JJA HHE frequency maps, no clustering, no skill metric.

Just "how often was each grid cell an HHE day, on average over 1980-2016" for
ERA5 and ACE2 side by side — descriptive companion to the tau/BSS/precision/
recall skill maps, which all answer "how well does ACE2 track ERA5" rather
than "how often does this actually happen."

Reuses the already-cached jja_seasonal_freqs.nc (no recomputation needed).

Outputs -> outputs/lag_may/seasonal_jja_sliding7d/jja_hhe_freq_era5_vs_ace2.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seasonal_jja_skill import cos_lat_mean, _roll_to_180, _draw_coast, _SKILL_CMAP

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"


def _panel(ax, field, lat, lon, title, vmax):
    field_r, lon_r = _roll_to_180(field, lon)
    LON2D, LAT2D = np.meshgrid(lon_r, lat)
    ax.set_facecolor("#d0e8f0")
    mesh = ax.pcolormesh(LON2D, LAT2D, field_r, cmap=_SKILL_CMAP, vmin=0.0, vmax=vmax,
                         shading="nearest", zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    _draw_coast(ax)
    full_mean = cos_lat_mean(field, lat)
    ax.text(0.01, 0.03, f"cos-lat mean freq = {full_mean:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
    ax.set_xticks(range(-180, 181, 60))
    ax.set_xticklabels(["180°", "120°W", "60°W", "0°", "60°E", "120°E", "180°"], fontsize=8)
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=10)
    return mesh


def main():
    freq_nc = OUT_DIR / "jja_seasonal_freqs.nc"
    with xr.open_dataset(freq_nc) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        era5_mean = ds["era5_freq"].mean(dim="year").values.astype(np.float32)
        ace2_mean = ds["ace2_freq"].mean(dim="year").values.astype(np.float32)
        yr0, yr1 = int(ds["year"].values[0]), int(ds["year"].values[-1])

    vmax = float(np.nanpercentile(np.concatenate([era5_mean.ravel(), ace2_mean.ravel()]), 99))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    _panel(axes[0], era5_mean, lat, lon, f"ERA5  |  JJA {yr0}-{yr1}", vmax)
    mesh = _panel(axes[1], ace2_mean, lat, lon, f"ACE2  |  JJA {yr0}-{yr1}", vmax)
    fig.colorbar(mesh, ax=axes, shrink=0.7, orientation="vertical",
                label="Mean JJA HHE frequency (fraction of days, ±7d LOO 90th-pct)")
    fig.suptitle("Climatological JJA heat-extreme frequency  |  no clustering, no skill metric",
                fontsize=12, y=1.0)

    out_path = OUT_DIR / "jja_hhe_freq_era5_vs_ace2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
