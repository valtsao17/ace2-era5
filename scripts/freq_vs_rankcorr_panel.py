#!/usr/bin/env python3
"""Contextual 3-panel CONUS comparison:

    [ mean ACE2 HHE frequency ]  [ Kendall tau ]  [ modified-Kendall z ]

Puts the extreme-frequency climatology next to the two rank-correlation skill
maps so spatial skill can be read against where HHE extremes actually occur.

Sources (all on the same 1x1 grid, CONUS box):
  mean ACE2 freq → seasonal_jja_sliding7d/jja_seasonal_freqs.nc  (ace2_freq mean)
  Kendall tau    → seasonal_jja_sliding7d/skill_jja_seasonal.nc  (kendall_tau)
  mod-Kendall z  → seasonal_jja_sliding7d_modkendall/skill_jja_seasonal.nc (kendall_tau holds z)

Output → outputs/lag_may/seasonal_jja_sliding7d_modkendall/freq_vs_rankcorr_conus.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)

SLIDING_DIR   = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
MODK_DIR      = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall"


def _panel(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1,
                   interpolation="nearest")
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)


def main():
    # mean ACE2 HHE seasonal frequency (CONUS)
    ds = xr.open_dataset(SLIDING_DIR / "jja_seasonal_freqs.nc")
    lat = ds["lat"].values[CONUS_LAT_SLICE]
    lon = ds["lon"].values[CONUS_LON_SLICE]
    ace2_freq = ds["ace2_freq"].values[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
    ds.close()
    freq_mean = np.nanmean(ace2_freq, axis=0)

    # Kendall tau (CONUS)
    ds = xr.open_dataset(SLIDING_DIR / "skill_jja_seasonal.nc")
    tau = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    ds.close()

    # modified-Kendall z (CONUS) — variable name is kendall_tau but holds z
    ds = xr.open_dataset(MODK_DIR / "skill_jja_seasonal.nc")
    zmap = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    zk = int(ds["kendall_tau"].attrs.get("truncation_k", 10))
    ds.close()

    # symmetric, data-driven limits for the diverging skill panels
    tlim = float(np.nanpercentile(np.abs(tau[np.isfinite(tau)]), 98))
    zlim = float(np.nanpercentile(np.abs(zmap[np.isfinite(zmap)]), 98))
    # frequency is near-uniform (percentile-threshold definition); stretch the
    # color range to p2–p98 so the small spatial structure is visible.
    ffin = freq_mean[np.isfinite(freq_mean)]
    fmin = float(np.nanpercentile(ffin, 2))
    fmax = float(np.nanpercentile(ffin, 98))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _panel(axes[0], freq_mean, lat, lon,
           "Mean ACE2 JJA HHE frequency (1980–2016)",
           "YlOrRd", fmin, fmax, "frequency (fraction of days)")
    _panel(axes[1], tau, lat, lon,
           "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -tlim, tlim, "τ")
    _panel(axes[2], zmap, lat, lon,
           f"Modified-Kendall z  (k={zk})",
           _TAU_CMAP, -zlim, zlim, "z")

    fig.suptitle("CONUS HHE frequency vs. rank-correlation skill", fontsize=13, y=1.00)
    fig.tight_layout()
    out = MODK_DIR / "freq_vs_rankcorr_conus.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"  freq max≈{fmax:.2f}  τ-lim≈{tlim:.2f}  z-lim≈{zlim:.2f}", flush=True)


if __name__ == "__main__":
    main()
