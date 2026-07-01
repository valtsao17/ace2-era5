#!/usr/bin/env python3
"""Contextual 3-panel CONUS comparison for dry/raw heat extremes:

    [ mean ACE2 raw-extreme frequency ]  [ Kendall tau ]  [ modified-Kendall z ]

Puts the extreme-frequency climatology next to the two rank-correlation skill
maps so spatial skill can be read against where dry/raw heat extremes actually
occur.

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
from mod_kendall_metric import normalized_z_for_plot

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
    # Mean ACE2 dry/raw seasonal heat-extreme frequency (CONUS).
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

    # Kendall tau and normalized modified-Kendall z share the same [-1, 1]
    # plotting range for direct visual comparison.
    z_plot, z_scale = normalized_z_for_plot(zmap)
    # A seasonal top-decile frequency should be close to 0.10 by construction.
    # Use a fixed physical scale instead of p2-p98 stretching, otherwise tiny
    # sampling/missingness differences look like meaningful spatial structure.
    ffin = freq_mean[np.isfinite(freq_mean)]
    fmin = 0.0
    fmax = 0.10

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _panel(axes[0], freq_mean, lat, lon,
           "Mean ACE2 JJA raw heat-extreme frequency (1980-2016)",
           "YlOrRd", fmin, fmax, "frequency (fraction of finite JJA member-days)")
    _panel(axes[1], tau, lat, lon,
           "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -1.0, 1.0, "τ")
    _panel(axes[2], z_plot, lat, lon,
           f"Normalized modified-Kendall z  (k={zk})",
           _TAU_CMAP, -1.0, 1.0, "normalized z")

    fig.suptitle("CONUS raw TMP2m heat-extreme frequency vs. rank-correlation skill",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = MODK_DIR / "freq_vs_rankcorr_conus.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    raw_out = MODK_DIR / "freq_vs_rankcorr_conus_jja_raw.png"
    fig.savefig(raw_out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"wrote {raw_out}", flush=True)
    print(f"  freq mean≈{np.nanmean(ffin):.3f}  freq range≈{np.nanmin(ffin):.3f}-{np.nanmax(ffin):.3f}  "
          f"τ-lim=1.00  z-plot-scale≈{z_scale:.2f}", flush=True)


if __name__ == "__main__":
    main()
