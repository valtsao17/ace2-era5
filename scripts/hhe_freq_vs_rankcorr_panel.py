#!/usr/bin/env python3
"""CONUS true-HHE 3-panel: seasonal frequency vs. rank-correlation skill.

    [ mean ERA5 HHE seasonal freq ]  [ Kendall τ ]  [ modified-Kendall z (k=10) ]

True humid-heat extreme (NOAA Heat Index >= 105 F), seasonal-frequency over the
whole JJA season, no-LOO, bias-corrected ACE2. Frequency backdrop is ERA5 (the
observed "where HHE occurs"), which is also where τ/z are defined. Parallels
freq_vs_rankcorr_panel.py but on the real HHE instead of the percentile extreme.

Sources (CONUS box):
  ERA5 HHE freq → heat_index_era5/jja_hi_freq_era5.nc  (era5_hi_freq, mean over yrs)
  Kendall τ     → heat_index_era5/skill_panel_combined_hhe.nc  (kendall_tau)
  mod-Kendall z → heat_index_era5/skill_panel_combined_hhe.nc  (mod_kendall_z, k=10)

Output → outputs/lag_may/heat_index_era5/hhe_freq_vs_rankcorr_conus.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from seasonal_jja_skill import cos_lat_mean

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
LA, LO  = CONUS_LAT_SLICE, CONUS_LON_SLICE
WRED    = LinearSegmentedColormap.from_list(
    "wred", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#a50f15"])


def _stipple(ax, sig, lat, lon):
    """Black dots at cell centres where `sig` is True (p<0.05 / |z|>1.96)."""
    if sig is None or not np.any(sig):
        return
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.8, c="k", alpha=0.55,
               linewidths=0, zorder=6)


def _panel(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label, sig=None,
           mean_lbl=None):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    ax.set_facecolor("white")
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1, interpolation="nearest")
    _stipple(ax, sig, lat, lon)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)
    if mean_lbl is not None:
        ax.text(0.015, 0.04, mean_lbl, transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left", zorder=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))


def main():
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as ds:
        lat = ds["lat"].values[LA]
        lon = ds["lon"].values[LO]
        era5_freq = np.nanmean(ds["era5_hi_freq"].values, axis=0)[LA, LO] * 100.0

    with xr.open_dataset(HHE_DIR / "skill_panel_combined_hhe.nc") as ds:
        tau = ds["kendall_tau"].values
        zmap = ds["mod_kendall_z"].values
        zk = int(ds.attrs.get("truncation_k", 10))

    # τ significance (p<0.05) from the seasonal HHE skill file, CONUS slice
    with xr.open_dataset(HHE_DIR / "skill_hhe_seasonal.nc") as ds:
        tau_p = ds["tau_p_value"].values[LA, LO]
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))   # stipple = NOT sig
    z_ns   = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)                  # |z|<=1.96 -> p>=0.05

    # show the frequency only where HHE skill is defined (parallels τ/z mask)
    era5_freq = np.where(np.isfinite(tau), era5_freq, np.nan)

    tlim = float(np.nanpercentile(np.abs(tau[np.isfinite(tau)]), 98))
    zlim = float(np.nanpercentile(np.abs(zmap[np.isfinite(zmap)]), 98))
    ffin = era5_freq[np.isfinite(era5_freq)]
    fmax = float(np.nanpercentile(ffin, 98)) if ffin.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    _panel(axes[0], era5_freq, lat, lon, "Mean ERA5 JJA HHE frequency (HI≥105°F)",
           WRED, 0.0, fmax, "% of JJA days")
    tau_m = cos_lat_mean(tau, lat)
    z_m   = cos_lat_mean(zmap, lat)
    _panel(axes[1], tau, lat, lon, "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -tlim, tlim, "τ", sig=tau_ns, mean_lbl=f"mean τ = {tau_m:.3f}")
    _panel(axes[2], zmap, lat, lon, f"Modified-Kendall z  (k={zk})",
           _TAU_CMAP, -zlim, zlim, "z", sig=z_ns, mean_lbl=f"mean z = {z_m:.3f}")
    fig.suptitle("CONUS true-HHE (HI≥105°F) seasonal frequency vs. rank-correlation skill  |  "
                 "37 JJA year-pairs, no-LOO  (stipple = NOT significant: τ p≥0.05 / |z|≤1.96)",
                 fontsize=13)
    out = HHE_DIR / "hhe_freq_vs_rankcorr_conus.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"  freq max≈{fmax:.2f}%  τ-lim≈{tlim:.2f}  z-lim≈{zlim:.2f}  "
          f"τ-nonsig cells={int(tau_ns.sum())}  z-nonsig cells={int(z_ns.sum())}", flush=True)


if __name__ == "__main__":
    main()
