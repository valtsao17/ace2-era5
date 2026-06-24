#!/usr/bin/env python3
"""Single combined CONUS panel of all four day/season HHE skill diagnostics:

    [ Kendall τ        ]   [ modified-Kendall z ]
    [ Precision        ]   [ Recall                  ]

Puts the two rank-correlation skill metrics (signed, diverging blue-white-red)
above the two day-level classification metrics (0–1, sequential) so the whole
skill story reads from one figure. All four fields already exist on disk on the
same 1° grid; this script only assembles + renders them (CONUS box).

Sources (CONUS slice):
  Kendall τ      → seasonal_jja_sliding7d/skill_jja_seasonal.nc          (kendall_tau)
  mod-Kendall z  → seasonal_jja_sliding7d_modkendall/skill_jja_seasonal.nc (kendall_tau holds z)
  precision      → seasonal_jja_sliding7d/precision_recall_jja_seasonal.nc (precision)
  recall         → seasonal_jja_sliding7d/precision_recall_jja_seasonal.nc (recall)

Output → outputs/lag_may/seasonal_jja_sliding7d_modkendall/skill_panel_combined.png
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
from seasonal_jja_skill import _SKILL_CMAP, cos_lat_mean

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
MODK_DIR    = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall"


def _stipple(ax, sig, lat, lon):
    """Black dots at cell centres where `sig` is True (p<0.05 / |z|>1.96)."""
    if sig is None or not np.any(sig):
        return
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.8, c="k", alpha=0.55,
               linewidths=0, zorder=6)


def _panel(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label, mean_lbl=None,
           sig=None):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1,
                   interpolation="nearest")
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
    # Kendall τ (+ p-value for stippling)
    with xr.open_dataset(SLIDING_DIR / "skill_jja_seasonal.nc") as ds:
        lat = ds["lat"].values[CONUS_LAT_SLICE]
        lon = ds["lon"].values[CONUS_LON_SLICE]
        tau = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        tau_p = ds["tau_p_value"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]

    # modified-Kendall z (variable name is kendall_tau but holds z)
    with xr.open_dataset(MODK_DIR / "skill_jja_seasonal.nc") as ds:
        zmap = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        zk = int(ds["kendall_tau"].attrs.get("truncation_k", 10))

    # precision / recall
    with xr.open_dataset(SLIDING_DIR / "precision_recall_jja_seasonal.nc") as ds:
        prec = ds["precision"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        rec  = ds["recall"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]

    # NON-significance masks for the rank-correlation panels (stipple = p>=0.05)
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))
    z_ns   = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)       # |z|<=1.96 -> p>=0.05

    # symmetric, data-driven limits for the diverging skill panels
    tlim = float(np.nanpercentile(np.abs(tau[np.isfinite(tau)]), 98))
    zlim = float(np.nanpercentile(np.abs(zmap[np.isfinite(zmap)]), 98))

    tau_m  = cos_lat_mean(tau, lat)
    z_m    = cos_lat_mean(zmap, lat)
    prec_m = cos_lat_mean(prec, lat)
    rec_m  = cos_lat_mean(rec, lat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    _panel(axes[0, 0], tau, lat, lon, "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -tlim, tlim, "τ", f"mean τ = {tau_m:.3f}", sig=tau_ns)
    _panel(axes[0, 1], zmap, lat, lon, f"Modified-Kendall z  (k={zk})",
           _TAU_CMAP, -zlim, zlim, "z", f"mean z = {z_m:.3f}", sig=z_ns)
    _panel(axes[1, 0], prec, lat, lon, "Precision  (day-level)",
           _SKILL_CMAP, 0.0, 1.0, "precision", f"mean = {prec_m:.3f}")
    _panel(axes[1, 1], rec, lat, lon, "Recall  (day-level)",
           _SKILL_CMAP, 0.0, 1.0, "recall", f"mean = {rec_m:.3f}")

    fig.suptitle("CONUS raw heat-extreme (90th-pct TMP2m) skill — rank-correlation (top, "
                 "stipple = NOT significant: τ p≥0.05 / |z|≤1.96) vs day-level classification "
                 "(bottom)  |  JJA 1980–2016, seasonal, no-LOO", fontsize=12)
    out = MODK_DIR / "skill_panel_combined_raw.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"  CONUS means: τ={tau_m:.3f}  z={z_m:.3f}  prec={prec_m:.3f}  rec={rec_m:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
