#!/usr/bin/env python3
"""HHE PER-METRIC seasonal figures (one map per metric), the HHE counterpart of
monthly_raw_freq_rankcorr.py:

    Figure 1  frequency (ERA5 | ACE2)    Figure 2  Kendall τ    Figure 3  modified-Kendall z

True humid-heat extreme (NOAA Heat Index ≥ 105°F), whole-JJA seasonal frequency,
no-LOO, bias-corrected ACE2, on the new 92-day data. One figure per metric (not
split by month — the HHE side stays seasonal). The τ and z maps carry the
all/land/sea score legend and stipple = NOT significant (τ p≥0.05 / |z|≤1.96).

Sources (CONUS):
  frequency → heat_index_era5/jja_hi_freq_{era5,ace2}.nc  (mean over 37 years)
  Kendall τ + mod-z → heat_index_era5/skill_panel_combined_hhe.nc
  τ p-value         → heat_index_era5/skill_hhe_seasonal.nc (tau_p_value)

Output → outputs/lag_may/heat_index_era5/
           hhe_frequency.png  hhe_kendalltau.png  hhe_modkendalltau.png
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

from cluster_skill_analysis_sliding7d import _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE
from seasonal_jja_skill import cos_lat_mean, load_land_mask, domain_scores_label
from monthly_raw_freq_rankcorr import _map, WRED   # shared single-map renderer
from mod_kendall_metric import normalized_z_for_plot

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
LA, LO  = CONUS_LAT_SLICE, CONUS_LON_SLICE


def main():
    with xr.open_dataset(HHE_DIR / "skill_panel_combined_hhe.nc") as ds:
        lat = ds["lat"].values; lon = ds["lon"].values
        tau = ds["kendall_tau"].values
        zmap = ds["mod_kendall_z"].values
        zk = int(ds.attrs.get("truncation_k", 10))
    with xr.open_dataset(HHE_DIR / "skill_hhe_seasonal.nc") as ds:
        tau_p = ds["tau_p_value"].values[LA, LO]
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        ec = np.nanmean(d["era5_hi_freq"].values, axis=0)[LA, LO] * 100.0
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ac = np.nanmean(d["ace2_hi_freq"].values, axis=0)[LA, LO] * 100.0

    land = load_land_mask(lat, lon)
    suff = "JJA 1980–2016, seasonal, no-LOO  (stipple = NOT significant)"

    # Figure 1 — frequency (ERA5 | ACE2)
    both = np.concatenate([ec[np.isfinite(ec)], ac[np.isfinite(ac)]])
    fmax = float(np.nanpercentile(both, 98)) if both.size else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    _map(axes[0], ec, lat, lon, "ERA5", WRED, 0.0, fmax, "% of JJA days")
    _map(axes[1], ac, lat, lon, "ACE2", WRED, 0.0, fmax, "% of JJA days")
    fig.suptitle("True HHE (HI ≥ 105°F) seasonal frequency", fontsize=13)
    fig.savefig(HHE_DIR / "hhe_frequency.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Figure 2 — Kendall τ
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    _map(ax, tau, lat, lon, "Kendall τ — HHE  (ACE2 vs ERA5)", _TAU_CMAP, -1.0, 1.0,
         "τ", sig=tau_ns, mean_lbl=domain_scores_label("mean τ", tau, lat, land))
    fig.suptitle(suff, fontsize=10)
    fig.savefig(HHE_DIR / "hhe_kendalltau.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Figure 3 — modified-Kendall z, normalized to [-1, 1] for plotting.
    z_plot, z_scale = normalized_z_for_plot(zmap)
    z_ns = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    _map(ax, z_plot, lat, lon, f"Normalized modified-Kendall z (k={zk}) — HHE", _TAU_CMAP,
         -1.0, 1.0, "normalized z", sig=z_ns,
         mean_lbl=domain_scores_label("mean norm z", z_plot, lat, land))
    fig.suptitle(suff, fontsize=10)
    fig.savefig(HHE_DIR / "hhe_modkendalltau.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote hhe_frequency/kendalltau/modkendalltau.png  "
          f"(τ mean={cos_lat_mean(tau, lat):.3f}  z mean={cos_lat_mean(zmap, lat):.3f}  "
          f"z-plot-scale={z_scale:.3f})", flush=True)


if __name__ == "__main__":
    main()
