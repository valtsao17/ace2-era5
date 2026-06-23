#!/usr/bin/env python3
"""Seasonal-frequency forecast skill for the TRUE humid-heat extreme (HHE).

This is the deliverable that matches the standard sub-seasonal-to-seasonal
verification framing: for each of the 37 years we already have the JJA HHE
*frequency* (fraction of the 92 JJA days with NOAA Heat Index >= 105 F — an
ABSOLUTE threshold, so the day-count is physically meaningful and not
self-referential like the 90th-percentile temperature index). We score the
37 forecast-verification (ACE2, ERA5) frequency pairs per grid cell:

    Kendall tau   — interannual rank agreement
    Pearson r     — interannual linear agreement (magnitude-sensitive)

Target = "predict the number of HHE days in the upcoming summer", NOT whether a
particular day is extreme. τ/r are only defined where HHE actually occurs (many
cells have ~0 HHE days every year → no interannual variance → NaN), so the map
is populated over the HHE regions (US Gulf Coast, Sahel, Persian Gulf, Indo-
Gangetic plain, etc.).

Sources: outputs/lag_may/heat_index_era5/jja_hi_freq_{ace2,era5}.nc
Outputs → outputs/lag_may/heat_index_era5/
  skill_hhe_seasonal.nc
  hhe_seasonal_tau_global.png , hhe_seasonal_tau_conus.png
  hhe_seasonal_skill_panel.png   (global τ | CONUS τ | CONUS r)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from seasonal_jja_skill import kendall_tau_map, pearson_r_map, cos_lat_mean
from cluster_skill_analysis_sliding7d import (
    _single_map_figure, _plain_map_axes, _TAU_CMAP,
    CONUS_LAT_SLICE, CONUS_LON_SLICE,
)

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"


def conus_mean(field, lat):
    return cos_lat_mean(field[CONUS_LAT_SLICE, CONUS_LON_SLICE], lat[CONUS_LAT_SLICE])


def _panel(ax, field, lat, lon, title, vmin, vmax, cbar_label):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    ax.set_facecolor("#eaeaea")
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=_TAU_CMAP, zorder=1,
                   interpolation="nearest")
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)


def main():
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values
        lon = d["lon"].values
        era5 = d["era5_hi_freq"].values.astype(np.float32)   # (37, lat, lon)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2 = d["ace2_hi_freq"].values.astype(np.float32)
    print(f"HHE freq arrays: ERA5 {era5.shape}  ACE2 {ace2.shape}", flush=True)

    # only score cells where HHE actually occurs in ERA5 (else no signal to predict)
    era5_clim = np.nanmean(era5, axis=0)
    occurs = era5_clim > (1.0 / 92.0)   # at least ~1 HHE day per summer on average
    print(f"  cells with HHE (ERA5 clim > 1 day/season): {int(occurs.sum())}", flush=True)

    print("Computing Kendall τ across 37 years ...", flush=True)
    tau, tau_p = kendall_tau_map(ace2, era5)
    print("Computing Pearson r ...", flush=True)
    r, r_p = pearson_r_map(ace2, era5)

    for arr in (tau, tau_p, r, r_p):
        arr[~occurs] = np.nan

    tau_glob = cos_lat_mean(tau, lat)
    tau_conus = conus_mean(tau, lat)
    r_glob = cos_lat_mean(r, lat)
    r_conus = conus_mean(r, lat)
    # skill over HHE regions only (where it's defined)
    w = np.cos(np.deg2rad(lat))[:, None]
    tau_hhe = float(np.nansum(tau * w) / np.nansum(np.where(np.isfinite(tau), w, 0.0)))
    print(f"  cos-lat mean τ: global={tau_glob:.3f}  CONUS={tau_conus:.3f}  "
          f"HHE-cells={tau_hhe:.3f}", flush=True)
    print(f"  cos-lat mean r: global={r_glob:.3f}  CONUS={r_conus:.3f}", flush=True)

    xr.Dataset(
        {"kendall_tau": (("lat", "lon"), tau), "tau_p_value": (("lat", "lon"), tau_p),
         "pearson_r": (("lat", "lon"), r), "r_p_value": (("lat", "lon"), r_p),
         "era5_hhe_clim": (("lat", "lon"), era5_clim)},
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "Seasonal-frequency forecast skill, true HHE (HI>=105F), "
                            "ACE2 vs ERA5, 37 JJA year-pairs 1980-2016",
               "threshold": "absolute NOAA Heat Index >= 105 F"},
    ).to_netcdf(HHE_DIR / "skill_hhe_seasonal.nc")
    print(f"wrote {HHE_DIR / 'skill_hhe_seasonal.nc'}", flush=True)

    # individual global + CONUS τ maps (reuse the equal-aspect renderer)
    _single_map_figure(tau, lat, lon,
                       "Seasonal HHE-frequency skill — Kendall τ  (ACE2 vs ERA5, HI≥105°F)",
                       -0.6, 0.6, _TAU_CMAP, "τ",
                       HHE_DIR / "hhe_seasonal_tau_global.png")
    tau_c = tau[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    latc, lonc = lat[CONUS_LAT_SLICE], lon[CONUS_LON_SLICE]
    _single_map_figure(tau_c, latc, lonc,
                       "CONUS seasonal HHE-frequency skill — Kendall τ  (HI≥105°F)",
                       -0.6, 0.6, _TAU_CMAP, "τ",
                       HHE_DIR / "hhe_seasonal_tau_conus.png")

    # combined panel: global τ | CONUS τ | CONUS r
    r_c = r[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2))
    _panel(axes[0], tau, lat, lon, f"Global τ  (HHE-cells mean={tau_hhe:.2f})", -0.6, 0.6, "τ")
    _panel(axes[1], tau_c, latc, lonc, f"CONUS τ  (mean={tau_conus:.2f})", -0.6, 0.6, "τ")
    _panel(axes[2], r_c, latc, lonc, f"CONUS Pearson r  (mean={r_conus:.2f})", -0.8, 0.8, "r")
    fig.suptitle("True HHE (HI≥105°F) seasonal-frequency forecast skill — ACE2 vs ERA5, 37 JJA year-pairs",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HHE_DIR / "hhe_seasonal_skill_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {HHE_DIR / 'hhe_seasonal_skill_panel.png'}", flush=True)

    (HHE_DIR / "hhe_seasonal_skill.json").write_text(json.dumps({
        "definition": "Kendall τ / Pearson r on 37-yr ACE2 vs ERA5 JJA HHE seasonal "
                      "frequency, absolute threshold HI>=105F",
        "n_hhe_cells": int(occurs.sum()),
        "tau_global": round(float(tau_glob), 4), "tau_conus": round(float(tau_conus), 4),
        "tau_hhe_cells": round(float(tau_hhe), 4),
        "r_global": round(float(r_glob), 4), "r_conus": round(float(r_conus), 4),
    }, indent=2))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
