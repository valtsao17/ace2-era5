#!/usr/bin/env python3
"""Compare true Humid Heat Extreme (HHE) frequency against the dry/temperature
extreme, for ERA5 and ACE2.

True HHE  = day with NOAA Heat Index >= 105 F (from daily Tmax + daily RHmin),
            per the reference paper (npj Clim Atmos Sci 2024, s41612-024-00723-0).
            Fields: outputs/lag_may/heat_index_era5/jja_hi_freq_{era5,ace2}.nc
Dry/temp  = the existing 90th-pct (+/-7d LOO) TMP2m extreme:
            outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc

Top row  : ERA5 HHE | ACE2 HHE | ACE2-ERA5 HHE difference   (shared HHE scale)
Bottom   : ERA5 dry | ACE2 dry | ACE2 HHE vs dry overlay context (own scale)

Outputs -> outputs/lag_may/heat_index_era5/hhe_vs_dry_compare.png  (+ .json stats)
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

from seasonal_jja_skill import cos_lat_mean, _roll_to_180, _draw_coast, _SKILL_CMAP  # noqa: E402
from cluster_skill_analysis_sliding7d import CONUS_LAT_SLICE, CONUS_LON_SLICE  # noqa: E402

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
DRY_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"


def conus_mean(field, lat, lon):
    sub = field[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    return cos_lat_mean(sub, lat[CONUS_LAT_SLICE])


def _panel(ax, field, lat, lon, title, vmin, vmax, cmap=_SKILL_CMAP, diff=False):
    field_r, lon_r = _roll_to_180(field, lon)
    LON2D, LAT2D = np.meshgrid(lon_r, lat)
    ax.set_facecolor("#d0e8f0")
    mesh = ax.pcolormesh(LON2D, LAT2D, field_r, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="nearest", zorder=1)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_aspect("equal")   # keep true 2:1 geographic proportions, no stretch
    _draw_coast(ax)
    g = cos_lat_mean(field, lat); c = conus_mean(field, lat, lon)
    ax.text(0.01, 0.03, f"global {g:.4f}\nCONUS {c:.4f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xticklabels(["180", "120W", "60W", "0", "60E", "120E", "180"], fontsize=7)
    ax.set_yticklabels(["90S", "60S", "30S", "0", "30N", "60N", "90N"], fontsize=7)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=10)
    return mesh


def main():
    # --- load true HHE ---
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values; lon = d["lon"].values
        era5_hhe = d["era5_hi_freq"].mean("year").values.astype(np.float32)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2_hhe = d["ace2_hi_freq"].mean("year").values.astype(np.float32)
    # --- load dry/temperature extreme ---
    with xr.open_dataset(DRY_NC) as d:
        era5_dry = d["era5_freq"].mean("year").values.astype(np.float32)
        ace2_dry = d["ace2_freq"].mean("year").values.astype(np.float32)

    vmax_hhe = max(float(np.nanpercentile(np.r_[era5_hhe.ravel(), ace2_hhe.ravel()], 99)), 1e-6)
    vmax_dry = max(float(np.nanpercentile(np.r_[era5_dry.ravel(), ace2_dry.ravel()], 99)), 1e-6)
    dmax = max(float(np.nanpercentile(np.abs(ace2_hhe - era5_hhe), 99)), 1e-6)

    fig, ax = plt.subplots(2, 3, figsize=(21, 7.2))
    m1 = _panel(ax[0, 0], era5_hhe, lat, lon, "ERA5  HHE (HI>=105F)", 0, vmax_hhe)
    _panel(ax[0, 1], ace2_hhe, lat, lon, "ACE2  HHE (HI>=105F)", 0, vmax_hhe)
    md = _panel(ax[0, 2], ace2_hhe - era5_hhe, lat, lon, "ACE2 - ERA5  HHE",
                -dmax, dmax, cmap="RdBu_r")
    m2 = _panel(ax[1, 0], era5_dry, lat, lon, "ERA5  dry extreme (TMP2m 90th pct)", 0, vmax_dry)
    _panel(ax[1, 1], ace2_dry, lat, lon, "ACE2  dry extreme (TMP2m 90th pct)", 0, vmax_dry)
    md2 = _panel(ax[1, 2], ace2_hhe - ace2_dry, lat, lon, "ACE2  HHE - dry",
                 -vmax_dry, vmax_dry, cmap="RdBu_r")

    fig.colorbar(m1, ax=ax[0, :2], shrink=0.8, label="HHE freq (frac of JJA days)")
    fig.colorbar(md, ax=ax[0, 2], shrink=0.8, label="diff")
    fig.colorbar(m2, ax=ax[1, :2], shrink=0.8, label="dry-extreme freq")
    fig.colorbar(md2, ax=ax[1, 2], shrink=0.8, label="diff")
    fig.suptitle("True humid-heat extreme vs dry/temperature extreme  |  JJA 1980-2016",
                 fontsize=13, y=0.99)

    out_png = HHE_DIR / "hhe_vs_dry_compare.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)

    stats = {
        "definition_hhe": "NOAA Heat Index >= 105 F from daily Tmax + daily RHmin",
        "era5_hhe_global": cos_lat_mean(era5_hhe, lat), "era5_hhe_conus": conus_mean(era5_hhe, lat, lon),
        "ace2_hhe_global": cos_lat_mean(ace2_hhe, lat), "ace2_hhe_conus": conus_mean(ace2_hhe, lat, lon),
        "era5_dry_global": cos_lat_mean(era5_dry, lat), "era5_dry_conus": conus_mean(era5_dry, lat, lon),
        "ace2_dry_global": cos_lat_mean(ace2_dry, lat), "ace2_dry_conus": conus_mean(ace2_dry, lat, lon),
    }
    stats = {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in stats.items()}
    (HHE_DIR / "hhe_vs_dry_compare.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
