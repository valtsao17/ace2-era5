#!/usr/bin/env python3
"""Paper-orientation regression pattern: gridded HHE frequency regressed onto a
standardized box-averaged SST index (%/sigma), ERA5 vs ACE2.

This reproduces the Jia et al. regression figure ("regression coefficients of
the frequency of HHE with the standardized box-averaged JJA mean SSTs ...") but
swaps the paper's TNA index for OUR strongest-correlation SST box -- the
E-Pacific-off-SW-Mexico box (16.5-26.5N, 100.5-115.5W) found in
sst_hhe_fig2box_corr.py as the strongest SST link to the Gulf-of-California HHE
box.  The SAME ERA5-forcing SST index is used for both panels (as the paper uses
one TNA index for both ERA5 and SPEAR), so the panels are directly comparable.

Direction (matches the paper, opposite of sst_hhe_fig2box_corr/_regression):
    predictor (scalar) : standardized box-mean detrended JJA SST index
    response  (field)  : gridded JJA HHE frequency over land (CONUS)
    -> slope at each LAND cell = Delta HHE freq (% of JJA days) per 1 sigma SST.
       Map + significance stipple are over LAND.

Outputs -> outputs/lag_may/sst_teleconnection_hhe_lagged/figures/
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

from sst_teleconnection_jja_sliding7d import (
    bbox_mean, detrend_1d, detrend_along_year, regression_map,
    _render_regression_ax, CONUS_EXTENT,
)
from sst_teleconnection_hhe_lagged import build_lagged_sst, HHE_DIR, OUT_DIR, FIG_DIR

# strongest-correlation ERA5 SST box (from sst_hhe_fig2box_summary.json)
SUMMARY = OUT_DIR / "sst_hhe_fig2box_summary.json"


def sst_index_box():
    """ERA5 strongest-corr box as a (la0, la1, lo0, lo1) bbox in 0-360 lon."""
    j = json.loads(SUMMARY.read_text())["highest_corr_box_era5"]
    la0, la1 = j["latN"]
    loW0, loW1 = j["lonW"]            # degrees west
    lo0, lo1 = 360.0 - loW1, 360.0 - loW0
    return (float(la0), float(la1), float(lo0), float(lo1)), j


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    box, jinfo = sst_index_box()
    la0, la1, lo0, lo1 = box
    print(f"SST index box (strongest ERA5 corr): lat {la0:.1f}-{la1:.1f}N  "
          f"lon {360-lo1:.0f}-{360-lo0:.0f}°W  (mean r was {jinfo['mean_r']})", flush=True)

    # --- standardized box-mean JJA SST index (detrended, unit variance) ---
    sst, slat, slon = build_lagged_sst()
    sst_idx = bbox_mean(np.asarray(sst["lag0_JJA"], np.float32), slat, slon, box)
    sst_idx_dt = detrend_1d(sst_idx)
    sd = float(np.nanstd(sst_idx_dt, ddof=1))
    sst_idx_std = sst_idx_dt / sd                         # ~ unit variance
    print(f"  SST index: mean={np.nanmean(sst_idx):.3f} K  detrended std={sd:.3f} K "
          f"-> standardized predictor", flush=True)

    # --- gridded true-HHE frequency fields, in % of JJA days, detrended ---
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values; lon = d["lon"].values
        era5 = d["era5_hi_freq"].values.astype(np.float32) * 100.0
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2 = d["ace2_hi_freq"].values.astype(np.float32) * 100.0
    era5_dt = detrend_along_year(era5)
    ace2_dt = detrend_along_year(ace2)

    print("Regressing gridded ERA5 HHE freq on standardized SST index ...", flush=True)
    slope_e, pval_e = regression_map(era5_dt, sst_idx_std)
    print("Regressing gridded ACE2 HHE freq on standardized SST index ...", flush=True)
    slope_a, pval_a = regression_map(ace2_dt, sst_idx_std)

    # --- render over CONUS (land), shared symmetric %/sigma scale ---
    lon_min, lon_max, lat_min, lat_max = CONUS_EXTENT
    lon_180 = np.where(lon > 180.0, lon - 360.0, lon)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_180 >= lon_min) & (lon_180 <= lon_max)
    lat_sub, lon_sub = lat[lat_sel], lon_180[lon_sel]

    def _sub(a):
        return a[lat_sel, :][:, lon_sel]
    s_e, p_e = _sub(slope_e), _sub(pval_e)
    s_a, p_a = _sub(slope_a), _sub(pval_a)

    finite = np.concatenate([s_e[np.isfinite(s_e)], s_a[np.isfinite(s_a)]])
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-6)

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.035],
        left=0.055, right=0.94, bottom=0.12, top=0.84, wspace=0.08,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    _render_regression_ax(axes[0], s_e, p_e, lon_sub, lat_sub, vmax,
                          "a  ERA5 reanalysis")
    mesh = _render_regression_ax(axes[1], s_a, p_a, lon_sub, lat_sub, vmax,
                                 "b  ACE2 hindcast")
    fig.colorbar(mesh, cax=cax, orientation="vertical",
                 label="Regression coeff.  (Δ HHE freq %  per 1σ of box SST)")
    fig.suptitle("HHE frequency regressed onto the standardized E-Pacific (16–26 N, "
                 "100–116 °W) JJA SST index  |  stipple p<0.05", fontsize=12, y=0.91)
    out = FIG_DIR / "sst_hhe_strongestbox_regression.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    xr.Dataset(
        {"slope_era5": (("lat", "lon"), slope_e), "pval_era5": (("lat", "lon"), pval_e),
         "slope_ace2": (("lat", "lon"), slope_a), "pval_ace2": (("lat", "lon"), pval_a)},
        coords={"lat": lat, "lon": lon},
        attrs={"sst_index_box": f"lat {la0:.1f}-{la1:.1f}N lon {lo0:.1f}-{lo1:.1f}E "
                                "(strongest ERA5 SST corr box)",
               "units": "percent of JJA days per 1 std-dev of detrended box-mean JJA SST",
               "direction": "gridded HHE freq (response) on scalar SST index (predictor)"},
    ).to_netcdf(OUT_DIR / "sst_hhe_strongestbox_regression.nc")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
