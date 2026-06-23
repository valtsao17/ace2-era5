#!/usr/bin/env python3
"""SST ↔ HHE-frequency relationship, indexed over the FIGURE-2 HHE box.

Remake of the SST–HHE teleconnection figure, but the HHE index region is no
longer the hardcoded SEUS box — it's the *same* highest-HHE-concentration box
drawn on Figure 2 (extreme_freq_boxed_panels.find_box on the ERA5 HHE
climatology, 8°×12°, lands on the Gulf-of-California / NW-Mexico hotspot).

For each year we take the cos-lat-mean true-HHE frequency over that box (an
index), detrend it, and correlate (Pearson, detrended) the contemporaneous JJA
SST field against it at every ocean cell. SST = the prescribed ERA5 forcing
surface_temperature (the SAME field ACE2 is forced with), so the two panels
isolate the HHE *response*:

    left  : SST × ERA5 HHE index   (observed teleconnection)
    right : SST × ACE2 HHE index   (does ACE2's HHE respond to SST the same way?)

Stippling = p < 0.05. A green box marks the region of highest SST correlation in
each panel.

Outputs → outputs/lag_may/sst_teleconnection_hhe_lagged/figures/
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
    bbox_mean, detrend_along_year, pearson_corr_map, _render_corr_ax,
)
from sst_teleconnection_hhe_lagged import build_lagged_sst, HHE_DIR, OUT_DIR, FIG_DIR
from extreme_freq_boxed_panels import find_box as hhe_find_box
from cluster_skill_analysis_sliding7d import CONUS_LAT_SLICE, CONUS_LON_SLICE

# highest-correlation predictor box (search domain = the rendered SST domain)
CORR_BOX_LAT, CORR_BOX_LON = 10.0, 15.0
SEARCH = dict(lat_s=-25.0, lat_n=70.0, lon_w=190.0, lon_e=320.0)   # = -170..-40°W
MIN_SIG = 10                                                       # min p<0.05 cells


def find_corr_box(corr, pval, lat, lon):
    """Slide a fixed CORR_BOX over the SST domain and place it on the strongest
    POSITIVE, significant (p<0.05) correlation — maximize the mean r over the
    box's positive-significant ocean cells (magnitude, not dot count). Land/ice
    is NaN so it's never counted; the box stays over ocean.
    Returns (bbox, mean_r, n_sig)."""
    w = SEARCH
    cand_lat = lat[(lat >= w["lat_s"]) & (lat <= w["lat_n"])]
    cand_lon = lon[(lon >= w["lon_w"]) & (lon <= w["lon_e"])]
    best = None
    for la0 in cand_lat:
        la1 = la0 + CORR_BOX_LAT
        if la1 > w["lat_n"]:
            continue
        sla = (lat >= la0) & (lat <= la1)
        for lo0 in cand_lon:
            lo1 = lo0 + CORR_BOX_LON
            if lo1 > w["lon_e"]:
                continue
            slo = (lon >= lo0) & (lon <= lo1)
            sub_c = corr[np.ix_(sla, slo)]
            sub_p = pval[np.ix_(sla, slo)]
            # keep the box (mostly) over ocean — skip if >40% of cells are land/ice
            if np.isfinite(sub_c).mean() < 0.6:
                continue
            sig = np.isfinite(sub_c) & np.isfinite(sub_p) & (sub_p < 0.05) & (sub_c > 0)
            nsig = int(sig.sum())
            if nsig < MIN_SIG:
                continue
            mean_r = float(np.nanmean(sub_c[sig]))
            # maximize correlation magnitude over positive significant cells
            if best is None or mean_r > best[0]:
                best = (mean_r, nsig, (float(la0), float(la1), float(lo0), float(lo1)))
    if best is None:
        return None, None, 0
    return best[2], best[0], best[1]


def _draw_box(ax, box, label):
    la0, la1, lo0, lo1 = box
    x0 = lo0 - 360.0 if lo0 > 180 else lo0
    x1 = lo1 - 360.0 if lo1 > 180 else lo1
    ax.add_patch(plt.Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="#00b050", lw=2.4, zorder=7))
    ax.text(x0, la1 + 1.0, label, color="#00b050", fontsize=8, weight="bold", zorder=8)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values; lon = d["lon"].values
        era5 = d["era5_hi_freq"].values.astype(np.float32)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2 = d["ace2_hi_freq"].values.astype(np.float32)
    era5_clim = np.nanmean(era5, axis=0)

    # --- Figure-2 HHE box: highest ERA5 HHE concentration over CONUS (8x12) ---
    e_conus = era5_clim[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    latc = lat[CONUS_LAT_SLICE]; lonc = lon[CONUS_LON_SLICE]
    hhe_box, _ = hhe_find_box(e_conus, latc, lonc)            # (la0,la1,lo0,lo1)
    la0, la1, lo0, lo1 = hhe_box
    print(f"Figure-2 HHE index box: lat {la0:.1f}-{la1:.1f}N  "
          f"lon {360-lo1:.0f}-{360-lo0:.0f}°W", flush=True)

    era5_idx = bbox_mean(era5, lat, lon, hhe_box)
    ace2_idx = bbox_mean(ace2, lat, lon, hhe_box)
    print(f"  HHE index clim:  ERA5={np.nanmean(era5_idx):.4f}  ACE2={np.nanmean(ace2_idx):.4f}",
          flush=True)

    # --- contemporaneous JJA SST (lag0) from the ERA5 forcing (cached) ---
    sst, slat, slon = build_lagged_sst()
    sst_dt = detrend_along_year(sst["lag0_JJA"])

    print("Correlating SST × ERA5 HHE index ...", flush=True)
    corr_e, pval_e = pearson_corr_map(sst_dt, era5_idx)
    print("Correlating SST × ACE2 HHE index ...", flush=True)
    corr_a, pval_a = pearson_corr_map(sst_dt, ace2_idx)

    box_e, se, ne = find_corr_box(corr_e, pval_e, slat, slon)
    box_a, sa, na = find_corr_box(corr_a, pval_a, slat, slon)
    print(f"  strongest +corr box ERA5: lon {360-box_e[3]:.0f}-{360-box_e[2]:.0f}°W "
          f"lat {box_e[0]:.0f}-{box_e[1]:.0f}N  n_sig={ne}  mean r={se:.2f}", flush=True)
    print(f"  strongest +corr box ACE2: lon {360-box_a[3]:.0f}-{360-box_a[2]:.0f}°W "
          f"lat {box_a[0]:.0f}-{box_a[1]:.0f}N  n_sig={na}  mean r={sa:.2f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
    mesh = _render_corr_ax(axes[0], corr_e, pval_e, slat, slon,
                           "HHE index: ERA5   (SST × ERA5 HHE freq)")
    _draw_box(axes[0], box_e, f"r={se:.2f}")
    mesh = _render_corr_ax(axes[1], corr_a, pval_a, slat, slon,
                           "HHE index: ACE2   (SST × ACE2 HHE freq)")
    _draw_box(axes[1], box_a, f"r={sa:.2f}")
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Pearson r  (detrended JJA SST × HHE-freq index)")
    fig.suptitle("JJA SST vs HHE frequency over the Gulf-of-California box  |  "
                 "stipple p<0.05, box = strongest positive correlation", fontsize=12)
    out = FIG_DIR / "sst_hhe_fig2box_corr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    xr.Dataset(
        {"corr_era5": (("lat", "lon"), corr_e), "pval_era5": (("lat", "lon"), pval_e),
         "corr_ace2": (("lat", "lon"), corr_a), "pval_ace2": (("lat", "lon"), pval_a)},
        coords={"lat": slat, "lon": slon},
        attrs={"hhe_index_box": f"lat {la0:.2f}-{la1:.2f}N lon {lo0:.2f}-{lo1:.2f}E (highest HHE concentration)",
               "sst": "ERA5 forcing surface_temperature, JJA mean (lag0), detrended"},
    ).to_netcdf(OUT_DIR / "sst_hhe_fig2box_corr.nc")
    (OUT_DIR / "sst_hhe_fig2box_summary.json").write_text(json.dumps({
        "hhe_index_box_latN": [round(la0, 2), round(la1, 2)],
        "hhe_index_box_lonW": [round(360 - lo1, 1), round(360 - lo0, 1)],
        "era5_idx_clim": round(float(np.nanmean(era5_idx)), 5),
        "ace2_idx_clim": round(float(np.nanmean(ace2_idx)), 5),
        "highest_corr_box_era5": {"mean_r": round(se, 3),
                                  "lonW": [round(360 - box_e[3], 1), round(360 - box_e[2], 1)],
                                  "latN": [round(box_e[0], 1), round(box_e[1], 1)]},
        "highest_corr_box_ace2": {"mean_r": round(sa, 3),
                                  "lonW": [round(360 - box_a[3], 1), round(360 - box_a[2], 1)],
                                  "latN": [round(box_a[0], 1), round(box_a[1], 1)]},
    }, indent=2))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
