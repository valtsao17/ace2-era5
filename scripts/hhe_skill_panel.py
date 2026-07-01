#!/usr/bin/env python3
"""2x2 CONUS skill panel for the TRUE humid-heat extreme (HHE), the HHE twin of
skill_panel_combined.py:

    [ Kendall τ        ]   [ modified-Kendall z (k=10) ]
    [ Precision        ]   [ Recall                    ]

All four are on the absolute HHE definition (NOAA Heat Index >= 105 °F), 1980-2016,
seasonal-frequency / day-level over the WHOLE JJA season (not individual target
dates), NO leave-one-out, bias-corrected ACE2 (paper §8):

  Kendall τ      — interannual rank corr of the 37-yr HHE seasonal frequency
                   (fraction of JJA days HI>=105), ACE2 vs ERA5     [skill_hhe_seasonal.nc]
  mod-Kendall z  — Zheng & Lo top-k weighted variant, k=10, same 37-yr freq pairs
  precision/     — day-level: ERA5 day is HHE (HI>=105); ACE2 predicts HHE by
  recall           majority vote (>=13/25 members HI>=105 that day, bias-corrected);
                   pooled over years x JJA days per cell.

Fields masked to cells where ERA5 HHE climatology > ~1 day/season (else no signal).

Output -> outputs/lag_may/heat_index_era5/skill_panel_combined_hhe.png  (+ .nc)
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from seasonal_jja_skill import _SKILL_CMAP, cos_lat_mean, load_land_mask, domain_scores_label
from mod_kendall_metric import mk_z, DEFAULT_K, normalized_z_for_plot
from heat_index_era5 import heat_index, HI_THRESH

from copy import copy
# precision is undefined (NaN) where the model never predicts an HHE day
# (TP+FP=0); draw those cells neutral grey so they're distinct from a genuine 0
# (the plain _SKILL_CMAP maps both 0 and NaN to white).
_SKILL_CMAP_NA = copy(_SKILL_CMAP)
_SKILL_CMAP_NA.set_bad("#bdbdbd")
from hhe_ace2 import N_MEMBERS, YEARS

HHE_DIR   = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
CACHE_DIR = HHE_DIR / "ace2_daily_cache"
ERA5_MON  = HHE_DIR / "monthly_cache"
BIAS_NC   = HHE_DIR / "bias_fields.nc"
K_TRUNC   = 10
VOTE      = 0.5      # majority vote fraction of members
LA, LO    = CONUS_LAT_SLICE, CONUS_LON_SLICE


def mk_z_map_conus(pred, obs, k):
    """modified-Kendall z per CONUS cell from the 37-yr freq pairs."""
    _, nlat, nlon = pred.shape
    z = np.full((nlat, nlon), np.nan, dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            z[i, j] = mk_z(pred[:, i, j], obs[:, i, j], k)
    return z


def day_precision_recall():
    """CONUS day-level HHE precision/recall, ERA5 obs vs ACE2 majority vote."""
    with xr.open_dataset(BIAS_NC) as b:
        bt = b["bias_Tmax"].values[LA, LO]
        br = b["bias_RHmin"].values[LA, LO]
    nlat, nlon = bt.shape
    tp = np.zeros((nlat, nlon)); fp = np.zeros((nlat, nlon)); fn = np.zeros((nlat, nlon))

    for Y in YEARS:
        # ERA5 daily HHE indicator (JJA days), CONUS
        ef = sorted(glob.glob(str(ERA5_MON / f"hi_daily_y{Y:04d}_m*.nc")))
        if not ef:
            continue
        era = xr.concat([xr.open_dataset(f)["hi_F"] for f in ef], dim="time").sortby("time")
        era = era.assign_coords(time=era["time"].dt.floor("D"))
        era = era.isel(lat=LA, lon=LO)

        # ACE2 members: bias-corrected daily HHE indicator, CONUS
        mem = []
        for idx in range(N_MEMBERS):
            cp = CACHE_DIR / f"daily_y{Y}_mem{idx:02d}.nc"
            if not cp.exists():
                continue
            with xr.open_dataset(cp) as d:
                tmaxC = d["tmax_C"].values[:, LA, LO]
                rh    = d["rhmin_pct"].values[:, LA, LO]
                tvals = d["time"].values
            hi = heat_index((tmaxC - bt) * 9.0 / 5.0 + 32.0, np.clip(rh - br, 0.0, 100.0))
            da = xr.DataArray((hi >= HI_THRESH), dims=["time", "lat", "lon"],
                              coords={"time": tvals})
            mem.append(da.assign_coords(time=da["time"].dt.floor("D")))
        if not mem:
            continue
        ref = max(mem, key=lambda m: m.sizes["time"])         # common day axis
        stack = xr.concat([m.reindex(time=ref["time"], fill_value=False) for m in mem],
                          dim="member")
        pred = (stack.mean("member") >= VOTE).values          # (ndays, lat, lon)
        obs  = era.reindex(time=ref["time"], fill_value=False).values >= HI_THRESH

        tp += (pred & obs).sum(axis=0)
        fp += (pred & ~obs).sum(axis=0)
        fn += (~pred & obs).sum(axis=0)
        print(f"  {Y}: pooled days, TP sum so far={tp.sum():.0f}", flush=True)

    prec = np.where((tp + fp) > 0, tp / (tp + fp), np.nan).astype(np.float32)
    rec  = np.where((tp + fn) > 0, tp / (tp + fn), np.nan).astype(np.float32)
    return prec, rec


def _stipple(ax, sig, lat, lon):
    """Black dots at cell centres where `sig` is True (stipple = NOT significant)."""
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
    ax.set_facecolor("white")
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1, interpolation="nearest")
    _stipple(ax, sig, lat, lon)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)
    if mean_lbl is not None:
        ax.legend([Line2D([], [], linestyle="none")], [mean_lbl],
                  loc="lower left", fontsize=8, handlelength=0, handletextpad=0,
                  framealpha=1.0, borderpad=0.5).set_zorder(7)


def main():
    # τ + ERA5 HHE clim mask (global -> CONUS)
    with xr.open_dataset(HHE_DIR / "skill_hhe_seasonal.nc") as ds:
        lat = ds["lat"].values[LA]; lon = ds["lon"].values[LO]
        tau = ds["kendall_tau"].values[LA, LO]
        tau_p = ds["tau_p_value"].values[LA, LO]
        era5_clim = ds["era5_hhe_clim"].values[LA, LO]
    occurs = era5_clim > (1.0 / 92.0)

    # 37-yr HHE seasonal freq pairs (CONUS) for mod-Kendall z
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        era5 = d["era5_hi_freq"].values[:, LA, LO].astype(np.float32)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2 = d["ace2_hi_freq"].values[:, LA, LO].astype(np.float32)
    print(f"Computing CONUS modified-Kendall z (k={K_TRUNC}) ...", flush=True)
    zmap = mk_z_map_conus(ace2, era5, K_TRUNC)

    print("Computing CONUS day-level HHE precision/recall ...", flush=True)
    prec, rec = day_precision_recall()

    for arr in (tau, zmap, prec, rec):
        arr[~occurs] = np.nan

    # stipple = NOT significant (τ p>=0.05 / |z|<=1.96), only where defined
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))
    z_ns   = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)

    z_plot, z_scale = normalized_z_for_plot(zmap)
    land = load_land_mask(lat, lon)         # all / land / sea score breakdown
    tau_m = cos_lat_mean(tau, lat); z_m = cos_lat_mean(zmap, lat)
    zn_m = cos_lat_mean(z_plot, lat)
    prec_m = cos_lat_mean(prec, lat); rec_m = cos_lat_mean(rec, lat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    _panel(axes[0, 0], tau, lat, lon, "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -1.0, 1.0, "τ", domain_scores_label("mean τ", tau, lat, land),
           sig=tau_ns)
    _panel(axes[0, 1], z_plot, lat, lon, f"Normalized modified-Kendall z  (k={K_TRUNC})",
           _TAU_CMAP, -1.0, 1.0, "normalized z",
           domain_scores_label("mean norm z", z_plot, lat, land),
           sig=z_ns)
    _panel(axes[1, 0], prec, lat, lon, "Precision  (day-level HHE)",
           _SKILL_CMAP_NA, 0.0, 1.0, "precision", domain_scores_label("precision", prec, lat, land))
    _panel(axes[1, 1], rec, lat, lon, "Recall  (day-level HHE)",
           _SKILL_CMAP_NA, 0.0, 1.0, "recall", domain_scores_label("recall", rec, lat, land))
    # grey = undefined: precision where the model predicted no HHE days
    # (TP+FP=0); recall where ERA5 observed none (TP+FN=0). Annotate only when
    # such cells exist.
    _bbox = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="0.6")
    n_na_p = int(np.sum(~np.isfinite(prec)))
    if n_na_p:
        axes[1, 0].text(0.98, 0.03, f"grey = undefined (no predicted\nHHE days): {n_na_p} cells",
                        transform=axes[1, 0].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)
    n_na_r = int(np.sum(~np.isfinite(rec)))
    if n_na_r:
        axes[1, 1].text(0.98, 0.03, f"grey = undefined (no observed\nHHE days): {n_na_r} cells",
                        transform=axes[1, 1].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)
    fig.suptitle("CONUS humid-heat-extreme (HI≥105°F) skill — rank-correlation (top, stipple = "
                 "NOT significant: τ p≥0.05 / |z|≤1.96) vs day-level classification (bottom)  |  "
                 "JJA 1980–2016, seasonal, no-LOO", fontsize=12)
    out = HHE_DIR / "skill_panel_combined_hhe.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    xr.Dataset(
        {"kendall_tau": (("lat", "lon"), tau), "mod_kendall_z": (("lat", "lon"), zmap),
         "precision": (("lat", "lon"), prec), "recall": (("lat", "lon"), rec)},
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "CONUS HHE (HI>=105F) skill panel fields, seasonal/day-level, "
                            "no-LOO, bias-corrected ACE2", "truncation_k": K_TRUNC},
    ).to_netcdf(HHE_DIR / "skill_panel_combined_hhe.nc")
    print(f"wrote {out}", flush=True)
    print(f"  CONUS means: τ={tau_m:.3f}  z={z_m:.3f}  norm_z={zn_m:.3f}  "
          f"z-plot-scale={z_scale:.3f}  prec={prec_m:.3f}  rec={rec_m:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
