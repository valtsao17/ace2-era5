#!/usr/bin/env python3
"""2x2 CONUS precipitation skill panel — the precip twin of hhe_skill_panel.py /
skill_panel_combined.py:

    [ Kendall tau   (stipple = NOT significant) ]   [ modified-Kendall z (k=10) ]
    [ Precision     (day-level heavy precip)    ]   [ Recall                     ]

Heavy-precip extreme (parallels the RAW 90th-pct temperature extreme): a day is
"heavy" if its daily precip exceeds that cell's JJA-window 90th-pct threshold
(each dataset its own threshold, no-LOO). 2002-2016 (15 yrs), JJA, 25 members.

  Kendall tau    — interannual rank corr of the 15-yr heavy-precip seasonal
                   frequency (fraction of JJA days > p90), ACE2 ens-mean vs ERA5.
  mod-Kendall z  — Zheng & Lo top-k weighted variant, k=10, same 15 year-pairs.
  precision/     — day-level: ERA5 day is heavy; ACE2 predicts heavy by majority
  recall           vote (>=13/25 members heavy that day); pooled years x JJA days.

Sources (CONUS):
  ACE2 -> precip_jja/ace2_precip_cache/precip_y{Y}_mem{ii}.nc  (CONUS, mm/day)
  ERA5 -> precip_jja/era5_precip_cache/era5_precip_jja_{Y}.nc  (full grid -> CONUS)

Output -> outputs/lag_may/precip_jja/figures/precip_skill_panel.png  (+ .nc)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from seasonal_jja_skill import _SKILL_CMAP, cos_lat_mean, load_land_mask, domain_scores_label  # noqa: E402
from monthly_raw_freq_rankcorr import rankcorr_maps  # noqa: E402  (kendalltau + mk_z per cell)
from mod_kendall_metric import DEFAULT_K, normalized_z_for_plot  # noqa: E402

from copy import copy
# precision is undefined (NaN) where the model never predicts a heavy day
# (TP+FP=0); draw those cells neutral grey so they're distinct from a genuine 0
# (the plain _SKILL_CMAP maps both 0 and NaN to white).
_SKILL_CMAP_NA = copy(_SKILL_CMAP)
_SKILL_CMAP_NA.set_bad("#bdbdbd")

PJJA_DIR  = PROJECT_ROOT / "outputs/lag_may/precip_jja"
ACE2_DIR  = PJJA_DIR / "ace2_precip_cache"
ERA5_DIR  = PJJA_DIR / "era5_precip_cache"
FIG_DIR   = PJJA_DIR / "figures"
LA, LO    = CONUS_LAT_SLICE, CONUS_LON_SLICE
HOT_PCT   = 90.0
VOTE      = 0.5
N_MEMBERS = 25


def load_ace2(years):
    """(nyr, nmem, 92, nlat, nlon) CONUS daily precip mm/day."""
    out = None
    lat = lon = None
    for yi, Y in enumerate(years):
        for ii in range(N_MEMBERS):
            p = ACE2_DIR / f"precip_y{Y}_mem{ii:02d}.nc"
            if not p.exists():
                continue
            with xr.open_dataset(p) as d:
                a = d["precip_mmday"].values.astype(np.float32)
                if lat is None:
                    lat = d["lat"].values; lon = d["lon"].values
            if out is None:
                out = np.full((len(years), N_MEMBERS, a.shape[0], a.shape[1], a.shape[2]),
                              np.nan, np.float32)
            n = min(a.shape[0], out.shape[2])
            out[yi, ii, :n] = a[:n]
    return out, lat, lon


def load_era5(years):
    """(nyr, 92, nlat, nlon) CONUS daily precip mm/day."""
    out = None
    for yi, Y in enumerate(years):
        p = ERA5_DIR / f"era5_precip_jja_{Y}.nc"
        with xr.open_dataset(p) as d:
            a = d["precip_mmday"].values[:, LA, LO].astype(np.float32)
        if out is None:
            out = np.full((len(years), a.shape[0], a.shape[1], a.shape[2]), np.nan, np.float32)
        n = min(a.shape[0], out.shape[1])
        out[yi, :n] = a[:n]
    return out


def p90_thresh(data, sample_axes):
    grid = data.shape[-2:]
    moved = np.moveaxis(data, sample_axes, range(len(sample_axes))).reshape(-1, *grid)
    thr = np.empty(grid, np.float32)
    for i in range(grid[0]):
        thr[i] = np.nanpercentile(moved[:, i, :], HOT_PCT, axis=0)
    return thr


def seasonal_freq(data, thr, sample_axes):
    """Fraction of days > thr, per year. data (nyr, [nmem,] 92, lat, lon)."""
    ge = data > thr[None, ...]
    valid = np.isfinite(data)
    # average over member+day axes (everything except year + grid)
    ax = tuple(a for a in range(1, data.ndim - 2))
    num = (ge & valid).sum(axis=ax)
    den = np.maximum(valid.sum(axis=ax), 1)
    return (num / den).astype(np.float32)


def precision_recall(era5, ace2, thr_e, thr_a):
    nyr = era5.shape[0]
    nlat, nlon = era5.shape[-2:]
    tp = np.zeros((nlat, nlon)); fp = np.zeros((nlat, nlon)); fn = np.zeros((nlat, nlon))
    for y in range(nyr):
        obs = era5[y] > thr_e[None]                         # (92, lat, lon)
        memhit = ace2[y] > thr_a[None, None]                # (mem, 92, lat, lon)
        pred = (np.nanmean(memhit, axis=0) >= VOTE)         # (92, lat, lon)
        ovalid = np.isfinite(era5[y])
        obs = obs & ovalid
        tp += (pred & obs).sum(0)
        fp += (pred & ~obs & ovalid).sum(0)
        fn += (~pred & obs).sum(0)
    prec = np.where((tp + fp) > 0, tp / (tp + fp), np.nan).astype(np.float32)
    rec  = np.where((tp + fn) > 0, tp / (tp + fn), np.nan).astype(np.float32)
    return prec, rec


def _stipple(ax, sig, lat, lon):
    if sig is None or not np.any(sig):
        return
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.8, c="k", alpha=0.55, linewidths=0, zorder=6)


def _panel(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label, mean_lbl=None, sig=None):
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
        ax.legend([Line2D([], [], linestyle="none")], [mean_lbl], loc="lower left",
                  fontsize=8, handlelength=0, handletextpad=0, framealpha=1.0,
                  borderpad=0.5).set_zorder(7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2002-2016")
    args = ap.parse_args()
    a, b = args.years.split("-"); years = list(range(int(a), int(b) + 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("loading ACE2 precip cache ...", flush=True)
    ace2, lat, lon = load_ace2(years)
    print("loading ERA5 precip cache ...", flush=True)
    era5 = load_era5(years)
    assert ace2.shape[-2:] == era5.shape[-2:], "grid mismatch ACE2 vs ERA5"
    nday = min(ace2.shape[2], era5.shape[1])
    ace2 = ace2[:, :, :nday]; era5 = era5[:, :nday]
    print(f"  ACE2 {ace2.shape}  ERA5 {era5.shape}", flush=True)

    print("thresholds (JJA p90, no-LOO) ...", flush=True)
    thr_e = p90_thresh(era5, (0, 1))
    thr_a = p90_thresh(ace2, (0, 1, 2))

    fe = seasonal_freq(era5, thr_e, (0, 1))            # (nyr, lat, lon)
    fa = seasonal_freq(ace2, thr_a, (0, 1, 2))
    print("Kendall tau / mod-Kendall z per cell ...", flush=True)
    tau, tau_p, zmap = rankcorr_maps(fe, fa, DEFAULT_K)   # (obs=ERA5, pred=ACE2)

    print("precision / recall (day-level, majority vote) ...", flush=True)
    prec, rec = precision_recall(era5, ace2, thr_e, thr_a)

    # mask degenerate cells (no precip variability -> threshold ~0 everywhere)
    occurs = thr_e > 0.1
    for arr in (tau, tau_p, zmap, prec, rec):
        arr[~occurs] = np.nan

    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))   # stipple = NOT sig
    z_ns   = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)
    z_plot, z_scale = normalized_z_for_plot(zmap)
    land = load_land_mask(lat, lon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    _panel(axes[0, 0], tau, lat, lon, "Kendall τ  (ACE2 vs ERA5)", _TAU_CMAP, -1.0, 1.0, "τ",
           domain_scores_label("mean τ", tau, lat, land), sig=tau_ns)
    _panel(axes[0, 1], z_plot, lat, lon, f"Normalized modified-Kendall z  (k={DEFAULT_K})", _TAU_CMAP,
           -1.0, 1.0, "normalized z", domain_scores_label("mean norm z", z_plot, lat, land),
           sig=z_ns)
    _panel(axes[1, 0], prec, lat, lon, "Precision  (day-level heavy precip)", _SKILL_CMAP_NA,
           0.0, 1.0, "precision", domain_scores_label("precision", prec, lat, land))
    _panel(axes[1, 1], rec, lat, lon, "Recall  (day-level heavy precip)", _SKILL_CMAP_NA,
           0.0, 1.0, "recall", domain_scores_label("recall", rec, lat, land))
    # grey = undefined: precision where the model predicted no heavy-precip days
    # (TP+FP=0); recall where ERA5 observed none (TP+FN=0). Annotate only when
    # such cells exist.
    _bbox = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="0.6")
    n_na_p = int(np.sum(~np.isfinite(prec)))
    if n_na_p:
        axes[1, 0].text(0.98, 0.03, f"grey = undefined (no predicted\nheavy days): {n_na_p} cells",
                        transform=axes[1, 0].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)
    n_na_r = int(np.sum(~np.isfinite(rec)))
    if n_na_r:
        axes[1, 1].text(0.98, 0.03, f"grey = undefined (no observed\nheavy days): {n_na_r} cells",
                        transform=axes[1, 1].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)
    fig.suptitle("CONUS heavy-precip (daily > JJA p90) skill — rank-correlation (top, stipple = "
                 "NOT significant: τ p≥0.05 / |z|≤1.96) vs day-level classification (bottom)", fontsize=11)
    out = FIG_DIR / "precip_skill_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    xr.Dataset(
        {"kendall_tau": (("lat", "lon"), tau), "tau_p_value": (("lat", "lon"), tau_p),
         "mod_kendall_z": (("lat", "lon"), zmap), "precision": (("lat", "lon"), prec),
         "recall": (("lat", "lon"), rec)},
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "CONUS heavy-precip (>JJA p90) skill, seasonal/day-level, no-LOO",
               "years": f"{years[0]}-{years[-1]}", "truncation_k": DEFAULT_K},
    ).to_netcdf(FIG_DIR / "precip_skill_panel.nc")
    print(f"wrote {out}", flush=True)
    print(f"  CONUS means: τ={cos_lat_mean(tau, lat):.3f}  z={cos_lat_mean(zmap, lat):.3f}  "
          f"norm_z={cos_lat_mean(z_plot, lat):.3f}  z-plot-scale={z_scale:.3f}  "
          f"prec={cos_lat_mean(prec, lat):.3f}  rec={cos_lat_mean(rec, lat):.3f}", flush=True)
    print(f"  τ significant cells: {int((np.isfinite(tau_p)&(tau_p<0.05)).sum())} / "
          f"{int(np.isfinite(tau).sum())}", flush=True)


if __name__ == "__main__":
    main()
