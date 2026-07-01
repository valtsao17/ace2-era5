#!/usr/bin/env python3
"""Monthly RAW heat-extreme PER-METRIC figures (separate, one map per metric):

    Figure 1  frequency        Figure 2  Kendall τ        Figure 3  modified-Kendall z

For the RAW 90th-percentile heat extreme, each metric is rendered as its OWN
figure, separately for June, July and August (3 monthly maps per metric → 9 raw
figures). The HHE counterpart (one seasonal map per metric) is hhe_metric_maps.py.

Definition (mirrors monthly_raw_extreme_panels.py): each dataset's OWN full-JJA-
window 90th-pct Tmax threshold per cell (no-LOO); a day is "extreme" if its Tmax
exceeds that fixed threshold; per-year MONTHLY frequency is the fraction of that
month's days (ACE2: × 25 members) that are extreme. Kendall τ and top-k-weighted
modified-Kendall z score the 37 (ACE2, ERA5) monthly-frequency pairs per cell.
The τ and z maps carry the all/land/sea score legend and stipple = NOT significant
(τ p≥0.05 / |z|≤1.96).

Fast, cache-based (no 371 MB run re-reads), full 92-day JJA:
  ERA5 Tmax → postprocess_jja/era5_cache  (seasonal_jja_skill._load_all_era5_jja)
  ACE2 Tmax → heat_index_era5/ace2_daily_cache/daily_y{Y}_mem{ii}.nc (tmax_C, 92d)

Computed monthly fields are cached to <MODK_DIR>/monthly_raw_rc_fields.npz so
`--replot` re-renders instantly without the heavy reload.

Output → outputs/lag_may/seasonal_jja_sliding7d_modkendall/
           raw_frequency_{jun,jul,aug}.png
           raw_kendalltau_{jun,jul,aug}.png
           raw_modkendalltau_{jun,jul,aug}.png
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from extreme_freq_boxed_panels import find_box as find_hotspot_box
from seasonal_jja_skill import (
    YEARS, N_MEMBERS, HOT_PCT, JJA_SEQ, _load_all_era5_jja,
    cos_lat_mean, load_land_mask, domain_scores_label,
)
from mod_kendall_metric import mk_z, DEFAULT_K, normalized_z_for_plot

LA, LO   = CONUS_LAT_SLICE, CONUS_LON_SLICE
CACHE    = PROJECT_ROOT / "outputs/lag_may/heat_index_era5/ace2_daily_cache"
MODK_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall"
NPZ      = MODK_DIR / "monthly_raw_rc_fields.npz"
WRED     = LinearSegmentedColormap.from_list(
    "wred", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#a50f15"])
MONTHS   = {6: "June", 7: "July", 8: "August"}
TAG      = {6: "jun", 7: "jul", 8: "aug"}
MPOS     = {m: [i for i, (mm, _d) in enumerate(JJA_SEQ) if mm == m] for m in (6, 7, 8)}


# ── data / metrics ──────────────────────────────────────────────────────────

def load_ace2_conus():
    nyr, nm, nday = len(YEARS), N_MEMBERS, len(JJA_SEQ)
    out = None
    for yi, Y in enumerate(YEARS):
        for ii in range(nm):
            cp = CACHE / f"daily_y{Y}_mem{ii:02d}.nc"
            if not cp.exists():
                continue
            with xr.open_dataset(cp) as d:
                t = d["tmax_C"].values[:, LA, LO].astype(np.float32)
            if out is None:
                out = np.full((nyr, nm, nday, t.shape[1], t.shape[2]), np.nan, np.float32)
            n = min(t.shape[0], nday)
            out[yi, ii, :n] = t[:n]
        print(f"  loaded ACE2 {Y}", flush=True)
    return out


def fullwindow_thresh(data, sample_axes):
    grid = data.shape[-2:]
    moved = np.moveaxis(data, sample_axes, range(len(sample_axes))).reshape(-1, *grid)
    thr = np.empty(grid, np.float32)
    for i in range(grid[0]):
        thr[i] = np.nanpercentile(moved[:, i, :], HOT_PCT, axis=0)
    return thr


def monthly_freq_per_year(era5, ace2):
    thr_e = fullwindow_thresh(era5, (0, 1))
    thr_a = fullwindow_thresh(ace2, (0, 1, 2))
    out = {}
    for m in (6, 7, 8):
        pe = era5[:, MPOS[m]]; ve = np.isfinite(pe)
        fe = ((pe > thr_e[None, None]) & ve).sum(1) / np.maximum(ve.sum(1), 1)
        pa = ace2[:, :, MPOS[m]]; va = np.isfinite(pa)
        fa = ((pa > thr_a[None, None, None]) & va).sum((1, 2)) / np.maximum(va.sum((1, 2)), 1)
        out[m] = (fe.astype(np.float32), fa.astype(np.float32))
    return out


def rankcorr_maps(fe, fa, k=DEFAULT_K):
    _, nlat, nlon = fe.shape
    tau = np.full((nlat, nlon), np.nan, np.float32)
    tp  = np.full((nlat, nlon), np.nan, np.float32)
    z   = np.full((nlat, nlon), np.nan, np.float32)
    for i in range(nlat):
        for j in range(nlon):
            o = fe[:, i, j]; p = fa[:, i, j]
            if not (np.all(np.isfinite(o)) and np.all(np.isfinite(p))):
                continue
            if o.std() < 1e-12 or p.std() < 1e-12:
                continue
            t, pv = kendalltau(p, o)
            if np.isfinite(t):
                tau[i, j] = t; tp[i, j] = pv
            z[i, j] = mk_z(p, o, k)
    return tau, tp, z


# ── rendering: one figure per metric ────────────────────────────────────────

def _stipple(ax, sig, lat, lon):
    if sig is None or not np.any(sig):
        return
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.8, c="k", alpha=0.55, linewidths=0, zorder=6)


def _draw_box(ax, box):
    la0, la1, lo0, lo1 = box
    x0 = lo0 - 360.0 if lo0 > 180 else lo0
    x1 = lo1 - 360.0 if lo1 > 180 else lo1
    ax.add_patch(Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                           edgecolor="#00b050", linewidth=2.6, zorder=7))


def _map(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label, sig=None,
         mean_lbl=None, box=None, add_colorbar=True):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    ax.set_facecolor("white")
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1, interpolation="nearest")
    _stipple(ax, sig, lat, lon)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    if box is not None:
        _draw_box(ax, box)
    ax.set_title(title, fontsize=11)
    if add_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)
    if mean_lbl is not None:
        ax.legend([Line2D([], [], linestyle="none")], [mean_lbl],
                  loc="lower left", fontsize=8, handlelength=0, handletextpad=0,
                  framealpha=1.0, borderpad=0.5).set_zorder(7)
    return im


def render_month(month, fe, fa, tau, tp, zmap, lat, lon, land, zk):
    tag, name = TAG[month], MONTHS[month]
    suff = "37 year-pairs, no-LOO  (stipple = NOT significant)"

    # Figure 1 — frequency (ERA5 | ACE2)
    ec = np.nanmean(fe, 0) * 100.0; ac = np.nanmean(fa, 0) * 100.0
    both = np.concatenate([ec[np.isfinite(ec)], ac[np.isfinite(ac)]])
    fmax = float(np.nanpercentile(both, 98)) if both.size else 1.0
    box, box_score = find_hotspot_box(ec, lat, lon)
    la0, la1, lo0, lo1 = box
    print(f"  {name} ERA5 raw-frequency box: lat {la0:.1f}-{la1:.1f}  "
          f"lon {360-lo1:.0f}-{360-lo0:.0f}°W  mean={box_score:.2f}%", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    im = _map(axes[0], ec, lat, lon, "ERA5", WRED, 0.0, fmax, "% of month days",
              box=box, add_colorbar=False)
    _map(axes[1], ac, lat, lon, "ACE2", WRED, 0.0, fmax, "% of month days",
         add_colorbar=False)
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", location="bottom",
                        fraction=0.05, pad=0.08, shrink=0.5, aspect=40)
    cbar.set_label("% of month days")
    fig.suptitle(f"RAW heat-extreme {name} frequency (Tmax > full-JJA 90th-pct)", fontsize=13)
    fig.savefig(MODK_DIR / f"raw_frequency_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Figure 2 — Kendall τ
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tp) & (tp < 0.05))
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    _map(ax, tau, lat, lon, f"Kendall τ — RAW {name}  (ACE2 vs ERA5)", _TAU_CMAP,
         -1.0, 1.0, "τ", sig=tau_ns, mean_lbl=domain_scores_label("mean τ", tau, lat, land))
    fig.suptitle(suff, fontsize=10)
    fig.savefig(MODK_DIR / f"raw_kendalltau_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Figure 3 — modified-Kendall z, normalized to [-1, 1] for plotting.
    z_plot, z_scale = normalized_z_for_plot(zmap)
    z_ns = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    _map(ax, z_plot, lat, lon, f"Normalized modified-Kendall z (k={zk}) — RAW {name}", _TAU_CMAP,
         -1.0, 1.0, "normalized z", sig=z_ns,
         mean_lbl=domain_scores_label("mean norm z", z_plot, lat, land))
    fig.suptitle(suff, fontsize=10)
    fig.savefig(MODK_DIR / f"raw_modkendalltau_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote raw_frequency/kendalltau/modkendalltau_{tag}.png  z-plot-scale={z_scale:.3f}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────

def compute_fields():
    print("Loading ERA5 JJA Tmax (cache) ...", flush=True)
    era5 = _load_all_era5_jja(180, 360)[:, :, LA, LO]
    print("Loading ACE2 JJA Tmax (per-member 92-day cache) ...", flush=True)
    ace2 = load_ace2_conus()
    print("Computing per-year monthly frequencies ...", flush=True)
    freqs = monthly_freq_per_year(era5, ace2)
    save = {}
    for m in (6, 7, 8):
        fe, fa = freqs[m]
        print(f"  {MONTHS[m]}: τ / mod-z maps ...", flush=True)
        tau, tp, zmap = rankcorr_maps(fe, fa)
        save[f"fe{m}"] = fe; save[f"fa{m}"] = fa
        save[f"tau{m}"] = tau; save[f"tp{m}"] = tp; save[f"z{m}"] = zmap
    with xr.open_dataset(PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc") as d:
        save["lat"] = d["lat"].values[LA]; save["lon"] = d["lon"].values[LO]
    np.savez(NPZ, **save)
    print(f"cached fields -> {NPZ}", flush=True)
    return save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="reload cached fields from npz and only re-render figures")
    args = ap.parse_args()
    MODK_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        print(f"--replot: loading {NPZ}", flush=True)
        save = dict(np.load(NPZ))
    else:
        save = compute_fields()

    lat, lon = save["lat"], save["lon"]
    land = load_land_mask(lat, lon)
    for m in (6, 7, 8):
        render_month(m, save[f"fe{m}"], save[f"fa{m}"], save[f"tau{m}"], save[f"tp{m}"],
                     save[f"z{m}"], lat, lon, land, DEFAULT_K)
        print(f"  {MONTHS[m]} domain-mean: τ={cos_lat_mean(save[f'tau{m}'], lat):.3f}  "
              f"z={cos_lat_mean(save[f'z{m}'], lat):.3f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
