#!/usr/bin/env python3
"""Reference-style SOM similarity: weather-typing of daily spatial maps.

Unlike som_optimal_clusters.py (which clusters grid cells by their time series),
this reproduces the *reference figure's* setup: the objects clustered are daily
spatial maps, a SOM node is a composite map (a weather regime), and within/
between use **spatial** pattern correlation. As the SOM grows, extra nodes
become near-duplicate composites → between-cluster correlation rises toward
within → the crossover that bounds the optimal map size.

Samples : daily JJA Tmax maps (CONUS) from combined_jja/tmax_jja_*.nc, one
          member trajectory (default member 12, the May-1-init run), as
          standardized anomalies (per-cell climatology removed, per-cell z).
Metric  : within/between spatial pattern correlation (same cluster_similarity
          helper, transposed so rows = maps, columns = grid cells).

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus/
            optimal_clusters_som_weatherregime.png / .json
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

from cluster_skill_analysis_sliding7d import CONUS_LAT_SLICE, CONUS_LON_SLICE  # noqa: E402
from redcap_optimal_clusters import cluster_similarity, find_knee  # noqa: E402
from som_optimal_clusters import som_labels  # noqa: E402

COMBINED_DIR = PROJECT_ROOT / "outputs/lag_may/combined_jja"
YEARS = list(range(1980, 2017))


def load_daily_maps(member: int, conus: bool, standardize_cells: bool = False):
    """Stack daily JJA Tmax maps across years → (n_samples, n_cells) anomalies."""
    samples, lat, lon = [], None, None
    for y in YEARS:
        f = COMBINED_DIR / f"tmax_jja_{y}.nc"
        if not f.exists():
            continue
        da = xr.open_dataset(f)["TMP2m"]            # (member, time, lat, lon)
        if conus:
            da = da.isel(lat=CONUS_LAT_SLICE, lon=CONUS_LON_SLICE)
        arr = da.isel(member=member).values         # (time, lat, lon)
        lat = da["lat"].values; lon = da["lon"].values
        samples.append(arr)
    maps = np.concatenate(samples, axis=0)          # (n_days_total, lat, lon)

    nlat, nlon = maps.shape[1:]
    flat = maps.reshape(maps.shape[0], nlat * nlon)
    # keep cells finite in every sample
    valid = np.all(np.isfinite(flat), axis=0)
    X = flat[:, valid]                              # (n_samples, n_cells)
    # drop all-NaN samples (out-of-range member days)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    # anomalies: remove per-cell climatology. Whitening (per-cell std) is
    # optional — whitening orthogonalizes composites (between→0); keeping raw
    # amplitude retains the dominant mode so composites stay positively
    # correlated and between rises with N, like the reference.
    mu = X.mean(axis=0, keepdims=True)
    X = X - mu
    if standardize_cells:
        sd = X.std(axis=0, keepdims=True)
        X = X / np.where(sd < 1e-9, 1.0, sd)
    return X.astype(np.float32), int(valid.sum())


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--member", type=int, default=12)
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--standardize-cells", action="store_true",
                   help="Whiten each grid cell (orthogonalizes composites; between→0)")
    p.add_argument("--global", dest="globaldomain", action="store_true")
    args = p.parse_args()
    conus = not args.globaldomain

    out_dir = PROJECT_ROOT / ("outputs/lag_may/cluster_analysis_sliding7d_conus"
                              if conus else "outputs/lag_may/cluster_analysis_sliding7d")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading daily JJA Tmax maps (member {args.member}) ...", flush=True)
    X, n_cells = load_daily_maps(args.member, conus, standardize_cells=args.standardize_cells)
    n_samples = X.shape[0]
    print(f"  samples (daily maps) = {n_samples}   cells (features) = {n_cells}", flush=True)

    weights = np.ones(n_samples, dtype=np.float64)   # each day weighted equally
    k_vals = list(range(args.kmin, args.kmax + 1))
    within_m, within_s, between_m, between_s = [], [], [], []
    print(f"\n  N   within   between   (avg {args.seeds} seeds, spatial pattern corr)", flush=True)
    for k in k_vals:
        wm_s, ws_s, bm_s, bs_s = [], [], [], []
        for seed in range(args.seeds):
            labels = som_labels(X, k, seed=42 + seed)
            k_eff = int(labels.max()) + 1
            wm, ws, bm, bs = cluster_similarity(X, labels, weights, k_eff)
            wm_s.append(wm); ws_s.append(ws); bm_s.append(bm); bs_s.append(bs)
        within_m.append(np.nanmean(wm_s)); within_s.append(np.nanmean(ws_s))
        between_m.append(np.nanmean(bm_s)); between_s.append(np.nanmean(bs_s))
        print(f"  {k:2d}   {within_m[-1]:.3f}    {between_m[-1]:.3f}", flush=True)

    within_m = np.array(within_m); within_s = np.array(within_s)
    between_m = np.array(between_m); between_s = np.array(between_s)

    cross_idx = np.where(between_m >= within_m)[0]
    k_cross = int(k_vals[cross_idx[0]]) if cross_idx.size else None
    upto = [i for i, k in enumerate(k_vals) if (k_cross is None or k <= k_cross)]
    k_opt = find_knee([k_vals[i] for i in upto], within_m[upto])
    print(f"\nCrossover N: {k_cross}   Optimal N (knee): {k_opt}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    BLUE, RED = "#1f5fd0", "#e8202a"
    band_lo, band_hi = k_opt, (k_cross if k_cross is not None else k_opt + 1)
    if band_hi <= band_lo:
        band_hi = band_lo + 1
    ax.axvspan(band_lo - 0.5, band_hi - 0.5, color="0.85", zorder=0)
    ax.fill_between(k_vals, within_m - within_s, within_m + within_s, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(k_vals, between_m - between_s, between_m + between_s, color=RED, alpha=0.18, lw=0)
    ax.plot(k_vals, within_m, "-", color=BLUE, lw=2.5, label="Within clusters")
    ax.plot(k_vals, between_m, "-", color=RED, lw=2.5, label="Between clusters")
    ax.axvline(k_opt, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.annotate(f"optimal N = {k_opt}", xy=(k_opt, within_m.min()),
                xytext=(k_opt + 0.2, within_m.min()), fontsize=9, color="0.2")
    ax.set_xlabel("SOM map size (N×1)", fontsize=12)
    ax.set_ylabel("Pattern correlation", fontsize=12)
    ax.set_xticks(k_vals)
    ax.set_title("(A) Similarity", fontsize=14, weight="bold", loc="left")
    ax.legend(fontsize=11, loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png = out_dir / "optimal_clusters_som_weatherregime.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)

    (out_dir / "optimal_clusters_som_weatherregime.json").write_text(json.dumps({
        "method": "SOM (N×1) weather-typing of daily JJA Tmax anomaly maps; spatial pattern correlation",
        "member": args.member, "n_samples": n_samples, "n_cells": n_cells,
        "seeds": args.seeds, "k_values": k_vals,
        "within_mean": within_m.tolist(), "within_std": within_s.tolist(),
        "between_mean": between_m.tolist(), "between_std": between_s.tolist(),
        "k_crossover": k_cross, "optimal_k": k_opt,
    }, indent=2))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
