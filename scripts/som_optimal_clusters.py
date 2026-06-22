#!/usr/bin/env python3
"""Optimal SOM cluster count via within/between pattern-correlation similarity.

The SOM counterpart of redcap_optimal_clusters.py — and the direct analogue of
the reference "(A) Similarity" panel, which sweeps a 1-D SOM of map size N×1.

For each N (number of SOM nodes, N×1 map) we train a SOM on the standardized
37-year HHE-frequency time series at each CONUS grid cell, assign each cell to
its best-matching unit, then compute:

  within(N)  = cos-lat-weighted mean over nodes of the mean correlation between
               each member cell's series and its node centroid (compactness).
  between(N) = mean correlation between distinct node centroids (redundancy).

SOM training is stochastic, so each N is trained with several random seeds and
the curves are averaged (band = mean cluster/pair spread, as in REDCAP). The
optimal N is the **knee of the within-cluster curve**, bounded at/below the
crossover where between rises to meet within.

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus[/_<domain>]/
            optimal_clusters_som.png
            optimal_clusters_som.json
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    SLIDING_DIR, CONUS_LAT_SLICE, CONUS_LON_SLICE,
    build_features, _apply_domain_mask,
)
from redcap_optimal_clusters import cluster_similarity, find_knee  # noqa: E402


def som_labels(features, n_nodes, seed):
    """Train an N×1 SOM and return the best-matching-unit label per sample."""
    n_features = features.shape[1]
    som = MiniSom(n_nodes, 1, n_features,
                  sigma=max(1.0, n_nodes / 4.0), learning_rate=0.5,
                  random_seed=seed)
    som.train_random(features, num_iteration=2000, verbose=False)
    return np.array([som.winner(x)[0] for x in features])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--global", dest="globaldomain", action="store_true",
                   help="Full global domain instead of CONUS (default CONUS)")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5,
                   help="SOM trainings averaged per N (stochastic init)")
    args = p.parse_args()
    conus = not args.globaldomain

    domain_suffix = "" if args.domain == "all" else f"_{args.domain}"
    if conus:
        out_dir = PROJECT_ROOT / f"outputs/lag_may/cluster_analysis_sliding7d_conus{domain_suffix}"
    else:
        out_dir = PROJECT_ROOT / f"outputs/lag_may/cluster_analysis_sliding7d{domain_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_nc = SLIDING_DIR / "jja_seasonal_freqs.nc"
    print(f"Loading {freq_nc}", flush=True)
    ds = xr.open_dataset(freq_nc)
    lat = ds["lat"].values
    lon = ds["lon"].values
    pred = ds["ace2_freq"].values.astype(np.float32)
    obs = ds["era5_freq"].values.astype(np.float32)
    ds.close()

    if conus:
        pred = pred[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        obs = obs[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        lat = lat[CONUS_LAT_SLICE]
        lon = lon[CONUS_LON_SLICE]
        print(f"CONUS box: lat {lat[0]:.1f}..{lat[-1]:.1f}  lon {lon[0]:.1f}..{lon[-1]:.1f}", flush=True)

    if args.domain != "all":
        pred, obs = _apply_domain_mask(args.domain, lat, lon, pred, obs)
        print(f"Domain: {args.domain}-only", flush=True)

    _, features_temporal, valid_mask = build_features(pred, lat, lon)
    n_valid = int(valid_mask.sum())
    print(f"features_temporal: {features_temporal.shape}  n_valid={n_valid}", flush=True)

    lat_g = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], valid_mask.shape)
    weights = lat_g[valid_mask].astype(np.float64)

    k_vals = list(range(args.kmin, args.kmax + 1))
    within_m, within_s, between_m, between_s = [], [], [], []
    print(f"\n  N   within   between   (avg of {args.seeds} SOM seeds)", flush=True)
    for k in k_vals:
        wm_s, ws_s, bm_s, bs_s = [], [], [], []
        for seed in range(args.seeds):
            labels = som_labels(features_temporal, k, seed=42 + seed)
            k_eff = int(labels.max()) + 1
            wm, ws, bm, bs = cluster_similarity(features_temporal, labels, weights, k_eff)
            wm_s.append(wm); ws_s.append(ws); bm_s.append(bm); bs_s.append(bs)
        within_m.append(np.nanmean(wm_s)); within_s.append(np.nanmean(ws_s))
        between_m.append(np.nanmean(bm_s)); between_s.append(np.nanmean(bs_s))
        print(f"  {k:2d}   {within_m[-1]:.3f}    "
              f"{between_m[-1] if np.isfinite(between_m[-1]) else float('nan'):.3f}", flush=True)

    within_m = np.array(within_m); within_s = np.array(within_s)
    between_m = np.array(between_m); between_s = np.array(between_s)

    cross_idx = np.where(between_m >= within_m)[0]
    k_cross = int(k_vals[cross_idx[0]]) if cross_idx.size else None
    upto = [i for i, k in enumerate(k_vals) if (k_cross is None or k <= k_cross)]
    k_opt = find_knee([k_vals[i] for i in upto], within_m[upto])

    print(f"\nCrossover N (between >= within): {k_cross}", flush=True)
    print(f"Optimal SOM N (within-curve knee): {k_opt}", flush=True)

    # ── plot (reference "(A) Similarity" style) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.2))
    BLUE, RED = "#1f5fd0", "#e8202a"

    band_lo, band_hi = k_opt, (k_cross if k_cross is not None else k_opt + 1)
    if band_hi <= band_lo:
        band_hi = band_lo + 1
    ax.axvspan(band_lo - 0.5, band_hi - 0.5, color="0.85", zorder=0)

    ax.fill_between(k_vals, within_m - within_s, within_m + within_s,
                    color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(k_vals, between_m - between_s, between_m + between_s,
                    color=RED, alpha=0.18, lw=0)
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
    out_png = out_dir / "optimal_clusters_som.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)

    summary = {
        "method": "SOM (N×1) within/between pattern correlation",
        "domain": "conus" if conus else "global",
        "subdomain": args.domain,
        "n_valid": n_valid,
        "seeds": args.seeds,
        "k_range": [args.kmin, args.kmax],
        "k_values": k_vals,
        "within_mean": within_m.tolist(),
        "within_std": within_s.tolist(),
        "between_mean": between_m.tolist(),
        "between_std": between_s.tolist(),
        "k_crossover": k_cross,
        "optimal_k": k_opt,
        "selection_rule": "knee of within-cluster curve, bounded at/below crossover",
    }
    (out_dir / "optimal_clusters_som.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'optimal_clusters_som.json'}", flush=True)


if __name__ == "__main__":
    main()
