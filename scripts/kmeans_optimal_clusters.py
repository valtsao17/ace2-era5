#!/usr/bin/env python3
"""Optimal k-means cluster count via within/between pattern-correlation similarity.

The k-means counterpart of redcap_optimal_clusters.py / som_optimal_clusters.py,
reproducing the reference "(A) Similarity" panel. For each K we run k-means on
the standardized 37-year HHE-frequency time series at each CONUS grid cell, then:

  within(K)  = cos-lat-weighted mean over clusters of the mean correlation
               between each member cell's series and its cluster centroid.
  between(K) = 90th-percentile of the correlations between distinct cluster
               centroids (the most-similar/redundant cluster pair) — so the red
               line rises to MEET the within (blue) line.

k-means is stochastic, so each K is run with several random seeds; the solid
line is the median across seeds and the band is the ensemble spread. The optimal
K is the **red∩blue intersection**.

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus[/_<domain>]/
            optimal_clusters_kmeans.png
            optimal_clusters_kmeans.json
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    SLIDING_DIR, CONUS_LAT_SLICE, CONUS_LON_SLICE,
    build_features, _apply_domain_mask,
)
from redcap_optimal_clusters import (  # noqa: E402
    cluster_similarity, select_optimal_intersection, plot_similarity,
)


def kmeans_labels(features, k, seed):
    """k-means best-matching-cluster label per sample (single random init)."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=300)
    return km.fit_predict(features)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--global", dest="globaldomain", action="store_true",
                   help="Full global domain instead of CONUS (default CONUS)")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5,
                   help="k-means runs averaged per K (stochastic init)")
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
    print(f"\n  K   within   between   (median of {args.seeds} k-means seeds)", flush=True)
    for k in k_vals:
        wm_s, bm_s = [], []
        for seed in range(args.seeds):
            labels = kmeans_labels(features_temporal, k, seed=42 + seed)
            k_eff = int(labels.max()) + 1
            wm, _ws, bm, _bs = cluster_similarity(features_temporal, labels, weights, k_eff)
            wm_s.append(wm); bm_s.append(bm)
        within_m.append(np.nanmedian(wm_s)); within_s.append(np.nanstd(wm_s))
        between_m.append(np.nanmedian(bm_s)); between_s.append(np.nanstd(bm_s))
        print(f"  {k:2d}   {within_m[-1]:.3f}    "
              f"{between_m[-1] if np.isfinite(between_m[-1]) else float('nan'):.3f}", flush=True)

    within_m = np.array(within_m); within_s = np.array(within_s)
    between_m = np.array(between_m); between_s = np.array(between_s)

    k_opt, band_lo, band_hi, k_cross = select_optimal_intersection(
        k_vals, within_m, within_s, between_m, between_s)
    print(f"\nCrossover K (between p90 >= within): {k_cross}", flush=True)
    print(f"Optimal k-means K (red∩blue intersection): {k_opt}  band {band_lo}-{band_hi}",
          flush=True)

    out_png = out_dir / "optimal_clusters_kmeans.png"
    plot_similarity(k_vals, within_m, within_s, between_m, between_s,
                    k_opt, band_lo, band_hi, "Number of clusters (k-means)", out_png,
                    opt_word="optimal K")
    print(f"wrote {out_png}", flush=True)

    summary = {
        "method": "k-means; within = member↔centroid corr, between = 90th-pct cluster-pair corr",
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
        "optimal_band": [band_lo, band_hi],
        "selection_rule": "red∩blue intersection (between 90th-pct rises to meet within)",
    }
    (out_dir / "optimal_clusters_kmeans.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'optimal_clusters_kmeans.json'}", flush=True)


if __name__ == "__main__":
    main()
