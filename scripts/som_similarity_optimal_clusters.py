#!/usr/bin/env python3
"""Optimal SOM size via REDCAP-style within/between similarity plot.

This is the SOM counterpart to redcap_optimal_clusters.py. It uses the same
grid-cell time-series feature matrix and the same within/between pattern-
correlation metric, but cluster labels come from a 1-D SOM (N x 1).

Outputs -> outputs/lag_may/cluster_analysis_sliding7d_conus[/_<domain>]/
             optimal_clusters_som.png
             optimal_clusters_som.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    SLIDING_DIR,
    CONUS_LAT_SLICE,
    CONUS_LON_SLICE,
    build_features,
    _apply_domain_mask,
)
from redcap_optimal_clusters import (  # noqa: E402
    _standardize_rows,
    select_optimal_intersection,
    plot_similarity,
)


def som_labels(features_temporal, n_nodes, seed):
    """Cluster grid-cell time series with an N x 1 SOM."""
    n_features = features_temporal.shape[1]
    som = MiniSom(
        n_nodes, 1, n_features,
        sigma=max(1.0, n_nodes / 4.0),
        learning_rate=0.5,
        random_seed=seed,
    )
    som.train_random(features_temporal, num_iteration=2000, verbose=False)
    return np.array([som.winner(x)[0] for x in features_temporal], dtype=np.int32)


def som_cluster_similarity(features, labels, weights, k, between_pctl=90.0):
    """Within similarity and upper-tail between-centroid similarity for SOM.

    The paper/REDCAP-style red curve uses the most-redundant cluster pairs,
    represented by the 90th percentile of centroid-pair correlations.
    """
    T = features.shape[1]
    within_vals, within_wts, centroids = [], [], []
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            continue
        Xc = features[m]
        wc = weights[m]
        wc_sum = float(wc.sum())
        centroid = (Xc * wc[:, None]).sum(axis=0) / wc_sum
        cs = (centroid - centroid.mean()) / max(centroid.std(), 1e-12)
        centroids.append(cs)
        if n < 2:
            continue
        Xs = _standardize_rows(Xc)
        corrs = (Xs @ cs) / T
        within_vals.append(float((corrs * wc).sum() / wc_sum))
        within_wts.append(wc_sum)

    within_vals = np.asarray(within_vals)
    within_wts = np.asarray(within_wts)
    if within_vals.size:
        w = within_wts / within_wts.sum()
        within_mean = float((within_vals * w).sum())
        within_std = float(np.sqrt((w * (within_vals - within_mean) ** 2).sum()))
    else:
        within_mean = within_std = np.nan

    C = np.asarray(centroids)
    if C.shape[0] >= 2:
        corr = (C @ C.T) / T
        iu = np.triu_indices(C.shape[0], k=1)
        pair = corr[iu]
        between_mean = float(np.percentile(pair, between_pctl))
        between_std = float(pair.std())
    else:
        between_mean = between_std = np.nan
    return within_mean, within_std, between_mean, between_std


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--conus", action="store_true", default=True,
                   help="Use the CONUS box (default)")
    p.add_argument("--global", dest="globaldomain", action="store_true",
                   help="Use the full global domain instead of CONUS")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--between-pctl", type=float, default=90.0,
                   help="Percentile of cluster-pair correlations for the red line")
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
    with xr.open_dataset(freq_nc) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        pred = ds["ace2_freq"].values.astype(np.float32)
        obs = ds["era5_freq"].values.astype(np.float32)

    if conus:
        pred = pred[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        obs = obs[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        lat = lat[CONUS_LAT_SLICE]
        lon = lon[CONUS_LON_SLICE]
        print(f"CONUS box: lat {lat[0]:.1f}..{lat[-1]:.1f}  "
              f"lon {lon[0]:.1f}..{lon[-1]:.1f}", flush=True)

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
    print(f"\n  N   within   between   (mean of {args.seeds} SOM seeds)", flush=True)
    for k in k_vals:
        wm_s, ws_s, bm_s, bs_s = [], [], [], []
        for seed in range(args.seeds):
            labels = som_labels(features_temporal, k, seed=42 + seed)
            k_eff = int(labels.max()) + 1
            wm, ws, bm, bs = som_cluster_similarity(
                features_temporal, labels, weights, k_eff,
                between_pctl=args.between_pctl,
            )
            wm_s.append(wm); ws_s.append(ws); bm_s.append(bm); bs_s.append(bs)
        within_m.append(float(np.nanmean(wm_s)))
        within_s.append(float(np.nanmean(ws_s)))
        between_m.append(float(np.nanmean(bm_s)))
        between_s.append(float(np.nanmean(bs_s)))
        print(f"  {k:2d}   {within_m[-1]:.3f}    {between_m[-1]:.3f}", flush=True)

    within_m = np.array(within_m)
    within_s = np.array(within_s)
    between_m = np.array(between_m)
    between_s = np.array(between_s)

    k_opt, band_lo, band_hi, k_cross = select_optimal_intersection(
        k_vals, within_m, within_s, between_m, between_s)
    print(f"\nCrossover N (between p{args.between_pctl:g} >= within): {k_cross}", flush=True)
    print(f"Optimal SOM N (red-blue intersection): {k_opt}  band {band_lo}-{band_hi}", flush=True)

    out_png = out_dir / "optimal_clusters_som.png"
    plot_similarity(
        k_vals, within_m, within_s, between_m, between_s,
        k_opt, band_lo, band_hi,
        "SOM map size (N×1)", out_png, opt_word="optimal N",
        between_label=f"Between clusters ({args.between_pctl:g}th pct of pair corr)",
    )
    print(f"wrote {out_png}", flush=True)

    out_json = out_dir / "optimal_clusters_som.json"
    out_json.write_text(json.dumps({
        "method": "SOM (N x 1); within = member-to-centroid corr, between = "
                  f"{args.between_pctl:g}th-pct cluster-pair corr",
        "domain": "conus" if conus else "global",
        "subdomain": args.domain,
        "n_valid": n_valid,
        "seeds": args.seeds,
        "between_percentile": args.between_pctl,
        "k_range": [args.kmin, args.kmax],
        "k_values": k_vals,
        "within_mean": within_m.tolist(),
        "within_std": within_s.tolist(),
        "between_mean": between_m.tolist(),
        "between_std": between_s.tolist(),
        "k_crossover": k_cross,
        "optimal_k": k_opt,
        "optimal_band": [band_lo, band_hi],
        "selection_rule": "red-blue intersection (upper-tail between-cluster similarity rises to meet within)",
    }, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
