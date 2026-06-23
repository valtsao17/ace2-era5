#!/usr/bin/env python3
"""Render the CONUS cluster maps at the *similarity-optimal* cluster count.

The main sweep (cluster_skill_analysis_sliding7d.py) chooses which cluster map to
draw by argmax domain-mean τ, which rewards the fewest clusters and — for REDCAP,
whose distance-percentile sweep never merges below ~62 clusters — gets stuck at
k=62. That number is an artifact, not a principled cluster count.

The principled count comes from the within/between pattern-correlation knee in
optimal_clusters_{redcap,som}.py, which lands at **k=8** for both REDCAP and SOM.
This script clusters all three methods at that fixed k and regenerates:
  - the cluster-assignment map,
  - the per-cluster τ map,
  - the per-cluster precision and recall maps (day-level confusion counts pooled
    within each cluster).

REDCAP is cut from a freshly fit queen-connectivity Ward dendrogram with
criterion="maxclust", t=k (the only way to reach k=8); K-means and SOM (k×1,
matching the 1-D SOM used by the similarity analysis) are fit directly.

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus/optimal_k{K}/
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    build_features, eval_cluster_skill, domain_mean_tau, _expand_labels,
    build_queen_connectivity, _sklearn_to_scipy_linkage,
    plot_cluster_tau_map, _cluster_map_figure, _TAU_CMAP,
    CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from precision_recall_clusters import _pool_metric, plot_cluster_metric, PR_NC

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
CONUS_DIR   = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_sliding7d_conus"
K = 8


def kmeans_labels(features_full, k):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                         batch_size=min(4096, features_full.shape[0]))
    return km.fit_predict(features_full)


def som_labels(features_temporal, k):
    # k×1 (1-D) SOM, matching optimal_clusters_som.py
    som = MiniSom(k, 1, features_temporal.shape[1], sigma=1.0,
                  learning_rate=0.5, random_seed=42)
    som.train_random(features_temporal, num_iteration=2000, verbose=False)
    bmu = np.array([som.winner(x) for x in features_temporal])
    return bmu[:, 0] * 1 + bmu[:, 1]


def redcap_labels(features_temporal, valid_mask, k):
    conn = build_queen_connectivity(valid_mask)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                  linkage="ward", connectivity=conn,
                                  compute_full_tree=True)
    agg.fit(features_temporal)
    linkage_mat = _sklearn_to_scipy_linkage(agg.children_, agg.distances_,
                                            conn.shape[0])
    return fcluster(linkage_mat, t=k, criterion="maxclust") - 1


def main():
    out_dir = CONUS_DIR / f"optimal_k{K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(SLIDING_DIR / "jja_seasonal_freqs.nc")
    lat = ds["lat"].values[CONUS_LAT_SLICE]
    lon = ds["lon"].values[CONUS_LON_SLICE]
    pred = ds["ace2_freq"].values[:, CONUS_LAT_SLICE, CONUS_LON_SLICE].astype(np.float32)
    obs  = ds["era5_freq"].values[:, CONUS_LAT_SLICE, CONUS_LON_SLICE].astype(np.float32)
    ds.close()
    n_lat, n_lon = len(lat), len(lon)

    features_full, features_temporal, valid_mask = build_features(pred, lat, lon)
    print(f"features {features_full.shape}  valid {valid_mask.sum()}", flush=True)

    # day-level confusion counts (global) sliced to CONUS, for per-cluster P/R
    with xr.open_dataset(PR_NC) as pr:
        tp = pr["tp"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        fp = pr["fp"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        fn = pr["fn"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    tp_pos, tp_fn = tp + fp, tp + fn

    methods = {
        "kmeans": (kmeans_labels(features_full, K), f"k{K}", "K-means"),
        "som":    (som_labels(features_temporal, K), f"{K}x1", "SOM"),
        "redcap": (redcap_labels(features_temporal, valid_mask, K), f"k{K}", "REDCAP Ward"),
    }

    summary = {"k": K, "selection": "similarity-knee optimal (optimal_clusters_*)",
               "methods": {}}
    for name, (labels, tag, nice) in methods.items():
        nk = int(labels.max()) + 1
        tau_map, _bss, tau_cl, _bsscl, sizes = eval_cluster_skill(
            pred, obs, labels, valid_mask, lat, nk)
        tau_dom = domain_mean_tau(tau_map, lat)
        labels_full = _expand_labels(labels, valid_mask, n_lat, n_lon)

        # cluster-assignment map
        _cluster_map_figure(labels_full, lat, lon, nk,
                            f"{nice} cluster assignment  ({tag}, similarity-optimal)",
                            out_dir / f"{name}_cluster_map_{tag}.png")
        # per-cluster τ
        plot_cluster_tau_map(tau_map, labels_full, tau_cl, tau_dom, lat, lon,
                             f"{nice} τ per cluster  ({tag}, similarity-optimal)",
                             out_dir / f"tau_{name}_{tag}.png")
        # per-cluster precision / recall (pooled counts)
        prec_map, prec_cl = _pool_metric(tp, tp_pos, labels_full, nk)
        rec_map,  rec_cl  = _pool_metric(tp, tp_fn,  labels_full, nk)
        valid = labels_full >= 0
        prec_dom = float(np.nansum(tp[valid]) / max(np.nansum(tp_pos[valid]), 1))
        rec_dom  = float(np.nansum(tp[valid]) / max(np.nansum(tp_fn[valid]), 1))
        plot_cluster_metric(prec_map, labels_full, prec_cl, lat, lon,
                            f"{nice} precision per cluster  ({tag})", "precision",
                            prec_dom, out_dir / f"precision_{name}_{tag}.png")
        plot_cluster_metric(rec_map, labels_full, rec_cl, lat, lon,
                            f"{nice} recall per cluster  ({tag})", "recall",
                            rec_dom, out_dir / f"recall_{name}_{tag}.png")

        summary["methods"][name] = {
            "tag": tag, "n_clusters": nk,
            "tau_domain": round(float(tau_dom), 4),
            "precision_domain_pooled": round(prec_dom, 4),
            "recall_domain_pooled": round(rec_dom, 4),
        }
        print(f"  {nice:11s} {tag:5s}  τ={tau_dom:.3f}  "
              f"precision={prec_dom:.3f}  recall={rec_dom:.3f}", flush=True)

    (out_dir / "cluster_optimal_k_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'cluster_optimal_k_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
