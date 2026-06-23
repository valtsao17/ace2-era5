#!/usr/bin/env python3
"""Per-cluster precision/recall for JJA HHE day-level classification.

The cluster counterpart of precision_recall_jja_sliding7d.py (which is the
"no cluster", grid-point version). Here the same day-level confusion counts
(TP/FP/FN/TN per grid cell, pooled over 37×92 day-cells) are *aggregated within
each cluster*, then precision/recall are formed from the cluster totals:

    precision(cluster) = ΣTP / (ΣTP + ΣFP)
    recall(cluster)    = ΣTP / (ΣTP + ΣFN)

Pooling the raw counts (not averaging per-cell rates) is the statistically
correct way to roll precision/recall up to a region — it weights each cell by
how many events it actually contributes.

Cluster assignments come from the cached clustering sweep (the same K-means /
SOM / REDCAP optimum used for the τ-per-cluster maps), so these maps are the
precision/recall analogue of tau_kmeans_*.png etc.

Sources:
  counts  → seasonal_jja_sliding7d/precision_recall_jja_seasonal.nc (tp,fp,fn,tn)
  labels  → <cluster_dir>/sweep_cache.npz  (optimal km/som/rc labels, valid_mask)

Default cluster_dir = cluster_analysis_sliding7d_conus  (CONUS).

Outputs → <cluster_dir>/
  precision_<method>_k<k>.png , recall_<method>_k<k>.png
  precision_recall_clusters_summary.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _expand_labels, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from seasonal_jja_skill import _SKILL_CMAP, cos_lat_mean

PR_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/precision_recall_jja_seasonal.nc"


def _pool_metric(counts_num, counts_den, labels_full, n_clusters):
    """Cluster-pooled rate map + per-cluster value vector.

    counts_num/_den are (lat,lon) integer count grids; rate = ΣTP/(ΣTP+Σother).
    labels_full is (lat,lon) with -1 outside the valid mask.
    """
    n_lat, n_lon = labels_full.shape
    rate_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    rate_cl  = np.full(n_clusters, np.nan, dtype=np.float32)
    for c in range(n_clusters):
        m = labels_full == c
        if not m.any():
            continue
        num = float(np.nansum(counts_num[m]))
        den = float(np.nansum(counts_den[m]))
        if den <= 0:
            continue
        r = num / den
        rate_cl[c] = r
        rate_map[m] = r
    return rate_map, rate_cl


def _annotate_clusters(ax, labels_2d, val_cl, lon2d, lat2d, fontsize=7):
    for cl, v in enumerate(val_cl):
        if not np.isfinite(v):
            continue
        mask = labels_2d == cl
        if mask.sum() < 3:
            continue
        ax.text(float(lon2d[mask].mean()), float(lat2d[mask].mean()), f"{v:.2f}",
                fontsize=fontsize, ha="center", va="center", color="black",
                weight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def plot_cluster_metric(rate_map, labels_full, val_cl, lat, lon, title, cbar_label,
                        domain_mean, out):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    e = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
         float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(rate_map, origin="lower", extent=e, aspect="equal",
                   vmin=0.0, vmax=1.0, cmap=_SKILL_CMAP, zorder=1,
                   interpolation="nearest")
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    _annotate_clusters(ax, labels_full, val_cl, LON2D, LAT2D)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)
    ax.text(0.02, 0.03, f"Domain-pooled {cbar_label} = {domain_mean:.3f}",
            transform=ax.transAxes, fontsize=10, weight="bold", ha="left", va="bottom",
            zorder=7, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                                edgecolor="black"))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def _optimal(d, prefix):
    """Return (labels, k, tag) for the max-tau_domain result of a method, or None."""
    key = f"n_{prefix}"
    if key not in d:
        return None
    n = int(d[key][0])
    best = None
    for i in range(n):
        td = float(d[f"{prefix}_{i}_tau_domain"][0])
        if not np.isfinite(td):
            continue
        if best is None or td > best[0]:
            k = int(d[f"{prefix}_{i}_k"][0])
            tag = f"k{k}"
            if f"{prefix}_{i}_m" in d:
                tag = f"{int(d[f'{prefix}_{i}_m'][0])}x{int(d[f'{prefix}_{i}_n'][0])}"
            best = (td, d[f"{prefix}_{i}_labels"], k, tag)
    if best is None:
        return None
    return best[1], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-dir", default="cluster_analysis_sliding7d_conus",
                    help="output dir under outputs/lag_may holding sweep_cache.npz")
    ap.add_argument("--conus", action="store_true", default=True,
                    help="cluster cache is CONUS-sliced (default; counts sliced to match)")
    ap.add_argument("--global", dest="is_global", action="store_true",
                    help="cluster cache spans the full global grid (no slice)")
    args = ap.parse_args()

    cdir = PROJECT_ROOT / "outputs/lag_may" / args.cluster_dir
    cache = cdir / "sweep_cache.npz"
    if not cache.exists():
        print(f"ERROR: {cache} not found — run cluster_skill_analysis_sliding7d.py first.",
              flush=True)
        return

    d = np.load(cache, allow_pickle=False)
    lat, lon = d["lat"], d["lon"]
    valid_mask = d["valid_mask"].astype(bool)
    n_lat, n_lon = len(lat), len(lon)

    # day-level confusion counts (global) → slice to the cluster domain
    with xr.open_dataset(PR_NC) as pr:
        glat = pr["lat"].values
        tp = pr["tp"].values; fp = pr["fp"].values; fn = pr["fn"].values
    if not args.is_global and len(lat) != len(glat):
        tp = tp[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        fp = fp[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        fn = fn[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    assert tp.shape == (n_lat, n_lon), f"count grid {tp.shape} != cluster grid {(n_lat, n_lon)}"

    tp_pos = tp + fp   # precision denominator
    tp_fn  = tp + fn   # recall denominator

    summary = {"cluster_dir": args.cluster_dir, "methods": {}}
    for prefix, name in (("km", "kmeans"), ("som", "som"), ("rc", "redcap")):
        opt = _optimal(d, prefix)
        if opt is None:
            continue
        labels, k, tag = opt
        labels_full = _expand_labels(labels, valid_mask, n_lat, n_lon)
        nk = int(labels_full.max()) + 1

        prec_map, prec_cl = _pool_metric(tp, tp_pos, labels_full, nk)
        rec_map,  rec_cl  = _pool_metric(tp, tp_fn,  labels_full, nk)

        # domain-pooled scalar (all cells in the valid mask)
        valid = labels_full >= 0
        prec_dom = float(np.nansum(tp[valid]) / max(np.nansum(tp_pos[valid]), 1))
        rec_dom  = float(np.nansum(tp[valid]) / max(np.nansum(tp_fn[valid]), 1))

        plot_cluster_metric(prec_map, labels_full, prec_cl, lat, lon,
                            f"{name} precision per cluster  ({tag})", "precision",
                            prec_dom, cdir / f"precision_{name}_{tag}.png")
        plot_cluster_metric(rec_map, labels_full, rec_cl, lat, lon,
                            f"{name} recall per cluster  ({tag})", "recall",
                            rec_dom, cdir / f"recall_{name}_{tag}.png")

        summary["methods"][name] = {
            "k": k, "tag": tag,
            "precision_domain_pooled": round(prec_dom, 4),
            "recall_domain_pooled": round(rec_dom, 4),
            "precision_cluster_mean": round(float(np.nanmean(prec_cl)), 4),
            "recall_cluster_mean": round(float(np.nanmean(rec_cl)), 4),
        }
        print(f"  {name:7s} {tag:8s}  precision_pooled={prec_dom:.3f}  recall_pooled={rec_dom:.3f}",
              flush=True)

    (cdir / "precision_recall_clusters_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {cdir / 'precision_recall_clusters_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
