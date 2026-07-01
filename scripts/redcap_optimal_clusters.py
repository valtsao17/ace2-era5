#!/usr/bin/env python3
"""Optimal REDCAP cluster count via within/between pattern-correlation similarity.

Recreates the "(A) Similarity" panel (within-cluster vs between-cluster pattern
correlation as a function of the number of clusters) — but for the REDCAP
(Ward, spatially constrained) clustering instead of a 1-D SOM.

Method
------
The grid-point "pattern" is each cell's standardized 37-year HHE seasonal
frequency time series (same `features_temporal` used by the REDCAP sweep). For a
given cluster count K we cut the full Ward dendrogram with
`fcluster(..., criterion="maxclust")` to obtain exactly K spatially-contiguous
clusters, then:

  within(K)  = cos-lat-weighted mean over clusters of the mean Pearson
               correlation between each member's time series and its cluster
               centroid (how self-similar a cluster is).
  between(K) = mean Pearson correlation between distinct cluster centroids
               (how similar clusters are to each other).

As K grows, within rises (clusters tighten) and between rises (clusters become
redundant). The optimal K is the **knee of the within-cluster curve** — where
compactness gains level off — constrained to lie at or below the crossover
(where between rises to meet within).

Reuses feature/connectivity/linkage helpers from cluster_skill_analysis_sliding7d.

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus[/_<domain>]/
            optimal_clusters_redcap.png
            optimal_clusters_redcap.json
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    SLIDING_DIR,
    CONUS_LAT_SLICE,
    CONUS_LON_SLICE,
    build_features,
    build_queen_connectivity,
    _sklearn_to_scipy_linkage,
    _apply_domain_mask,
)


# ── similarity metrics ──────────────────────────────────────────────────────────

def _standardize_rows(X):
    """Z-score each row across its features (axis=1) for Pearson correlation."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def cluster_similarity(features, labels, weights, k, between_pctl=90.0):
    """Within- and between-cluster pattern correlation for one K.

    features : (n_valid, T) grid-point time-series patterns
    labels   : (n_valid,) cluster id in [0, k)
    weights  : (n_valid,) cos-lat area weights
    between_pctl : percentile of the cluster-pair correlation distribution used
                   for the "between" value (the reference uses the 90th
                   percentile — the most-similar/redundant cluster pair — so the
                   red line rises to meet the within-cluster blue line and they
                   intersect; the mean never would, since clustering keeps
                   within >= between).
    Returns (within_mean, within_std, between_value, between_std).
    Per-cluster within values are weighted by cluster area; the bands are the
    weighted std across clusters (within) and the std over centroid pairs
    (between), mirroring the shaded envelopes in the reference figure.
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
        # area-weighted centroid time series, then standardized for correlation
        centroid = (Xc * wc[:, None]).sum(axis=0) / wc_sum
        cs = (centroid - centroid.mean()) / max(centroid.std(), 1e-12)
        centroids.append(cs)
        if n < 2:
            # singleton: self-correlation is trivially 1; exclude from within
            continue
        Xs = _standardize_rows(Xc)
        corrs = (Xs @ cs) / T                      # Pearson(member, centroid)
        within_c = float((corrs * wc).sum() / wc_sum)
        within_vals.append(within_c)
        within_wts.append(wc_sum)

    within_vals = np.asarray(within_vals)
    within_wts = np.asarray(within_wts)
    if within_vals.size:
        w = within_wts / within_wts.sum()
        within_mean = float((within_vals * w).sum())
        within_std = float(np.sqrt((w * (within_vals - within_mean) ** 2).sum()))
    else:
        within_mean = within_std = np.nan

    C = np.asarray(centroids)                       # (k_eff, T) standardized
    if C.shape[0] >= 2:
        corr = (C @ C.T) / T
        iu = np.triu_indices(C.shape[0], k=1)
        pair = corr[iu]
        # 90th-percentile cluster-pair correlation = the most-redundant pair
        between_value = float(np.percentile(pair, between_pctl))
        between_std = float(pair.std())
    else:
        between_value = between_std = np.nan

    return within_mean, within_std, between_value, between_std


def find_knee(k_vals, y):
    """Kneedle for an increasing, concave curve: the point of maximum distance
    above the chord joining the endpoints (max compactness 'diminishing return').
    """
    x = np.asarray(k_vals, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    x, y, kk = x[ok], y[ok], np.asarray(k_vals)[ok]
    if x.size < 3:
        return int(kk[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return int(kk[np.argmax(yn - xn)])


def select_optimal_intersection(k_vals, within_m, within_s, between_m, between_s):
    """Optimal cluster count = where the between line (red, 90th-pct cluster-pair
    correlation) rises to MEET the within line (blue), i.e. the reference
    figure's red∩blue intersection.

    Returns (k_opt, band_lo, band_hi, k_cross):
      k_cross  : first K with between >= within (the crossover; None if never)
      k_opt    : the optimal K (k_cross, or the closest approach if no crossover)
      band_lo/hi: the grey 'intersection' band = contiguous K where the within
                  and between spread envelopes overlap.
    """
    k  = np.asarray(k_vals)
    wm = np.asarray(within_m, float); ws = np.nan_to_num(np.asarray(within_s, float))
    bm = np.asarray(between_m, float); bs = np.nan_to_num(np.asarray(between_s, float))

    diff = bm - wm
    cross = np.where(np.isfinite(diff) & (diff >= 0.0))[0]
    if cross.size:
        k_cross = int(k[cross[0]])
        k_opt = k_cross
        # band = contiguous K where the spread envelopes overlap, anchored on
        # the crossover (the genuine 'intersection' region)
        overlap = (np.isfinite(wm) & np.isfinite(bm)
                   & ((wm - ws) <= (bm + bs)) & ((bm - bs) <= (wm + ws)))
        idx = np.where(overlap)[0]
        if idx.size:
            band_lo, band_hi = int(k[idx[0]]), int(k[idx[-1]])
        else:
            band_lo = band_hi = k_opt
    else:
        # within stays above between throughout -> no real intersection;
        # report the closest approach and DON'T draw a spurious wide band.
        k_cross = None
        k_opt = int(k[np.nanargmax(diff)])
        band_lo = band_hi = k_opt
    return k_opt, band_lo, band_hi, k_cross


def plot_similarity(k_vals, within_m, within_s, between_m, between_s,
                    k_opt, band_lo, band_hi, xlabel, out_png, opt_word="optimal",
                    between_label="Between clusters (90th pct of pair corr)"):
    """Render the reference '(A) Similarity' panel: within (blue) and between
    (red) pattern-correlation curves with spread bands, the grey intersection
    band, and the optimal cluster count marked."""
    k_vals = list(k_vals)
    wm = np.asarray(within_m, float); ws = np.asarray(within_s, float)
    bm = np.asarray(between_m, float); bs = np.asarray(between_s, float)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    BLUE, RED = "#1f5fd0", "#e8202a"

    if band_hi >= band_lo:
        ax.axvspan(band_lo - 0.5, band_hi + 0.5, color="0.85", zorder=0,
                   label="intersection (optimal)")
    ax.fill_between(k_vals, wm - ws, wm + ws, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(k_vals, bm - bs, bm + bs, color=RED, alpha=0.18, lw=0)
    ax.plot(k_vals, wm, "-", color=BLUE, lw=2.5, label="Within clusters")
    ax.plot(k_vals, bm, "-", color=RED, lw=2.5, label=between_label)
    ax.axvline(k_opt, color="0.35", lw=1.0, ls="--", zorder=1)
    ymin = float(np.nanmin([np.nanmin(wm), np.nanmin(bm)]))
    ax.annotate(f"{opt_word} = {k_opt}", xy=(k_opt, ymin),
                xytext=(k_opt + 0.2, ymin), fontsize=9, color="0.2")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Pattern correlation", fontsize=12)
    ax.set_xticks(k_vals)
    ax.set_title("(A) Similarity", fontsize=14, weight="bold", loc="left")
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--conus", action="store_true", default=True,
                   help="Use the CONUS box (default; matches cluster_analysis_sliding7d_conus)")
    p.add_argument("--global", dest="globaldomain", action="store_true",
                   help="Use the full global domain instead of CONUS")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=20)
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

    # Feature matrix (standardized 37-yr time series per valid grid point)
    _, features_temporal, valid_mask = build_features(pred, lat, lon)
    n_valid = int(valid_mask.sum())
    print(f"features_temporal: {features_temporal.shape}  n_valid={n_valid}", flush=True)

    # cos-lat area weights aligned to valid grid points
    lat_g = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], valid_mask.shape)
    weights = lat_g[valid_mask].astype(np.float64)

    # Full Ward dendrogram with queen spatial connectivity (REDCAP backbone)
    print("Fitting full Ward dendrogram (queen-connectivity) ...", flush=True)
    conn = build_queen_connectivity(valid_mask)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                  linkage="ward", connectivity=conn,
                                  compute_full_tree=True)
    agg.fit(features_temporal)
    linkage_mat = _sklearn_to_scipy_linkage(agg.children_, agg.distances_, n_valid)

    # Sweep K = kmin..kmax, cutting the SAME tree for an exact cluster count
    k_vals = list(range(args.kmin, args.kmax + 1))
    within_m, within_s, between_m, between_s = [], [], [], []
    print("\n  K   within   between", flush=True)
    for k in k_vals:
        labels = fcluster(linkage_mat, t=k, criterion="maxclust") - 1
        k_eff = int(labels.max()) + 1
        wm, ws, bm, bs = cluster_similarity(features_temporal, labels, weights, k_eff)
        within_m.append(wm); within_s.append(ws)
        between_m.append(bm); between_s.append(bs)
        print(f"  {k:2d}   {wm:.3f}    {bm if np.isfinite(bm) else float('nan'):.3f}", flush=True)

    within_m = np.array(within_m); within_s = np.array(within_s)
    between_m = np.array(between_m); between_s = np.array(between_s)

    # optimal = red∩blue intersection (between rises to meet within)
    k_opt, band_lo, band_hi, k_cross = select_optimal_intersection(
        k_vals, within_m, within_s, between_m, between_s)
    print(f"\nCrossover K (between p90 >= within): {k_cross}", flush=True)
    print(f"Optimal REDCAP K (red∩blue intersection): {k_opt}  "
          f"band {band_lo}-{band_hi}", flush=True)

    out_png = out_dir / "optimal_clusters_redcap.png"
    plot_similarity(k_vals, within_m, within_s, between_m, between_s,
                    k_opt, band_lo, band_hi,
                    "Number of clusters (REDCAP, N×1)", out_png, opt_word="optimal K")
    print(f"wrote {out_png}", flush=True)

    summary = {
        "method": "REDCAP (Ward, queen-connectivity); within = member↔centroid corr, "
                  "between = 90th-pct cluster-pair corr",
        "domain": "conus" if conus else "global",
        "subdomain": args.domain,
        "n_valid": n_valid,
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
    out_json = out_dir / "optimal_clusters_redcap.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
