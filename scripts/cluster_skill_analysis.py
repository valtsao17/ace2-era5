#!/usr/bin/env python3
"""
Adaptive clustering evaluation of ACE2 JJA heat-extreme risk predictions.

Three methods:
  1. K-means with spatial proximity penalty    – sweep n_clusters
  2. Self-Organizing Map (SOM)                 – sweep grid size
  3. REDCAP (Ward full-order, spatially constrained) – sweep n_clusters
     Implemented via sklearn AgglomerativeClustering(linkage='ward',
     connectivity=queen_adjacency), which is equivalent to Guo 2008
     REDCAP full-order Ward: at every merge step only adjacent clusters are
     candidates, Ward criterion selects the pair minimising within-cluster
     variance increase.

Cluster assignment: unsupervised, based solely on predicted risk time-series
similarity.  No ground truth used in clustering.

Evaluation: aggregate both pred_prob and obs_ext within each cluster → compute
Kendall tau and BSS across 111 events (37 years × 3 targets).

"Goldilocks" curve: domain-mean tau vs n_clusters, expected to peak at the
resolution where the model's risk predictions are most reliable.

Outputs → outputs/lag_may/cluster_analysis/
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
from scipy.stats import kendalltau
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from tqdm.auto import tqdm
from minisom import MiniSom

import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

METRICS_DIR = PROJECT_ROOT / "outputs/lag_may/postprocess_jja/metrics"
THRESH_DIR  = PROJECT_ROOT / "outputs/lag_may/postprocess_jja/thresholds"
CACHE_DIR   = PROJECT_ROOT / "outputs/lag_may/postprocess_jja/era5_cache"
OUT_DIR     = PROJECT_ROOT / "outputs/lag_may/cluster_analysis"

YEARS   = list(range(1980, 2017))
TARGETS = [("Jun1", 6, 1), ("Jul1", 7, 1), ("Aug1", 8, 1)]

KMEANS_SIZES = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
SOM_SIZES    = [(2, 2), (3, 3), (5, 5), (7, 7), (10, 10), (13, 13), (15, 15), (20, 20)]

# REDCAP sweep sizes — fewer than K-means because Ward linkage is O(n²) per step
REDCAP_SIZES      = [5, 10, 25, 50, 100, 200, 400, 800]
REDCAP_SIZES_CONUS = [2, 3, 5, 10, 20, 50, 100, 200]

# North-America region sweep sizes (~6090 grid points)
KMEANS_SIZES_CONUS = [2, 3, 5, 10, 20, 50, 100, 200, 400]
SOM_SIZES_CONUS    = [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (10, 10), (13, 13)]

# North-America bounding box: lat 15–72°N, lon 200–305°E (160°W–55°W)
# Includes Alaska, CONUS, and northern Mexico; avoids dateline crossing.
CONUS_LAT_SLICE = slice(105, 163)
CONUS_LON_SLICE = slice(200, 305)

SPATIAL_ALPHA = 1.0   # relative weight of spatial features vs temporal

# Lazy-loaded land/state geometries for plain-matplotlib regional maps
_LAND_GEOMS  = None
_STATE_GEOMS = None

_PLATE = ccrs.PlateCarree()
_HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat_skill", ["white", "#FFE066", "#FF8C00", "#CC0000", "#67000d"], N=256)
_TAU_CMAP = "RdBu_r"   # diverging blue-white-red, signed tau (negative=blue, positive=red)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        10,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


# ── data loading ─────────────────────────────────────────────────────────────

def load_pred_prob_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pred_prob for all targets, stack to (111, lat, lon)."""
    arrays = []
    for label, tm, td in TARGETS:
        ds = xr.open_dataset(METRICS_DIR / f"pred_prob_{label.lower()}.nc")
        arrays.append(ds["pred_prob"].values.astype(np.float32))
    ds0 = xr.open_dataset(METRICS_DIR / "pred_prob_jun1.nc")
    lat = ds0["lat"].values
    lon = ds0["lon"].values
    return np.concatenate(arrays, axis=0), lat, lon


def reconstruct_obs_ext(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Reconstruct obs_ext (111, lat, lon) from cached ERA5 data + LOO thresholds."""
    obs_list = []
    for label, tm, td in TARGETS:
        ds_thresh = xr.open_dataset(THRESH_DIR / f"era5_loo_p90_{tm:02d}{td:02d}.nc")
        era5_thresh = ds_thresh["threshold"].values.astype(np.float32)  # (37, lat, lon)
        n_lat, n_lon = len(lat), len(lon)
        obs_yr = np.full((len(YEARS), n_lat, n_lon), np.nan, dtype=np.float32)

        for y_idx, year in enumerate(tqdm(YEARS, desc=f"obs_ext {label}", leave=False)):
            cache = CACHE_DIR / f"era5_daily_tmax_C_y{year:04d}_m{tm:02d}_on_grid.nc"
            with xr.open_dataset(cache) as ds_m:
                times = pd.DatetimeIndex(ds_m["era5_daily_tmax_C"].time.values)
                target = pd.Timestamp(year=year, month=tm, day=td)
                idx = int(np.argmin(np.abs(times - target)))
                era5_val = ds_m["era5_daily_tmax_C"].isel(time=idx).values.astype(np.float32)
            obs_yr[y_idx] = (era5_val > era5_thresh[y_idx]).astype(np.float32)

        obs_list.append(obs_yr)
        print(f"  obs_ext {label} done", flush=True)

    return np.concatenate(obs_list, axis=0)  # (111, lat, lon)


# ── feature matrix ────────────────────────────────────────────────────────────

def build_features(pred_prob: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   spatial_alpha: float = SPATIAL_ALPHA
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrices for clustering.

    Returns:
      features_full : (n_valid, n_events + 4) — temporal z-score + spatial (for k-means)
      features_temporal : (n_valid, n_events)  — temporal only (for SOM)
      valid_mask : (lat, lon) bool
    """
    n_events, n_lat, n_lon = pred_prob.shape
    valid_mask = np.all(np.isfinite(pred_prob), axis=0)

    temporal = pred_prob[:, valid_mask].T  # (n_valid, n_events)
    mean_ = temporal.mean(axis=1, keepdims=True)
    std_  = temporal.std(axis=1, keepdims=True)
    std_  = np.where(std_ == 0, 1.0, std_)
    temporal_z = (temporal - mean_) / std_

    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    lat_r  = np.deg2rad(lat_g[valid_mask])
    lon_r  = np.deg2rad(lon_g[valid_mask])
    spatial = np.stack([np.sin(lat_r), np.cos(lat_r),
                        np.sin(lon_r), np.cos(lon_r)], axis=1)
    # Scale so the 4 spatial features collectively contribute ~alpha times
    # as much to Euclidean distance as the n_events temporal features do.
    scale = spatial_alpha * np.sqrt(n_events / 4.0)
    spatial_scaled = (spatial * scale).astype(np.float32)

    features_full     = np.concatenate([temporal_z, spatial_scaled], axis=1).astype(np.float32)
    features_temporal = temporal_z.astype(np.float32)
    return features_full, features_temporal, valid_mask


# ── cluster-level evaluation ──────────────────────────────────────────────────

def eval_cluster_skill(pred_prob: np.ndarray, obs_ext: np.ndarray,
                       labels: np.ndarray, valid_mask: np.ndarray,
                       lat: np.ndarray, n_clusters: int):
    """
    Evaluate Kendall tau and BSS at cluster level.

    For each cluster: cos-lat weighted aggregate of pred_prob and obs_ext
    over its member grid points → τ and BSS across n_events.

    Returns:
      tau_map (lat, lon) : each point gets its cluster's tau
      bss_map (lat, lon) : each point gets its cluster's BSS
      tau_cl  (n_clusters,) : per-cluster tau
      bss_cl  (n_clusters,) : per-cluster BSS
      sizes   (n_clusters,) : n_valid gridpoints per cluster
    """
    n_events, n_lat, n_lon = pred_prob.shape

    cos_w = np.cos(np.deg2rad(lat))[:, np.newaxis]  # (lat, 1)
    cos_w_flat = np.broadcast_to(cos_w, (n_lat, n_lon))[valid_mask]  # (n_valid,)

    pred_v = pred_prob[:, valid_mask].T  # (n_valid, n_events)
    obs_v  = obs_ext[:, valid_mask].T   # (n_valid, n_events)

    tau_cl  = np.full(n_clusters, np.nan, dtype=np.float32)
    bss_cl  = np.full(n_clusters, np.nan, dtype=np.float32)
    sizes   = np.zeros(n_clusters, dtype=int)
    tau_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    bss_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    valid_idx = np.where(valid_mask.ravel())[0]  # indices of valid points in flat grid

    for c in range(n_clusters):
        idx = labels == c
        if idx.sum() == 0:
            continue
        sizes[c] = idx.sum()

        w  = cos_w_flat[idx]
        w  = w / w.sum()

        pred_c = (pred_v[idx] * w[:, np.newaxis]).sum(axis=0)  # (n_events,)
        obs_c  = (obs_v[idx]  * w[:, np.newaxis]).sum(axis=0)

        ok = np.isfinite(pred_c) & np.isfinite(obs_c)
        if ok.sum() < 10 or obs_c[ok].std() < 1e-9:
            continue

        tau, _ = kendalltau(pred_c[ok], obs_c[ok])
        tau_cl[c] = tau

        bs     = float(np.mean((pred_c - obs_c) ** 2))
        clim   = float(obs_c.mean())
        bs_clim = float(np.mean((clim - obs_c) ** 2))
        if bs_clim > 0:
            bss_cl[c] = 1.0 - bs / bs_clim

        # fill spatial maps
        flat_idx = valid_idx[idx]
        row, col = np.unravel_index(flat_idx, (n_lat, n_lon))
        tau_map[row, col] = tau
        if bs_clim > 0:
            bss_map[row, col] = 1.0 - bs / bs_clim

    return tau_map, bss_map, tau_cl, bss_cl, sizes


def domain_mean_tau(tau_map: np.ndarray, lat: np.ndarray) -> float:
    cos_w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(tau_map)
    num = float(np.nansum(tau_map * cos_w * valid))
    den = float(np.nansum(cos_w * valid))
    return num / den if den > 0 else float("nan")


def domain_mean_bss(bss_map: np.ndarray, lat: np.ndarray) -> float:
    return domain_mean_tau(bss_map, lat)  # same formula


# ── sweeps ────────────────────────────────────────────────────────────────────

def run_kmeans_sweep(features_full, pred_prob, obs_ext, valid_mask, lat):
    print("\n=== K-means sweep ===", flush=True)
    results = []
    for k in KMEANS_SIZES:
        print(f"  k={k}", flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                             batch_size=min(4096, features_full.shape[0]))
        labels = km.fit_predict(features_full)
        tau_map, bss_map, tau_cl, bss_cl, sizes = eval_cluster_skill(
            pred_prob, obs_ext, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_bss(bss_map, lat)
        print(f"    τ_domain={tau_d:.4f}  BSS_domain={bss_d:.4f}", flush=True)
        results.append(dict(k=k, tau_domain=tau_d, bss_domain=bss_d,
                            tau_map=tau_map, bss_map=bss_map,
                            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sizes, labels=labels))
    return results


def build_queen_connectivity(valid_mask: np.ndarray) -> sp.csr_matrix:
    """Queen adjacency sparse matrix for valid grid points on a regular lat/lon grid.

    Two valid points are queen-adjacent if they share an edge or corner in the
    (lat, lon) index space (8-connectivity).  Returns an (n_valid, n_valid)
    CSR matrix with 1s where points are neighbours.
    """
    n_lat, n_lon = valid_mask.shape
    flat_valid = valid_mask.ravel()
    n_valid = flat_valid.sum()

    # map flat grid index → compact index (−1 if not valid)
    compact = np.full(n_lat * n_lon, -1, dtype=np.int32)
    compact[flat_valid] = np.arange(n_valid, dtype=np.int32)

    rows, cols = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            lat_idx, lon_idx = np.where(valid_mask)
            ni = lat_idx + di
            nj = lon_idx + dj
            in_bounds = (ni >= 0) & (ni < n_lat) & (nj >= 0) & (nj < n_lon)
            ni, nj, li, lj = ni[in_bounds], nj[in_bounds], lat_idx[in_bounds], lon_idx[in_bounds]
            nbr_flat   = ni * n_lon + nj
            src_flat   = li * n_lon + lj
            nbr_valid  = flat_valid[nbr_flat]
            r = compact[src_flat[nbr_valid]]
            c = compact[nbr_flat[nbr_valid]]
            rows.append(r)
            cols.append(c)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    conn = sp.csr_matrix((data, (rows, cols)), shape=(n_valid, n_valid))
    return conn


def _sklearn_to_scipy_linkage(children: np.ndarray, distances: np.ndarray,
                               n_samples: int) -> np.ndarray:
    """Convert sklearn AgglomerativeClustering output to scipy linkage matrix.

    scipy format: (n_merges, 4) = [left, right, distance, cluster_size]
    sklearn stores node ids ≥ n_samples for merged clusters.
    """
    n_merges = n_samples - 1
    counts = np.zeros(n_merges, dtype=np.float64)
    leaf_counts = np.ones(n_samples, dtype=np.float64)

    for i, (left, right) in enumerate(children):
        lc = leaf_counts[left]  if left  < n_samples else counts[left  - n_samples]
        rc = leaf_counts[right] if right < n_samples else counts[right - n_samples]
        counts[i] = lc + rc

    return np.column_stack([children.astype(np.float64),
                             distances.astype(np.float64),
                             counts])


def run_redcap_sweep(features_temporal, pred_prob, obs_ext, valid_mask, lat,
                     n_thresholds: int = 40):
    """REDCAP Ward full-order, distance-cutoff sweep.

    Builds the full spatially-constrained Ward dendrogram ONCE, then slices it
    at `n_thresholds` evenly-spaced distance cutoffs.  K emerges from the
    cutoff — it is not fixed in advance.

    The distance metric is Euclidean on the z-scored temporal feature matrix
    (n_valid × n_events), so it captures similarity in predicted-risk patterns.
    """
    from scipy.cluster.hierarchy import fcluster

    print("\n=== REDCAP (Ward, spatially constrained) — distance-cutoff sweep ===", flush=True)
    print("  Building queen connectivity ...", flush=True)
    conn = build_queen_connectivity(valid_mask)
    n_valid = conn.shape[0]
    print(f"  n_valid={n_valid}  nnz={conn.nnz}", flush=True)

    print("  Fitting full Ward dendrogram (this runs once) ...", flush=True)
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,      # build complete tree
        linkage="ward",
        connectivity=conn,
        compute_full_tree=True,
    )
    agg.fit(features_temporal)

    linkage_mat = _sklearn_to_scipy_linkage(agg.children_, agg.distances_, n_valid)
    merge_dists = agg.distances_  # (n_valid-1,) ascending merge distances

    # Sweep distance thresholds: percentiles 1..99 of merge distances, deduplicated
    pct_vals = np.percentile(merge_dists, np.linspace(1, 99, n_thresholds))
    thresholds = np.unique(np.round(pct_vals, 6))
    print(f"  Sweeping {len(thresholds)} distance cutoffs "
          f"[{thresholds[0]:.3f} … {thresholds[-1]:.3f}]", flush=True)

    results = []
    for d in thresholds:
        raw_labels = fcluster(linkage_mat, t=d, criterion="distance")
        labels = raw_labels - 1  # 0-indexed
        k = int(labels.max()) + 1
        tau_map, bss_map, tau_cl, bss_cl, sz = eval_cluster_skill(
            pred_prob, obs_ext, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_bss(bss_map, lat)
        print(f"    cutoff={d:.3f}  K={k:4d}  τ={tau_d:.4f}  BSS={bss_d:.4f}", flush=True)
        results.append(dict(
            cutoff=float(d), k=k,
            tau_domain=tau_d, bss_domain=bss_d,
            tau_map=tau_map, bss_map=bss_map,
            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sz, labels=labels,
        ))

    return results, linkage_mat, merge_dists


def run_som_sweep(features_temporal, pred_prob, obs_ext, valid_mask, lat):
    print("\n=== SOM sweep ===", flush=True)
    n_features = features_temporal.shape[1]
    results = []
    for (m, n) in SOM_SIZES:
        k = m * n
        print(f"  SOM {m}×{n} (k={k})", flush=True)
        som = MiniSom(m, n, n_features, sigma=max(1.0, min(m, n) / 2.0),
                      learning_rate=0.5, random_seed=42)
        som.train_random(features_temporal, num_iteration=2000, verbose=False)
        bmu = np.array([som.winner(x) for x in tqdm(features_temporal,
                                                     desc=f"BMU {m}×{n}", leave=False)])
        labels = bmu[:, 0] * n + bmu[:, 1]
        tau_map, bss_map, tau_cl, bss_cl, sizes = eval_cluster_skill(
            pred_prob, obs_ext, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_bss(bss_map, lat)
        print(f"    τ_domain={tau_d:.4f}  BSS_domain={bss_d:.4f}", flush=True)
        results.append(dict(k=k, m=m, n=n, tau_domain=tau_d, bss_domain=bss_d,
                            tau_map=tau_map, bss_map=bss_map,
                            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sizes, labels=labels))
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_redcap_goldilocks(redcap_results, merge_dists, tau_gridpt: float):
    """REDCAP-specific goldilocks: τ vs distance cutoff with K on secondary x-axis."""
    cutoffs = np.array([r["cutoff"] for r in redcap_results])
    taus    = np.array([r["tau_domain"] for r in redcap_results])
    ks      = np.array([r["k"] for r in redcap_results])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cutoffs, taus, "^-", color="#2ca02c", linewidth=1.8, markersize=5,
             label="REDCAP Ward τ (domain mean)")
    ax1.axhline(tau_gridpt, color="0.4", linewidth=1.2, linestyle=":",
                label=f"Grid-point reference (τ={tau_gridpt:.3f})")
    ax1.set_xlabel("Ward distance cutoff (within-cluster variance scale)", fontsize=10)
    ax1.set_ylabel("Domain-mean Kendall τ", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate selected K values along the curve
    prev_k = -1
    for cut, tau, k in zip(cutoffs, taus, ks):
        if k != prev_k and k in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500}:
            ax1.annotate(f"K={k}", xy=(cut, tau),
                         xytext=(0, 7), textcoords="offset points",
                         fontsize=7, ha="center", color="#2ca02c")
            prev_k = k

    # Secondary x-axis: dendrogram height distribution
    ax2 = ax1.twinx()
    ax2.hist(merge_dists, bins=80, color="#2ca02c", alpha=0.15, density=True)
    ax2.set_ylabel("Merge-distance density", fontsize=9, color="#2ca02c", alpha=0.6)
    ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)

    ax1.set_title("REDCAP Ward: Skill vs Distance Cutoff\n"
                  "ACE2 JJA Heat Extremes, CONUS — K emerges from cutoff", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "redcap_goldilocks_distance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def plot_goldilocks(km_results, som_results, redcap_results,
                    tau_gridpt: float, bss_gridpt: float):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    km_k   = [r["k"] for r in km_results]
    km_tau = [r["tau_domain"] for r in km_results]
    km_bss = [r["bss_domain"] for r in km_results]
    som_k   = [r["k"] for r in som_results]
    som_tau = [r["tau_domain"] for r in som_results]
    som_bss = [r["bss_domain"] for r in som_results]
    # For the shared plot use K (derived from cutoff) as x-axis for comparability
    rc_k   = [r["k"] for r in redcap_results]
    rc_tau = [r["tau_domain"] for r in redcap_results]
    rc_bss = [r["bss_domain"] for r in redcap_results]

    ax = axes[0]
    ax.plot(km_k,  km_tau,  "o-",  color="#1f77b4", label="K-means (spatial penalty)")
    ax.plot(som_k, som_tau, "s--", color="#ff7f0e", label="SOM (temporal only)")
    ax.plot(rc_k,  rc_tau,  "^-",  color="#2ca02c", label="REDCAP Ward (spatially constrained)")
    ax.axhline(tau_gridpt, color="0.4", linewidth=1.2, linestyle=":",
               label=f"Grid-point level (τ={tau_gridpt:.3f})")
    ax.set_xscale("log")
    ax.set_ylabel("Domain-mean Kendall τ (cluster level)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title("Goldilocks Resolution: Cluster-level Skill vs Number of Clusters\n"
                 "ACE2 JJA Heat Extremes, 1980–2016, 111 events per cluster", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(km_k,  km_bss,  "o-",  color="#1f77b4", label="K-means")
    ax.plot(som_k, som_bss, "s--", color="#ff7f0e", label="SOM")
    ax.plot(rc_k,  rc_bss,  "^-",  color="#2ca02c", label="REDCAP Ward")
    ax.axhline(bss_gridpt, color="0.4", linewidth=1.2, linestyle=":",
               label=f"Grid-point level (BSS={bss_gridpt:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of clusters  (log scale)", fontsize=10)
    ax.set_ylabel("Domain-mean BSS (cluster level)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "goldilocks_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def _map_ax(fig, pos, data, lat, lon, title, vmin, vmax, cmap, cbar_label):
    ax = fig.add_subplot(pos, projection=_PLATE)
    m = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, transform=_PLATE)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4, zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="gray", alpha=0.4)
    ax.set_title(title, fontsize=9)
    plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label=cbar_label)
    return ax



def plot_cluster_map(labels_full, lat, lon, n_clusters: int,
                     title: str, out_path: Path):
    """Map of cluster assignments (each cluster = distinct color)."""
    n_lat, n_lon = len(lat), len(lon)
    cmap = cm.get_cmap("tab20b", n_clusters) if n_clusters <= 20 else cm.get_cmap("turbo", n_clusters)

    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw=dict(projection=_PLATE))
    m = ax.pcolormesh(lon, lat, labels_full.reshape(n_lat, n_lon),
                      shading="auto", cmap=cmap, vmin=0, vmax=n_clusters - 1,
                      transform=_PLATE)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4, zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="gray", alpha=0.4)
    ax.set_title(title, fontsize=10)
    plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def _expand_labels(labels: np.ndarray, valid_mask: np.ndarray,
                   n_lat: int, n_lon: int) -> np.ndarray:
    full = np.full(n_lat * n_lon, -1, dtype=int)
    full[valid_mask.ravel()] = labels
    return full.reshape(n_lat, n_lon)


def plot_cluster_size_hist(sizes, n_clusters: int, title: str, out_path: Path):
    nonzero = sizes[sizes > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nonzero, bins=30, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cluster size (n valid grid points)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\nMedian={int(np.median(nonzero))}, Mean={int(nonzero.mean())}, "
                 f"Max={nonzero.max()}, Min={nonzero.min()}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def plot_tau_vs_size(tau_cl, sizes, title: str, out_path: Path):
    ok = (sizes > 0) & np.isfinite(tau_cl)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sizes[ok], tau_cl[ok], c=tau_cl[ok], cmap=_TAU_CMAP,
                    alpha=0.6, s=20, vmin=-1.0, vmax=1.0)
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cluster size (n valid grid points)")
    ax.set_ylabel("Cluster Kendall τ")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="τ")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def _get_border_geoms():
    """Lazy-load 50m land polygons + state/province borders from Natural Earth."""
    global _LAND_GEOMS, _STATE_GEOMS
    if _LAND_GEOMS is None:
        shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
        _LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution='50m', category='cultural',
                                       name='admin_1_states_provinces_lakes')
        _STATE_GEOMS = list(shpreader.Reader(shp).geometries())
    return _LAND_GEOMS, _STATE_GEOMS


def _draw_borders(ax, xlim, ylim, land_geoms, state_geoms):
    """Draw land outlines and state borders clipped to xlim/ylim on a plain axes."""
    for g in land_geoms:
        parts = [g] if hasattr(g, 'exterior') else list(g.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            if float(min(xs)) < xlim[1] and float(max(xs)) > xlim[0]:
                ax.plot(xs, ys, 'k-', linewidth=0.5, zorder=3)
            for interior in part.interiors:
                xs, ys = interior.xy
                ax.plot(xs, ys, 'k-', linewidth=0.3, zorder=3)
    for g in state_geoms:
        parts = [g] if hasattr(g, 'exterior') else list(g.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            if (float(min(xs)) < xlim[1] and float(max(xs)) > xlim[0] and
                    float(min(ys)) < ylim[1] and float(max(ys)) > ylim[0]):
                ax.plot(xs, ys, color='0.5', linewidth=0.25, zorder=2)


def _plain_map_axes(ax, lon_360, lat, pad=3.0):
    """Set up a plain matplotlib axes for a regional map; return (xlim, ylim).
    lon_360: 1-D array in 0–360 encoding — converted to −180..180 internally.
    """
    lon = lon_360 - 360.0 if float(lon_360.mean()) > 180 else lon_360
    xlim = [float(lon[0]) - 0.5 - pad, float(lon[-1]) + 0.5 + pad]
    ylim = [float(lat[0]) - 0.5 - pad, float(lat[-1]) + 0.5 + pad]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Latitude (°N)')
    ax.set_xlabel('Longitude')
    xt = np.arange(int(np.ceil(xlim[0] / 10) * 10), int(xlim[1]) + 1, 10, dtype=float)
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{int(abs(x))}°W' if x < 0 else f'{int(x)}°E' for x in xt])
    ax.grid(True, linewidth=0.2, color='gray', alpha=0.4)
    _draw_borders(ax, xlim, ylim, *_get_border_geoms())
    return lon, xlim, ylim


def _plain_map_figure(data, lat, lon_360, title, vmin, vmax, cmap, cbar_label, out: Path):
    """Regional scalar map on a plain matplotlib axes (no cartopy projection)."""
    lon = lon_360 - 360.0 if float(lon_360.mean()) > 180 else lon_360
    e = [float(lon[0]) - 0.5, float(lon[-1]) + 0.5,
         float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(data, origin='lower', extent=e, aspect='auto',
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)
    _plain_map_axes(ax, lon_360, lat)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, out)


def _save_figure(fig, out: Path):
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"wrote {out}", flush=True)


def _single_map_figure(data, lat, lon, title, vmin, vmax, cmap, cbar_label, out: Path):
    """One map panel — dispatches to plain-matplotlib for regional, cartopy for global."""
    if len(lat) < 170:
        _plain_map_figure(data, lat, lon, title, vmin, vmax, cmap, cbar_label, out)
        return
    # Global path: cartopy GeoAxes with pcolormesh (0–360 lon, set_global — stable)
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111, projection=_PLATE)
    m = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, transform=_PLATE)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4, zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.2, color="gray", alpha=0.4)
    ax.set_title(title, fontsize=9)
    plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, out)


def _cluster_map_figure(labels_2d, lat, lon, n_clusters: int, title: str, out: Path):
    """Cluster assignment map — plain matplotlib for regional, cartopy for global."""
    from matplotlib import colormaps
    cmap = colormaps["turbo"].resampled(n_clusters)
    if len(lat) < 170:
        lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
        e = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
             float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(labels_2d.astype(float), origin='lower', extent=e, aspect='auto',
                       cmap=cmap, vmin=0, vmax=n_clusters - 1, zorder=1)
        _plain_map_axes(ax, lon, lat)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label='Cluster ID')
        fig.tight_layout()
        _save_figure(fig, out)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6), subplot_kw=dict(projection=_PLATE))
        m = ax.pcolormesh(lon, lat, labels_2d.astype(float), shading='auto',
                          cmap=cmap, vmin=0, vmax=n_clusters - 1, transform=_PLATE)
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.4, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.4)
        ax.set_title(title, fontsize=10)
        plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label='Cluster ID')
        fig.tight_layout()
        _save_figure(fig, out)


def _three_panel_map_with_coords(data_list, titles, suptitle, lat, lon,
                                  out: Path, vmin, vmax, cmap, cbar_label):
    """Produce 3 separate single-panel map files (row1/row2/row3 suffix).

    Cartopy segfaults when 3 GeoAxes are created in one figure, so each panel
    gets its own figure, save, and gc.collect() cycle.
    """
    stem, suffix, parent = out.stem, out.suffix, out.parent
    for i, (data, title) in enumerate(zip(data_list, titles), 1):
        _single_map_figure(data, lat, lon, title, vmin, vmax, cmap, cbar_label,
                           parent / f"{stem}_row{i}{suffix}")


def plot_full_comparison(km_results, som_results, redcap_results,
                         tau_gridpt_map, bss_gridpt_map, lat, lon, valid_mask):
    """Produce all comparison figures as separate files (memory-safe)."""
    km_opt  = max(km_results,  key=lambda r: r["tau_domain"])
    som_opt = max(som_results, key=lambda r: r["tau_domain"])
    rc_opt  = max(redcap_results, key=lambda r: r["tau_domain"]) if redcap_results else None
    n_lat, n_lon = len(lat), len(lon)

    # --- tau comparison ---
    print("  tau comparison maps ...", flush=True)
    tau_panels  = [tau_gridpt_map, km_opt["tau_map"], som_opt["tau_map"]]
    tau_titles  = [
        "Grid-point τ (reference)",
        f"K-means τ  k={km_opt['k']}  (domain τ={km_opt['tau_domain']:.3f})",
        f"SOM τ  {som_opt['m']}×{som_opt['n']}  (domain τ={som_opt['tau_domain']:.3f})",
    ]
    if rc_opt:
        tau_panels.append(rc_opt["tau_map"])
        tau_titles.append(f"REDCAP Ward τ  k={rc_opt['k']}  (domain τ={rc_opt['tau_domain']:.3f})")
    _three_panel_map_with_coords(
        tau_panels[:3], tau_titles[:3],
        "Kendall τ: Grid-point vs Cluster-level — ACE2 JJA 1980–2016",
        lat, lon, OUT_DIR / "tau_comparison_maps.png",
        vmin=-1.0, vmax=1.0, cmap=_TAU_CMAP, cbar_label="τ",
    )
    if rc_opt:
        _single_map_figure(rc_opt["tau_map"], lat, lon,
                           tau_titles[-1], -1.0, 1.0, _TAU_CMAP, "τ",
                           OUT_DIR / f"tau_redcap_k{rc_opt['k']}.png")

    # --- BSS comparison ---
    print("  BSS comparison maps ...", flush=True)
    bss_panels = [np.abs(bss_gridpt_map), np.abs(km_opt["bss_map"]), np.abs(som_opt["bss_map"])]
    bss_titles = [
        "Grid-point |BSS| (reference)",
        f"K-means |BSS|  k={km_opt['k']}  (domain BSS={km_opt['bss_domain']:.3f})",
        f"SOM |BSS|  {som_opt['m']}×{som_opt['n']}  (domain BSS={som_opt['bss_domain']:.3f})",
    ]
    _three_panel_map_with_coords(
        bss_panels, bss_titles,
        "|BSS|: Grid-point vs Cluster-level — ACE2 JJA 1980–2016",
        lat, lon, OUT_DIR / "bss_comparison_maps.png",
        vmin=0, vmax=0.5, cmap=_HEAT_CMAP, cbar_label="|BSS|",
    )

    # --- K-means cluster assignment ---
    print("  K-means cluster assignment map ...", flush=True)
    km_labels_full = _expand_labels(km_opt["labels"], valid_mask, n_lat, n_lon)
    _cluster_map_figure(km_labels_full, lat, lon, int(km_labels_full.max()) + 1,
                        f"K-means cluster assignment  k={km_opt['k']} (optimal τ)",
                        OUT_DIR / f"kmeans_cluster_map_k{km_opt['k']}.png")

    # --- SOM cluster assignment ---
    print("  SOM cluster assignment map ...", flush=True)
    som_labels_full = _expand_labels(som_opt["labels"], valid_mask, n_lat, n_lon)
    _cluster_map_figure(som_labels_full, lat, lon, int(som_labels_full.max()) + 1,
                        f"SOM cluster assignment  {som_opt['m']}×{som_opt['n']} (optimal τ)",
                        OUT_DIR / f"som_cluster_map_{som_opt['m']}x{som_opt['n']}.png")

    # --- REDCAP cluster assignment ---
    if rc_opt:
        print("  REDCAP cluster assignment map ...", flush=True)
        rc_labels_full = _expand_labels(rc_opt["labels"], valid_mask, n_lat, n_lon)
        _cluster_map_figure(rc_labels_full, lat, lon, int(rc_labels_full.max()) + 1,
                            f"REDCAP Ward cluster assignment  k={rc_opt['k']} (optimal τ)",
                            OUT_DIR / f"redcap_cluster_map_k{rc_opt['k']}.png")

    return km_opt, som_opt, rc_opt


# ── result caching ────────────────────────────────────────────────────────────

def _pack_results(arrays, results, prefix, extra_keys=()):
    arrays[f"n_{prefix}"] = np.array([len(results)])
    for i, r in enumerate(results):
        for key in ("tau_map", "bss_map", "tau_cl", "bss_cl", "sizes", "labels"):
            arrays[f"{prefix}_{i}_{key}"] = r[key]
        arrays[f"{prefix}_{i}_k"]          = np.array([r["k"]])
        arrays[f"{prefix}_{i}_tau_domain"] = np.array([r["tau_domain"]])
        arrays[f"{prefix}_{i}_bss_domain"] = np.array([r["bss_domain"]])
        for ek in extra_keys:
            arrays[f"{prefix}_{i}_{ek}"] = np.array([r[ek]])
        if "cutoff" in r:
            arrays[f"{prefix}_{i}_cutoff"] = np.array([r["cutoff"]])


def _unpack_results(d, prefix, extra_keys=()):
    results = []
    for i in range(int(d[f"n_{prefix}"][0])):
        entry = {
            "k":          int(d[f"{prefix}_{i}_k"][0]),
            "tau_domain": float(d[f"{prefix}_{i}_tau_domain"][0]),
            "bss_domain": float(d[f"{prefix}_{i}_bss_domain"][0]),
            "tau_map":    d[f"{prefix}_{i}_tau_map"],
            "bss_map":    d[f"{prefix}_{i}_bss_map"],
            "tau_cl":     d[f"{prefix}_{i}_tau_cl"],
            "bss_cl":     d[f"{prefix}_{i}_bss_cl"],
            "sizes":      d[f"{prefix}_{i}_sizes"],
            "labels":     d[f"{prefix}_{i}_labels"],
        }
        for ek in extra_keys:
            entry[ek] = int(d[f"{prefix}_{i}_{ek}"][0])
        results.append(entry)
    return results


def save_results(km_results, som_results, redcap_results,
                 tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon,
                 merge_dists=None):
    cache = OUT_DIR / "sweep_cache.npz"
    arrays = {
        "tau_gridpt_map": tau_gridpt_map,
        "bss_gridpt_map": bss_gridpt_map,
        "valid_mask": valid_mask,
        "lat": lat, "lon": lon,
    }
    if merge_dists is not None:
        arrays["rc_merge_dists"] = merge_dists
    _pack_results(arrays, km_results,     "km")
    _pack_results(arrays, som_results,    "som", extra_keys=("m", "n"))
    _pack_results(arrays, redcap_results, "rc")
    np.savez_compressed(cache, **arrays)
    print(f"Cached sweep results → {cache}", flush=True)


def load_results():
    cache = OUT_DIR / "sweep_cache.npz"
    d = np.load(cache, allow_pickle=False)
    lat, lon = d["lat"], d["lon"]
    tau_gridpt_map = d["tau_gridpt_map"]
    bss_gridpt_map = d["bss_gridpt_map"]
    valid_mask     = d["valid_mask"].astype(bool)
    km_results     = _unpack_results(d, "km")
    som_results    = _unpack_results(d, "som", extra_keys=("m", "n"))
    # backwards compat: old caches may not have redcap
    redcap_results = []
    if "n_rc" in d:
        redcap_results = _unpack_results(d, "rc")
        # restore cutoff field if present
        for i, r in enumerate(redcap_results):
            key = f"rc_{i}_cutoff"
            if key in d:
                r["cutoff"] = float(d[key][0])
    merge_dists = d["rc_merge_dists"] if "rc_merge_dists" in d else None
    return km_results, som_results, redcap_results, tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, merge_dists


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true",
                   help="Skip sweeps, load cached results, redo all plots only")
    p.add_argument("--conus", action="store_true",
                   help="Restrict analysis to CONUS bounding box (lat 20-55N, lon 230-300E)")
    p.add_argument("--skip-redcap", action="store_true",
                   help="Skip REDCAP sweep (useful for quick K-means/SOM runs)")
    return p.parse_args()


def main():
    args = parse_args()
    global OUT_DIR, KMEANS_SIZES, SOM_SIZES, REDCAP_SIZES

    if args.conus:
        OUT_DIR       = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_conus"
        KMEANS_SIZES  = KMEANS_SIZES_CONUS
        SOM_SIZES     = SOM_SIZES_CONUS
        REDCAP_SIZES  = REDCAP_SIZES_CONUS
        print("CONUS mode: lat 15–72°N, lon 200–305°E (160°W–55°W, includes Alaska + Mexico)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        print("--replot: loading cached sweep results ...", flush=True)
        km_results, som_results, redcap_results, tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, merge_dists = load_results()
        tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
        bss_gridpt_ref = domain_mean_bss(bss_gridpt_map, lat)
        print(f"  grid-point τ={tau_gridpt_ref:.4f}  BSS={bss_gridpt_ref:.4f}", flush=True)
    else:
        print("Loading pred_prob ...", flush=True)
        pred_prob, lat, lon = load_pred_prob_all()  # (111, 180, 360)
        print(f"  pred_prob: {pred_prob.shape}  NaN fraction: {np.isnan(pred_prob).mean():.3f}", flush=True)

        print("Reconstructing obs_ext from cached ERA5 ...", flush=True)
        obs_ext = reconstruct_obs_ext(lat, lon)  # (111, 180, 360)
        print(f"  obs_ext:   {obs_ext.shape}  mean extremes: {np.nanmean(obs_ext):.3f}", flush=True)

        print("Loading grid-point tau reference ...", flush=True)
        ds_tau = xr.open_dataset(METRICS_DIR / "tau_map_jja.nc")
        tau_gridpt_map = ds_tau["kendall_tau"].values  # (lat, lon)
        tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
        print(f"  Grid-point domain-mean τ = {tau_gridpt_ref:.4f}", flush=True)

        bs      = np.nanmean((pred_prob - obs_ext) ** 2, axis=0)
        clim    = np.nanmean(obs_ext, axis=0, keepdims=True)
        bs_clim = np.nanmean((clim - obs_ext) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bss_gridpt_map = np.where(bs_clim > 0, 1.0 - bs / bs_clim, np.nan).astype(np.float32)
        bss_gridpt_ref = domain_mean_bss(bss_gridpt_map, lat)
        print(f"  Grid-point domain-mean BSS = {bss_gridpt_ref:.4f}", flush=True)

        if args.conus:
            print(f"  Subsetting to CONUS ...", flush=True)
            pred_prob      = pred_prob[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
            obs_ext        = obs_ext[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
            tau_gridpt_map = tau_gridpt_map[CONUS_LAT_SLICE, CONUS_LON_SLICE]
            bss_gridpt_map = bss_gridpt_map[CONUS_LAT_SLICE, CONUS_LON_SLICE]
            lat            = lat[CONUS_LAT_SLICE]
            lon            = lon[CONUS_LON_SLICE]
            tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
            bss_gridpt_ref = domain_mean_bss(bss_gridpt_map, lat)
            print(f"  CONUS subset: pred_prob {pred_prob.shape}  "
                  f"lat {lat[0]:.1f}–{lat[-1]:.1f}  lon {lon[0]:.1f}–{lon[-1]:.1f}", flush=True)
            print(f"  CONUS grid-point τ={tau_gridpt_ref:.4f}  BSS={bss_gridpt_ref:.4f}", flush=True)

        print("Building feature matrices ...", flush=True)
        features_full, features_temporal, valid_mask = build_features(pred_prob, lat, lon)
        print(f"  features_full: {features_full.shape}  valid_mask: {valid_mask.sum()}/{valid_mask.size}", flush=True)

        km_results  = run_kmeans_sweep(features_full, pred_prob, obs_ext, valid_mask, lat)
        som_results = run_som_sweep(features_temporal, pred_prob, obs_ext, valid_mask, lat)

        if args.skip_redcap:
            redcap_results, merge_dists = [], None
        else:
            redcap_results, linkage_mat, merge_dists = run_redcap_sweep(
                features_temporal, pred_prob, obs_ext, valid_mask, lat)

        save_results(km_results, som_results, redcap_results,
                     tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon,
                     merge_dists=merge_dists)

    print("\nPlotting Goldilocks curve ...", flush=True)
    plot_goldilocks(km_results, som_results, redcap_results, tau_gridpt_ref, bss_gridpt_ref)

    if redcap_results and merge_dists is not None:
        print("Plotting REDCAP distance-cutoff curve ...", flush=True)
        plot_redcap_goldilocks(redcap_results, merge_dists, tau_gridpt_ref)

    print("Plotting comparison maps ...", flush=True)
    km_opt, som_opt, rc_opt = plot_full_comparison(
        km_results, som_results, redcap_results,
        tau_gridpt_map, bss_gridpt_map, lat, lon, valid_mask)

    n_lat, n_lon = len(lat), len(lon)

    # Tau vs cluster size scatter for optimal solutions
    plot_tau_vs_size(km_opt["tau_cl"], km_opt["sizes"],
                     f"K-means k={km_opt['k']}: Cluster τ vs Cluster Size",
                     OUT_DIR / f"kmeans_tau_vs_size_k{km_opt['k']}.png")
    plot_tau_vs_size(som_opt["tau_cl"], som_opt["sizes"],
                     f"SOM {som_opt['m']}×{som_opt['n']}: Cluster τ vs Cluster Size",
                     OUT_DIR / f"som_tau_vs_size_{som_opt['m']}x{som_opt['n']}.png")
    if rc_opt:
        plot_tau_vs_size(rc_opt["tau_cl"], rc_opt["sizes"],
                         f"REDCAP Ward k={rc_opt['k']}: Cluster τ vs Cluster Size",
                         OUT_DIR / f"redcap_tau_vs_size_k{rc_opt['k']}.png")

    # Cluster size histograms
    plot_cluster_size_hist(km_opt["sizes"], km_opt["k"],
                           f"K-means k={km_opt['k']} Cluster Size Distribution",
                           OUT_DIR / f"kmeans_size_hist_k{km_opt['k']}.png")
    plot_cluster_size_hist(som_opt["sizes"], som_opt["k"],
                           f"SOM {som_opt['m']}×{som_opt['n']} Cluster Size Distribution",
                           OUT_DIR / f"som_size_hist_{som_opt['m']}x{som_opt['n']}.png")
    if rc_opt:
        plot_cluster_size_hist(rc_opt["sizes"], rc_opt["k"],
                               f"REDCAP Ward k={rc_opt['k']} Cluster Size Distribution",
                               OUT_DIR / f"redcap_size_hist_k{rc_opt['k']}.png")

    # Save summary table
    import json
    summary = {
        "grid_point_tau": float(tau_gridpt_ref),
        "grid_point_bss": float(bss_gridpt_ref),
        "kmeans": [{"k": r["k"], "tau_domain": float(r["tau_domain"]),
                    "bss_domain": float(r["bss_domain"])} for r in km_results],
        "som": [{"k": r["k"], "m": r["m"], "n": r["n"],
                 "tau_domain": float(r["tau_domain"]),
                 "bss_domain": float(r["bss_domain"])} for r in som_results],
        "redcap": [{"k": r["k"], "tau_domain": float(r["tau_domain"]),
                    "bss_domain": float(r["bss_domain"])} for r in redcap_results],
        "optimal_kmeans_k": km_opt["k"],
        "optimal_som": f"{som_opt['m']}x{som_opt['n']}",
        "optimal_redcap_k": rc_opt["k"] if rc_opt else None,
    }
    (OUT_DIR / "cluster_skill_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {OUT_DIR/'cluster_skill_summary.json'}", flush=True)

    print("\n=== Results ===", flush=True)
    print(f"Grid-point τ: {tau_gridpt_ref:.4f}   BSS: {bss_gridpt_ref:.4f}", flush=True)
    print(f"Optimal K-means:     k={km_opt['k']}  τ={km_opt['tau_domain']:.4f}  BSS={km_opt['bss_domain']:.4f}", flush=True)
    print(f"Optimal SOM:    {som_opt['m']}×{som_opt['n']}  τ={som_opt['tau_domain']:.4f}  BSS={som_opt['bss_domain']:.4f}", flush=True)
    if rc_opt:
        print(f"Optimal REDCAP:      k={rc_opt['k']}  τ={rc_opt['tau_domain']:.4f}  BSS={rc_opt['bss_domain']:.4f}", flush=True)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
