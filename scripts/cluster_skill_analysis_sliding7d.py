#!/usr/bin/env python3
"""Cluster skill analysis using ±7-day LOO seasonal-frequency approach.

Parallel to cluster_skill_analysis.py but uses the 37-year seasonal frequency
arrays from seasonal_jja_skill.py instead of the 111-event per-date pred_prob
from postprocess_jja_lag.py.

pred = ace2_freq  (37, lat, lon) — ACE2 JJA HHE seasonal frequency (continuous)
obs  = era5_freq  (37, lat, lon) — ERA5 JJA HHE seasonal frequency (continuous)

Grid-point reference τ is loaded from skill_jja_seasonal.nc (Kendall τ).

Outputs → outputs/lag_may/cluster_analysis_sliding7d/
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import scipy.sparse as sp
from scipy.stats import kendalltau
from tqdm.auto import tqdm
from minisom import MiniSom
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import cartopy.io.shapereader as shpreader
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seasonal_jja_skill import load_land_mask

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
OUT_DIR     = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_sliding7d"

YEARS = list(range(1980, 2017))

KMEANS_SIZES = [5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]
SOM_SIZES    = [(2, 2), (3, 3), (5, 5), (7, 7), (10, 10), (13, 13), (15, 15), (20, 20)]
REDCAP_SIZES = [5, 10, 25, 50, 100, 200, 400, 800]
SPATIAL_ALPHA = 1.0

CONUS_LAT_SLICE = slice(105, 163)
CONUS_LON_SLICE = slice(200, 305)
KMEANS_SIZES_CONUS = [2, 3, 5, 10, 20, 50, 62, 100, 200, 400]
SOM_SIZES_CONUS    = [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (10, 10), (13, 13)]
REDCAP_SIZES_CONUS = [2, 3, 5, 10, 20, 50, 100, 200]

_LAND_GEOMS  = None
_STATE_GEOMS = None

_TAU_CMAP = "RdBu_r"   # diverging blue-white-red, signed tau (negative=blue, positive=red)

# ── association metric (Kendall τ  vs  modified-Kendall z) ──────────────────────
# METRIC selects how each grid-point / cluster pred–obs series pair is scored.
# "tau"        → scipy Kendall τ (bounded, default; original behavior)
# "modkendall" → modified-Kendall z, top-k weighted (Zheng & Lo 2006)
from mod_kendall_metric import mk_z, DEFAULT_K as _MK_DEFAULT_K  # noqa: E402

METRIC      = "tau"
METRIC_K    = _MK_DEFAULT_K
METRIC_SYM  = "τ"               # colorbar / short label
METRIC_NAME = "Kendall τ"        # axis / legend label
METRIC_FIXED_VLIM = 0.4          # symmetric scale for fixed-range τ panels


def assoc_metric(pred_c, obs_c):
    """Scalar association score for a pred/obs series pair (metric-aware)."""
    if METRIC == "modkendall":
        return mk_z(pred_c, obs_c, METRIC_K)
    tau, _ = kendalltau(pred_c, obs_c)
    return tau


def _metric_vlim(*arrays, fallback=0.4):
    """Symmetric color limit: fixed for τ, data-driven (98th pct) for z."""
    if METRIC != "modkendall":
        return METRIC_FIXED_VLIM
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if a is not None])
    return float(np.nanpercentile(np.abs(vals), 98)) if vals.size else fallback

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        10,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})


# ── border helpers ────────────────────────────────────────────────────────────

def _get_border_geoms():
    global _LAND_GEOMS, _STATE_GEOMS
    if _LAND_GEOMS is None:
        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
        _STATE_GEOMS = list(shpreader.Reader(shp).geometries())
    return _LAND_GEOMS, _STATE_GEOMS


def _draw_borders(ax, xlim, ylim, land_geoms, state_geoms):
    for g in land_geoms:
        parts = [g] if hasattr(g, "exterior") else list(g.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            if float(min(xs)) < xlim[1] and float(max(xs)) > xlim[0]:
                ax.plot(xs, ys, "k-", linewidth=0.5, zorder=3)
            for interior in part.interiors:
                xs, ys = interior.xy
                ax.plot(xs, ys, "k-", linewidth=0.3, zorder=3)
    for g in state_geoms:
        parts = [g] if hasattr(g, "exterior") else list(g.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            if (float(min(xs)) < xlim[1] and float(max(xs)) > xlim[0] and
                    float(min(ys)) < ylim[1] and float(max(ys)) > ylim[0]):
                ax.plot(xs, ys, color="0.5", linewidth=0.25, zorder=2)


def _plain_map_axes(ax, lon_360, lat, pad=3.0):
    lon = lon_360 - 360.0 if float(lon_360.mean()) > 180 else lon_360
    xlim = [float(lon[0]) - 0.5 - pad, float(lon[-1]) + 0.5 + pad]
    ylim = [float(lat[0]) - 0.5 - pad, float(lat[-1]) + 0.5 + pad]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlabel("Longitude")
    xt = np.arange(int(np.ceil(xlim[0] / 10) * 10), int(xlim[1]) + 1, 10, dtype=float)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(abs(x))}°W" if x < 0 else f"{int(x)}°E" for x in xt])
    ax.grid(True, linewidth=0.2, color="gray", alpha=0.4)
    _draw_borders(ax, xlim, ylim, *_get_border_geoms())
    return lon, xlim, ylim


def _roll_to_180(field, lon_360):
    split = np.searchsorted(lon_360, 180.0)
    n = len(lon_360)
    lon_r = np.concatenate([lon_360[split:] - 360.0, lon_360[:split]])
    field_r = np.roll(field, n - split, axis=-1)
    return field_r, lon_r


def _annotate_cluster_tau(ax, labels_2d, tau_cl, lon2d, lat2d, fontsize=7):
    for cl, tau in enumerate(tau_cl):
        if not np.isfinite(tau):
            continue
        mask = labels_2d == cl
        if mask.sum() < 3:
            continue
        lx = float(lon2d[mask].mean())
        ly = float(lat2d[mask].mean())
        ax.text(lx, ly, f"{tau:.2f}", fontsize=fontsize, ha="center", va="center",
                color="black", weight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def plot_cluster_tau_map(tau_map, labels_2d, tau_cl, tau_domain, lat, lon, title, out,
                         vmin=None, vmax=None, cmap=None, label_fontsize=7):
    """Per-cluster τ map (signed): each cluster colored by its own τ on a
    diverging blue-white-red scale, the numeric (signed) τ labeled at the
    cluster centroid, and the domain-mean τ called out in a corner annotation.

    vmax auto-scales to the data when not given: cluster-aggregated τ runs
    much higher than grid-point τ (e.g. ~0.4 at the pixel level vs ~0.8 for
    a 2-cluster split), so a fixed scale saturates and washes out low-k
    cluster maps. vmin defaults to -vmax for a zero-centered diverging scale.
    """
    cmap = cmap if cmap is not None else _TAU_CMAP
    if vmax is None:
        finite = tau_map[np.isfinite(tau_map)]
        vmax = max(float(np.nanmax(np.abs(finite))) * 1.05, 0.05) if finite.size else 0.4
    if vmin is None:
        vmin = -vmax
    if len(lat) < 170:
        lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
        e = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
             float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(tau_map, origin="lower", extent=e, aspect="equal",
                       vmin=vmin, vmax=vmax, cmap=cmap, zorder=1,
                       interpolation="nearest")
        LON2D, LAT2D = np.meshgrid(lon_plot, lat)
        _annotate_cluster_tau(ax, labels_2d, tau_cl, LON2D, LAT2D, label_fontsize)
        _plain_map_axes(ax, lon, lat, pad=0.0)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label=METRIC_SYM)
    else:
        data_r, lon_r = _roll_to_180(tau_map, lon)
        labels_r, _   = _roll_to_180(labels_2d.astype(float), lon)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_facecolor("#d0e8f0")
        LON2D, LAT2D = np.meshgrid(lon_r, lat)
        m = ax.pcolormesh(LON2D, LAT2D, data_r, shading="nearest",
                          cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        _annotate_cluster_tau(ax, labels_r, tau_cl, LON2D, LAT2D, label_fontsize)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_title(title, fontsize=9)
        plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label=METRIC_SYM)

    ax.text(0.02, 0.02, f"Domain-mean {METRIC_SYM} = {tau_domain:.3f}",
           transform=ax.transAxes, fontsize=10, weight="bold",
           ha="left", va="bottom", zorder=7,
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"))

    fig.tight_layout()
    _save_figure(fig, out)


def _single_map_figure(data, lat, lon, title, vmin, vmax, cmap, cbar_label, out):
    """Scalar map — plain matplotlib (regional) or plain-matplotlib global."""
    if len(lat) < 170:
        # Regional: use plain imshow with border overlay
        lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
        e = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
             float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(data, origin="lower", extent=e, aspect="equal",
                       vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)
        _plain_map_axes(ax, lon, lat)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label=cbar_label)
    else:
        # Global: plain pcolormesh, -180/180 centered
        data_r, lon_r = _roll_to_180(data, lon)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_facecolor("#d0e8f0")
        LON2D, LAT2D = np.meshgrid(lon_r, lat)
        m = ax.pcolormesh(LON2D, LAT2D, data_r, shading="nearest",
                          cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        from shapely.geometry import box as sbox
        vp = sbox(-180, -90, 180, 90)
        for geom in _get_border_geoms()[0]:
            try:
                g = geom.intersection(vp)
                if g.is_empty:
                    continue
                parts = [g] if hasattr(g, "exterior") else list(g.geoms)
                for part in parts:
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, "k-", linewidth=0.4, zorder=3)
            except Exception:
                continue
        ax.set_title(title, fontsize=9)
        plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label=cbar_label)
    fig.tight_layout()
    _save_figure(fig, out)


def _lightened_cmap(n_clusters, fade=0.45):
    """Turbo, pre-blended toward white and rendered fully opaque.

    Using Artist alpha< 1 at draw time leaves faint white seams between
    adjacent cells (antialiasing/compositing of the cell edges against the
    white background) — baking the lighter look into the colors themselves
    and drawing opaque avoids that seam entirely.
    """
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    base = colormaps["turbo"].resampled(n_clusters)
    colors = base(np.arange(n_clusters))
    colors[:, :3] = colors[:, :3] * (1 - fade) + fade
    colors[:, 3] = 1.0
    return ListedColormap(colors)


def _cluster_map_figure(labels_2d, lat, lon, n_clusters, title, out):
    cmap = _lightened_cmap(n_clusters)
    if len(lat) < 170:
        lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
        e = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
             float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor("white")
        im = ax.imshow(labels_2d.astype(float), origin="lower", extent=e, aspect="equal",
                       cmap=cmap, vmin=0, vmax=n_clusters - 1, zorder=1,
                       interpolation="nearest")
        _plain_map_axes(ax, lon, lat, pad=0.0)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Cluster ID")
    else:
        data_r, lon_r = _roll_to_180(labels_2d.astype(float), lon)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_facecolor("white")
        LON2D, LAT2D = np.meshgrid(lon_r, lat)
        m = ax.pcolormesh(LON2D, LAT2D, data_r, shading="nearest",
                          cmap=cmap, vmin=0, vmax=n_clusters - 1, zorder=1,
                          antialiased=False)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_title(title, fontsize=10)
        plt.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label="Cluster ID")
    fig.tight_layout()
    _save_figure(fig, out)


def _save_figure(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"wrote {out}", flush=True)


def _three_panel_map(data_list, titles, lat, lon, out, vmin, vmax, cmap, cbar_label):
    stem, suffix, parent = Path(out).stem, Path(out).suffix, Path(out).parent
    for i, (data, title) in enumerate(zip(data_list, titles), 1):
        _single_map_figure(data, lat, lon, title, vmin, vmax, cmap, cbar_label,
                           parent / f"{stem}_row{i}{suffix}")


def _expand_labels(labels, valid_mask, n_lat, n_lon):
    full = np.full(n_lat * n_lon, -1, dtype=int)
    full[valid_mask.ravel()] = labels
    return full.reshape(n_lat, n_lon)


# ── land/ocean domain masking ──────────────────────────────────────────────────

def _apply_domain_mask(domain, lat, lon, *arrays):
    """NaN out grid cells outside `domain` ('land' or 'ocean') in each array.

    Each array is either (n_years, lat, lon) or (lat, lon). Masking here, before
    build_features()/eval_cluster_skill() ever see the data, is enough to keep
    land and ocean from clustering together: valid_mask is derived from
    isfinite(pred), and the REDCAP queen-connectivity graph only connects
    points present in valid_mask, so a masked-out point simply can't bridge
    the two domains.
    """
    land_mask = load_land_mask(lat, lon)
    if land_mask is None:
        print("WARNING: land mask unavailable (no forcing_*.nc found) — ignoring --domain", flush=True)
        return arrays
    keep = land_mask if domain == "land" else ~land_mask
    out = []
    for a in arrays:
        a = a.copy()
        if a.ndim == 3:
            a[:, ~keep] = np.nan
        else:
            a[~keep] = np.nan
        out.append(a)
    return out


# ── clustering evaluation ─────────────────────────────────────────────────────

def build_features(pred, lat, lon, spatial_alpha=SPATIAL_ALPHA):
    n_events, n_lat, n_lon = pred.shape
    valid_mask = np.all(np.isfinite(pred), axis=0)
    temporal = pred[:, valid_mask].T   # (n_valid, n_events)
    mean_ = temporal.mean(axis=1, keepdims=True)
    std_  = temporal.std(axis=1, keepdims=True)
    std_  = np.where(std_ == 0, 1.0, std_)
    temporal_z = (temporal - mean_) / std_

    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    lat_r  = np.deg2rad(lat_g[valid_mask])
    lon_r  = np.deg2rad(lon_g[valid_mask])
    spatial = np.stack([np.sin(lat_r), np.cos(lat_r),
                        np.sin(lon_r), np.cos(lon_r)], axis=1)
    scale = spatial_alpha * np.sqrt(n_events / 4.0)
    spatial_scaled = (spatial * scale).astype(np.float32)

    features_full     = np.concatenate([temporal_z, spatial_scaled], axis=1).astype(np.float32)
    features_temporal = temporal_z.astype(np.float32)
    return features_full, features_temporal, valid_mask


def eval_cluster_skill(pred, obs, labels, valid_mask, lat, n_clusters):
    n_events, n_lat, n_lon = pred.shape
    cos_w      = np.cos(np.deg2rad(lat))[:, np.newaxis]
    cos_w_flat = np.broadcast_to(cos_w, (n_lat, n_lon))[valid_mask]
    pred_v = pred[:, valid_mask].T   # (n_valid, n_events)
    obs_v  = obs[:, valid_mask].T

    tau_cl  = np.full(n_clusters, np.nan, dtype=np.float32)
    bss_cl  = np.full(n_clusters, np.nan, dtype=np.float32)
    sizes   = np.zeros(n_clusters, dtype=int)
    tau_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    bss_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    valid_idx = np.where(valid_mask.ravel())[0]

    for c in range(n_clusters):
        idx = labels == c
        if idx.sum() == 0:
            continue
        sizes[c] = idx.sum()
        w = cos_w_flat[idx]
        w = w / w.sum()
        pred_c = (pred_v[idx] * w[:, np.newaxis]).sum(axis=0)
        obs_c  = (obs_v[idx]  * w[:, np.newaxis]).sum(axis=0)
        ok = np.isfinite(pred_c) & np.isfinite(obs_c)
        if ok.sum() < 10 or obs_c[ok].std() < 1e-9:
            continue
        tau = assoc_metric(pred_c[ok], obs_c[ok])
        tau_cl[c] = tau
        bs      = float(np.mean((pred_c - obs_c) ** 2))
        clim    = float(obs_c.mean())
        bs_clim = float(np.mean((clim - obs_c) ** 2))
        if bs_clim > 0:
            bss_cl[c] = 1.0 - bs / bs_clim
        flat_idx = valid_idx[idx]
        row, col = np.unravel_index(flat_idx, (n_lat, n_lon))
        tau_map[row, col] = tau
        if bs_clim > 0:
            bss_map[row, col] = 1.0 - bs / bs_clim

    return tau_map, bss_map, tau_cl, bss_cl, sizes


def domain_mean_tau(tau_map, lat):
    cos_w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(tau_map)
    num = float(np.nansum(tau_map * cos_w * valid))
    den = float(np.nansum(cos_w * valid))
    return num / den if den > 0 else float("nan")


# ── sweeps ────────────────────────────────────────────────────────────────────

def run_kmeans_sweep(features_full, pred, obs, valid_mask, lat, sizes):
    print("\n=== K-means sweep ===", flush=True)
    results = []
    for k in sizes:
        print(f"  k={k}", flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5,
                             batch_size=min(4096, features_full.shape[0]))
        labels = km.fit_predict(features_full)
        tau_map, bss_map, tau_cl, bss_cl, sz = eval_cluster_skill(pred, obs, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_tau(bss_map, lat)
        print(f"    τ_domain={tau_d:.4f}  BSS_domain={bss_d:.4f}", flush=True)
        results.append(dict(k=k, tau_domain=tau_d, bss_domain=bss_d,
                            tau_map=tau_map, bss_map=bss_map,
                            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sz, labels=labels))
    return results


def build_queen_connectivity(valid_mask):
    n_lat, n_lon = valid_mask.shape
    flat_valid = valid_mask.ravel()
    n_valid = flat_valid.sum()
    compact = np.full(n_lat * n_lon, -1, dtype=np.int32)
    compact[flat_valid] = np.arange(n_valid, dtype=np.int32)
    rows, cols = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            lat_idx, lon_idx = np.where(valid_mask)
            ni, nj = lat_idx + di, lon_idx + dj
            in_bounds = (ni >= 0) & (ni < n_lat) & (nj >= 0) & (nj < n_lon)
            ni, nj, li, lj = ni[in_bounds], nj[in_bounds], lat_idx[in_bounds], lon_idx[in_bounds]
            nbr_flat  = ni * n_lon + nj
            src_flat  = li * n_lon + lj
            nbr_valid = flat_valid[nbr_flat]
            rows.append(compact[src_flat[nbr_valid]])
            cols.append(compact[nbr_flat[nbr_valid]])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                         shape=(n_valid, n_valid))


def _sklearn_to_scipy_linkage(children, distances, n_samples):
    counts = np.zeros(n_samples - 1, dtype=np.float64)
    leaf_counts = np.ones(n_samples, dtype=np.float64)
    for i, (left, right) in enumerate(children):
        lc = leaf_counts[left]  if left  < n_samples else counts[left  - n_samples]
        rc = leaf_counts[right] if right < n_samples else counts[right - n_samples]
        counts[i] = lc + rc
    return np.column_stack([children.astype(np.float64),
                             distances.astype(np.float64), counts])


def run_redcap_sweep(features_temporal, pred, obs, valid_mask, lat, n_thresholds=40):
    from scipy.cluster.hierarchy import fcluster
    print("\n=== REDCAP (Ward, spatially constrained) ===", flush=True)
    conn = build_queen_connectivity(valid_mask)
    n_valid = conn.shape[0]
    print(f"  n_valid={n_valid}", flush=True)
    print("  Fitting full Ward dendrogram ...", flush=True)
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                   linkage="ward", connectivity=conn,
                                   compute_full_tree=True)
    agg.fit(features_temporal)
    linkage_mat  = _sklearn_to_scipy_linkage(agg.children_, agg.distances_, n_valid)
    merge_dists  = agg.distances_
    pct_vals     = np.percentile(merge_dists, np.linspace(1, 99, n_thresholds))
    thresholds   = np.unique(np.round(pct_vals, 6))
    print(f"  Sweeping {len(thresholds)} cutoffs", flush=True)
    results = []
    for d in thresholds:
        raw_labels = fcluster(linkage_mat, t=d, criterion="distance")
        labels = raw_labels - 1
        k = int(labels.max()) + 1
        tau_map, bss_map, tau_cl, bss_cl, sz = eval_cluster_skill(pred, obs, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_tau(bss_map, lat)
        print(f"    cutoff={d:.3f}  K={k:4d}  τ={tau_d:.4f}  BSS={bss_d:.4f}", flush=True)
        results.append(dict(cutoff=float(d), k=k, tau_domain=tau_d, bss_domain=bss_d,
                            tau_map=tau_map, bss_map=bss_map,
                            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sz, labels=labels))
    return results, linkage_mat, merge_dists


def run_som_sweep(features_temporal, pred, obs, valid_mask, lat, sizes):
    print("\n=== SOM sweep ===", flush=True)
    n_features = features_temporal.shape[1]
    results = []
    for m, n in sizes:
        k = m * n
        print(f"  SOM {m}×{n} (k={k})", flush=True)
        som = MiniSom(m, n, n_features, sigma=max(1.0, min(m, n) / 2.0),
                      learning_rate=0.5, random_seed=42)
        som.train_random(features_temporal, num_iteration=2000, verbose=False)
        bmu    = np.array([som.winner(x) for x in tqdm(features_temporal,
                                                        desc=f"BMU {m}×{n}", leave=False)])
        labels = bmu[:, 0] * n + bmu[:, 1]
        tau_map, bss_map, tau_cl, bss_cl, sz = eval_cluster_skill(pred, obs, labels, valid_mask, lat, k)
        tau_d = domain_mean_tau(tau_map, lat)
        bss_d = domain_mean_tau(bss_map, lat)
        print(f"    τ_domain={tau_d:.4f}  BSS_domain={bss_d:.4f}", flush=True)
        results.append(dict(k=k, m=m, n=n, tau_domain=tau_d, bss_domain=bss_d,
                            tau_map=tau_map, bss_map=bss_map,
                            tau_cl=tau_cl, bss_cl=bss_cl, sizes=sz, labels=labels))
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_goldilocks(km_results, som_results, redcap_results, tau_gridpt, bss_gridpt):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    km_k    = [r["k"] for r in km_results]
    som_k   = [r["k"] for r in som_results]
    rc_k    = [r["k"] for r in redcap_results]
    for ax, metric in zip(axes, ("tau_domain", "bss_domain")):
        ax.plot(km_k,  [r[metric] for r in km_results],  "o-",  color="#1f77b4", label="K-means")
        ax.plot(som_k, [r[metric] for r in som_results], "s--", color="#ff7f0e", label="SOM")
        if redcap_results:
            ax.plot(rc_k, [r[metric] for r in redcap_results], "^-", color="#2ca02c", label="REDCAP Ward")
        ref = tau_gridpt if metric == "tau_domain" else bss_gridpt
        lbl = (f"Grid-point {METRIC_SYM}={ref:.3f}" if metric == "tau_domain"
               else f"Grid-point BSS={ref:.3f}")
        ax.axhline(ref, color="0.4", linewidth=1.2, linestyle=":", label=lbl)
        ax.set_xscale("log")
        ax.set_ylabel("Domain-mean " + (METRIC_NAME if "tau" in metric else "BSS"), fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Number of clusters  (log scale)", fontsize=10)
    axes[0].set_title("Goldilocks Resolution  |  ±7-day seasonal frequency  |  ACE2 JJA 1980–2016",
                      fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "goldilocks_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def plot_tau_vs_size(tau_cl, sizes, title, out):
    ok = (sizes > 0) & np.isfinite(tau_cl)
    vlim = _metric_vlim(tau_cl[ok], fallback=0.5) if METRIC == "modkendall" else 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sizes[ok], tau_cl[ok], c=tau_cl[ok], cmap=_TAU_CMAP,
                    alpha=0.6, s=20, vmin=-vlim, vmax=vlim)
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cluster size (n valid grid points)")
    ax.set_ylabel(f"Cluster {METRIC_NAME}")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=METRIC_SYM)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, out)


def plot_comparison(km_results, som_results, redcap_results,
                    tau_gridpt_map, bss_gridpt_map, lat, lon, valid_mask):
    km_opt  = max(km_results,  key=lambda r: r["tau_domain"])
    som_opt = max(som_results, key=lambda r: r["tau_domain"])
    rc_opt  = max(redcap_results, key=lambda r: r["tau_domain"]) if redcap_results else None
    n_lat, n_lon = len(lat), len(lon)

    vlim = _metric_vlim(tau_gridpt_map, km_opt["tau_map"], som_opt["tau_map"])
    _three_panel_map(
        [tau_gridpt_map, km_opt["tau_map"], som_opt["tau_map"]],
        [f"Grid-point {METRIC_SYM} (reference)",
         f"K-means {METRIC_SYM}  k={km_opt['k']}  ({METRIC_SYM}={km_opt['tau_domain']:.3f})",
         f"SOM {METRIC_SYM}  {som_opt['m']}×{som_opt['n']}  ({METRIC_SYM}={som_opt['tau_domain']:.3f})"],
        lat, lon, OUT_DIR / "tau_comparison_maps.png",
        vmin=-vlim, vmax=vlim, cmap=_TAU_CMAP, cbar_label=METRIC_SYM,
    )

    km_labels_full = _expand_labels(km_opt["labels"], valid_mask, n_lat, n_lon)
    plot_cluster_tau_map(km_opt["tau_map"], km_labels_full, km_opt["tau_cl"],
                         km_opt["tau_domain"], lat, lon,
                         f"K-means {METRIC_SYM} per cluster  k={km_opt['k']}",
                         OUT_DIR / f"tau_kmeans_k{km_opt['k']}.png")
    _cluster_map_figure(km_labels_full, lat, lon, int(km_labels_full.max()) + 1,
                        f"K-means cluster assignment  k={km_opt['k']}",
                        OUT_DIR / f"kmeans_cluster_map_k{km_opt['k']}.png")

    som_labels_full = _expand_labels(som_opt["labels"], valid_mask, n_lat, n_lon)
    plot_cluster_tau_map(som_opt["tau_map"], som_labels_full, som_opt["tau_cl"],
                         som_opt["tau_domain"], lat, lon,
                         f"SOM {METRIC_SYM} per cluster  {som_opt['m']}×{som_opt['n']}",
                         OUT_DIR / f"tau_som_{som_opt['m']}x{som_opt['n']}.png")
    _cluster_map_figure(som_labels_full, lat, lon, int(som_labels_full.max()) + 1,
                        f"SOM cluster assignment  {som_opt['m']}×{som_opt['n']}",
                        OUT_DIR / f"som_cluster_map_{som_opt['m']}x{som_opt['n']}.png")

    if rc_opt:
        rc_labels_full = _expand_labels(rc_opt["labels"], valid_mask, n_lat, n_lon)
        plot_cluster_tau_map(rc_opt["tau_map"], rc_labels_full, rc_opt["tau_cl"],
                             rc_opt["tau_domain"], lat, lon,
                             f"REDCAP Ward {METRIC_SYM} per cluster  k={rc_opt['k']}",
                             OUT_DIR / f"tau_redcap_k{rc_opt['k']}.png")
        _cluster_map_figure(rc_labels_full, lat, lon, int(rc_labels_full.max()) + 1,
                            f"REDCAP Ward  k={rc_opt['k']}",
                            OUT_DIR / f"redcap_cluster_map_k{rc_opt['k']}.png")

        # Matched-k comparison: K-means/SOM forced to ~the same cluster count
        # as REDCAP's optimum, instead of each method's own (very different) k.
        km_match = next((r for r in km_results if r["k"] == rc_opt["k"]), None)
        if km_match:
            km_match_labels = _expand_labels(km_match["labels"], valid_mask, n_lat, n_lon)
            plot_cluster_tau_map(km_match["tau_map"], km_match_labels, km_match["tau_cl"],
                                 km_match["tau_domain"], lat, lon,
                                 f"K-means {METRIC_SYM} per cluster  k={km_match['k']}  (matched to REDCAP)",
                                 OUT_DIR / f"tau_kmeans_k{km_match['k']}_matched.png")
            _cluster_map_figure(km_match_labels, lat, lon, int(km_match_labels.max()) + 1,
                                f"K-means cluster assignment  k={km_match['k']}  (matched to REDCAP)",
                                OUT_DIR / f"kmeans_cluster_map_k{km_match['k']}_matched.png")

        som_match = min(som_results, key=lambda r: abs(r["k"] - rc_opt["k"]))
        som_match_labels = _expand_labels(som_match["labels"], valid_mask, n_lat, n_lon)
        plot_cluster_tau_map(som_match["tau_map"], som_match_labels, som_match["tau_cl"],
                             som_match["tau_domain"], lat, lon,
                             f"SOM {METRIC_SYM} per cluster  {som_match['m']}×{som_match['n']}"
                             f"  (k={som_match['k']}, matched to REDCAP k={rc_opt['k']})",
                             OUT_DIR / f"tau_som_{som_match['m']}x{som_match['n']}_matched.png")
        _cluster_map_figure(som_match_labels, lat, lon, int(som_match_labels.max()) + 1,
                            f"SOM cluster assignment  {som_match['m']}×{som_match['n']}"
                            f"  (matched to REDCAP k={rc_opt['k']})",
                            OUT_DIR / f"som_cluster_map_{som_match['m']}x{som_match['n']}_matched.png")
    return km_opt, som_opt, rc_opt


# ── result caching ────────────────────────────────────────────────────────────

def _pack_results(arrays, results, prefix):
    arrays[f"n_{prefix}"] = np.array([len(results)])
    for i, r in enumerate(results):
        for key in ("tau_map", "bss_map", "tau_cl", "bss_cl", "sizes", "labels"):
            arrays[f"{prefix}_{i}_{key}"] = r[key]
        arrays[f"{prefix}_{i}_k"]          = np.array([r["k"]])
        arrays[f"{prefix}_{i}_tau_domain"] = np.array([r["tau_domain"]])
        arrays[f"{prefix}_{i}_bss_domain"] = np.array([r["bss_domain"]])
        for ek in ("m", "n"):
            if ek in r:
                arrays[f"{prefix}_{i}_{ek}"] = np.array([r[ek]])
        if "cutoff" in r:
            arrays[f"{prefix}_{i}_cutoff"] = np.array([r["cutoff"]])


def _unpack_results(d, prefix):
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
        for ek in ("m", "n"):
            key = f"{prefix}_{i}_{ek}"
            if key in d:
                entry[ek] = int(d[key][0])
        if f"{prefix}_{i}_cutoff" in d:
            entry["cutoff"] = float(d[f"{prefix}_{i}_cutoff"][0])
        results.append(entry)
    return results


def save_results(km_results, som_results, redcap_results,
                 tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, merge_dists=None):
    cache = OUT_DIR / "sweep_cache.npz"
    arrays = {"tau_gridpt_map": tau_gridpt_map, "bss_gridpt_map": bss_gridpt_map,
              "valid_mask": valid_mask, "lat": lat, "lon": lon}
    if merge_dists is not None:
        arrays["rc_merge_dists"] = merge_dists
    _pack_results(arrays, km_results,     "km")
    _pack_results(arrays, som_results,    "som")
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
    som_results    = _unpack_results(d, "som")
    redcap_results = _unpack_results(d, "rc") if "n_rc" in d else []
    merge_dists    = d["rc_merge_dists"] if "rc_merge_dists" in d else None
    return km_results, som_results, redcap_results, tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, merge_dists


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global OUT_DIR, KMEANS_SIZES, SOM_SIZES, REDCAP_SIZES
    global METRIC, METRIC_K, METRIC_SYM, METRIC_NAME
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--replot",        action="store_true")
    p.add_argument("--conus",         action="store_true")
    p.add_argument("--skip-redcap",   action="store_true")
    p.add_argument("--redcap-only",   action="store_true",
                   help="Load cached K-means/SOM results, run REDCAP only, re-save and replot")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all",
                   help="Restrict clustering to land-only or ocean-only grid cells")
    p.add_argument("--metric", choices=["tau", "modkendall"], default="tau",
                   help="Association metric: Kendall tau (default) or modified-Kendall z")
    p.add_argument("--metric-k", type=int, default=METRIC_K,
                   help="Truncation value k for the modified-Kendall metric")
    args = p.parse_args()

    METRIC   = args.metric
    METRIC_K = args.metric_k
    metric_suffix = "_modkendall" if METRIC == "modkendall" else ""
    if METRIC == "modkendall":
        METRIC_SYM  = "z"
        METRIC_NAME = f"mod-Kendall z (k={METRIC_K})"
        print(f"Metric: modified-Kendall z  (k={METRIC_K})", flush=True)

    domain_suffix = "" if args.domain == "all" else f"_{args.domain}"
    if args.conus:
        OUT_DIR      = PROJECT_ROOT / f"outputs/lag_may/cluster_analysis_sliding7d_conus{domain_suffix}{metric_suffix}"
        KMEANS_SIZES = KMEANS_SIZES_CONUS
        SOM_SIZES    = SOM_SIZES_CONUS
        REDCAP_SIZES = REDCAP_SIZES_CONUS
        print("CONUS mode", flush=True)
    else:
        OUT_DIR = PROJECT_ROOT / f"outputs/lag_may/cluster_analysis_sliding7d{domain_suffix}{metric_suffix}"
    if args.domain != "all":
        print(f"Domain: {args.domain}-only", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    freq_nc = SLIDING_DIR / "jja_seasonal_freqs.nc"
    # Grid-point reference must use the same metric as the cluster maps.
    if METRIC == "modkendall":
        skill_nc = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall/skill_jja_seasonal.nc"
    else:
        skill_nc = SLIDING_DIR / "skill_jja_seasonal.nc"
    if not freq_nc.exists():
        print(f"ERROR: {freq_nc} not found. Run seasonal_jja_skill.py first.", flush=True)
        return

    if args.redcap_only:
        print("--redcap-only: loading cached K-means/SOM, running REDCAP ...", flush=True)
        km_results, som_results, _, tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, _ = load_results()
        tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
        bss_gridpt_ref = domain_mean_tau(bss_gridpt_map, lat)
        # Rebuild feature matrix for REDCAP
        freq_nc = SLIDING_DIR / "jja_seasonal_freqs.nc"
        ds = xr.open_dataset(freq_nc)
        lat_f = ds["lat"].values; lon_f = ds["lon"].values
        pred_full = ds["ace2_freq"].values.astype(np.float32)
        obs_full  = ds["era5_freq"].values.astype(np.float32)
        ds.close()
        pred = pred_full if not args.conus else pred_full[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        obs  = obs_full  if not args.conus else obs_full[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        if args.domain != "all":
            pred, obs = _apply_domain_mask(args.domain, lat, lon, pred, obs)
        _, features_temporal, _ = build_features(pred, lat, lon)
        redcap_results, _, merge_dists = run_redcap_sweep(features_temporal, pred, obs, valid_mask, lat)
        save_results(km_results, som_results, redcap_results,
                     tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon,
                     merge_dists=merge_dists)
    elif args.replot:
        print("--replot: loading cached sweep results ...", flush=True)
        km_results, som_results, redcap_results, tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon, merge_dists = load_results()
        tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
        bss_gridpt_ref = domain_mean_tau(bss_gridpt_map, lat)
    else:
        print(f"Loading seasonal frequencies from {freq_nc}", flush=True)
        ds = xr.open_dataset(freq_nc)
        lat = ds["lat"].values
        lon = ds["lon"].values
        pred = ds["ace2_freq"].values.astype(np.float32)   # (37, 180, 360)
        obs  = ds["era5_freq"].values.astype(np.float32)
        ds.close()
        print(f"  pred/obs: {pred.shape}", flush=True)

        print(f"Loading grid-point τ reference from {skill_nc}", flush=True)
        ds = xr.open_dataset(skill_nc)
        tau_gridpt_map = ds["kendall_tau"].values.astype(np.float32)
        ds.close()
        tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
        print(f"  Grid-point domain-mean τ = {tau_gridpt_ref:.4f}", flush=True)

        bs      = np.nanmean((pred - obs) ** 2, axis=0)
        clim    = np.nanmean(obs, axis=0, keepdims=True)
        bs_clim = np.nanmean((clim - obs) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bss_gridpt_map = np.where(bs_clim > 0, 1.0 - bs / bs_clim, np.nan).astype(np.float32)
        bss_gridpt_ref = domain_mean_tau(bss_gridpt_map, lat)
        print(f"  Grid-point domain-mean BSS = {bss_gridpt_ref:.4f}", flush=True)

        if args.conus:
            pred           = pred[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
            obs            = obs[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
            tau_gridpt_map = tau_gridpt_map[CONUS_LAT_SLICE, CONUS_LON_SLICE]
            bss_gridpt_map = bss_gridpt_map[CONUS_LAT_SLICE, CONUS_LON_SLICE]
            lat = lat[CONUS_LAT_SLICE]
            lon = lon[CONUS_LON_SLICE]
            tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
            bss_gridpt_ref = domain_mean_tau(bss_gridpt_map, lat)
            print(f"  CONUS τ={tau_gridpt_ref:.4f}  BSS={bss_gridpt_ref:.4f}", flush=True)

        if args.domain != "all":
            pred, obs, tau_gridpt_map, bss_gridpt_map = _apply_domain_mask(
                args.domain, lat, lon, pred, obs, tau_gridpt_map, bss_gridpt_map)
            tau_gridpt_ref = domain_mean_tau(tau_gridpt_map, lat)
            bss_gridpt_ref = domain_mean_tau(bss_gridpt_map, lat)
            n_valid = int(np.isfinite(tau_gridpt_map).sum())
            print(f"  {args.domain}-only: τ={tau_gridpt_ref:.4f}  BSS={bss_gridpt_ref:.4f}  "
                  f"n_valid={n_valid}", flush=True)

        print("Building feature matrices ...", flush=True)
        features_full, features_temporal, valid_mask = build_features(pred, lat, lon)
        print(f"  features_full: {features_full.shape}  valid: {valid_mask.sum()}", flush=True)

        km_results  = run_kmeans_sweep(features_full, pred, obs, valid_mask, lat, KMEANS_SIZES)
        som_results = run_som_sweep(features_temporal, pred, obs, valid_mask, lat, SOM_SIZES)

        if args.skip_redcap:
            redcap_results, merge_dists = [], None
        else:
            redcap_results, _, merge_dists = run_redcap_sweep(
                features_temporal, pred, obs, valid_mask, lat)

        save_results(km_results, som_results, redcap_results,
                     tau_gridpt_map, bss_gridpt_map, valid_mask, lat, lon,
                     merge_dists=merge_dists)

    print("\nPlotting Goldilocks curve ...", flush=True)
    plot_goldilocks(km_results, som_results, redcap_results, tau_gridpt_ref, bss_gridpt_ref)

    print("Plotting comparison maps ...", flush=True)
    km_opt, som_opt, rc_opt = plot_comparison(
        km_results, som_results, redcap_results,
        tau_gridpt_map, bss_gridpt_map, lat, lon, valid_mask)

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

    import json
    summary = {
        "grid_point_tau": float(tau_gridpt_ref),
        "grid_point_bss": float(bss_gridpt_ref),
        "kmeans":  [{"k": r["k"], "tau_domain": float(r["tau_domain"]),
                     "bss_domain": float(r["bss_domain"])} for r in km_results],
        "som":     [{"k": r["k"], "m": r["m"], "n": r["n"],
                     "tau_domain": float(r["tau_domain"]),
                     "bss_domain": float(r["bss_domain"])} for r in som_results],
        "redcap":  [{"k": r["k"], "tau_domain": float(r["tau_domain"]),
                     "bss_domain": float(r["bss_domain"])} for r in redcap_results],
        "optimal_kmeans_k": km_opt["k"],
        "optimal_som": f"{som_opt['m']}x{som_opt['n']}",
        "optimal_redcap_k": rc_opt["k"] if rc_opt else None,
    }
    (OUT_DIR / "cluster_skill_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved → {OUT_DIR / 'cluster_skill_summary.json'}", flush=True)
    print(f"Grid-point τ={tau_gridpt_ref:.4f}  BSS={bss_gridpt_ref:.4f}", flush=True)
    print(f"Optimal K-means: k={km_opt['k']}  τ={km_opt['tau_domain']:.4f}", flush=True)
    print(f"Optimal SOM: {som_opt['m']}×{som_opt['n']}  τ={som_opt['tau_domain']:.4f}", flush=True)
    if rc_opt:
        print(f"Optimal REDCAP: k={rc_opt['k']}  τ={rc_opt['tau_domain']:.4f}", flush=True)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
