#!/usr/bin/env python3
"""Matrix-factorization-only seasonal-frequency cluster skill panel.

Parallel to raw_frequency_cluster_metric_panel.py, but every column is a
matrix factorization variant from the NMF clustering literature:

  - Frobenius NMF on signed-split features
  - KL-NMF / PLSI-style NMF on signed-split features
  - Semi-NMF on signed z-scored temporal features
  - Convex-NMF on signed z-scored temporal features
  - RBF kernel Convex-NMF
  - Symmetric NMF on an RBF similarity matrix
  - Orthogonal tri-factorization on signed-split features

Rows are the same two skill scores as raw_frequency_cluster_metric_panel.py:

  top row    : Kendall tau
  bottom row : normalized modified-Kendall z

Input:
  outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc
    ace2_freq, era5_freq

Output:
  outputs/lag_may/cluster_analysis_sliding7d_conus/matrix_factorization_metric_panel/
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import kendalltau
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mod_kendall_metric import DEFAULT_K as DEFAULT_MODK_K, mk_z, normalized_z_for_plot  # noqa: E402


FREQ_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
OUT_DIR = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_sliding7d_conus/matrix_factorization_metric_panel"
CONUS_LAT_SLICE = slice(105, 163)
CONUS_LON_SLICE = slice(200, 305)

DEFAULT_K = 5
DEFAULT_MAX_ITER = 120
DEFAULT_TRI_FEATURE_K = None
MODK_K = DEFAULT_MODK_K
TAU_PLOT_VLIM = 1.0
EPS = 1e-9
_LAND_GEOMS = None
_STATE_GEOMS = None

METHODS = {
    "nmf": "NMF-Frob",
    "kl_nmf": "KL-NMF",
    "semi_nmf": "Semi-NMF",
    "convex_nmf": "Convex-NMF",
    "kernel_nmf": "Kernel-NMF",
    "symmetric_nmf": "Sym-NMF",
    "tri_nmf": "Tri-NMF",
}


def load_conus_frequency():
    """CONUS ACE2/ERA5 JJA sliding-7d seasonal-frequency fields."""
    if not FREQ_NC.exists():
        raise FileNotFoundError(FREQ_NC)
    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        pred = ds["ace2_freq"].values.astype(np.float32)
        obs = ds["era5_freq"].values.astype(np.float32)
    return (
        pred[:, CONUS_LAT_SLICE, CONUS_LON_SLICE],
        obs[:, CONUS_LAT_SLICE, CONUS_LON_SLICE],
        lat[CONUS_LAT_SLICE],
        lon[CONUS_LON_SLICE],
    )


def build_features(pred: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    n_events, _n_lat, _n_lon = pred.shape
    valid_mask = np.all(np.isfinite(pred), axis=0)
    temporal = pred[:, valid_mask].T
    mean_ = temporal.mean(axis=1, keepdims=True)
    std_ = temporal.std(axis=1, keepdims=True)
    std_ = np.where(std_ == 0, 1.0, std_)
    temporal_z = (temporal - mean_) / std_

    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    lat_r = np.deg2rad(lat_g[valid_mask])
    lon_r = np.deg2rad(lon_g[valid_mask])
    spatial = np.stack(
        [np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)],
        axis=1,
    )
    scale = np.sqrt(n_events / 4.0)
    spatial_scaled = (spatial * scale).astype(np.float32)
    features_full = np.concatenate([temporal_z, spatial_scaled], axis=1).astype(np.float32)
    features_temporal = temporal_z.astype(np.float32)
    return features_full, features_temporal, valid_mask


def expand_labels(labels: np.ndarray, valid_mask: np.ndarray, n_lat: int, n_lon: int):
    full = np.full(n_lat * n_lon, -1, dtype=np.int32)
    full[valid_mask.ravel()] = labels
    return full.reshape(n_lat, n_lon)


def domain_mean(field: np.ndarray, lat: np.ndarray):
    cos_w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(field)
    num = float(np.nansum(field * cos_w * valid))
    den = float(np.nansum(cos_w * valid))
    return num / den if den > 0 else float("nan")


def eval_cluster_metric(pred, obs, labels, valid_mask, lat, n_clusters, metric):
    n_years, n_lat, n_lon = pred.shape
    cos_w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    cos_w_flat = np.broadcast_to(cos_w, (n_lat, n_lon))[valid_mask]
    pred_v = pred[:, valid_mask].T
    obs_v = obs[:, valid_mask].T
    valid_idx = np.where(valid_mask.ravel())[0]

    metric_cl = np.full(n_clusters, np.nan, dtype=np.float32)
    metric_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    sizes = np.zeros(n_clusters, dtype=np.int32)

    for c in range(n_clusters):
        idx = labels == c
        if not np.any(idx):
            continue
        sizes[c] = int(idx.sum())
        w = cos_w_flat[idx].astype(np.float64)
        w /= w.sum()
        pred_c = (pred_v[idx] * w[:, np.newaxis]).sum(axis=0)
        obs_c = (obs_v[idx] * w[:, np.newaxis]).sum(axis=0)
        ok = np.isfinite(pred_c) & np.isfinite(obs_c)
        if int(ok.sum()) < 10 or pred_c[ok].std() < 1e-9 or obs_c[ok].std() < 1e-9:
            continue
        if metric == "tau":
            val = float(kendalltau(pred_c[ok], obs_c[ok])[0])
        elif metric == "modkendall":
            val = mk_z(pred_c[ok], obs_c[ok], MODK_K)
        else:
            raise ValueError(metric)
        metric_cl[c] = val
        rows, cols = np.unravel_index(valid_idx[idx], (n_lat, n_lon))
        metric_map[rows, cols] = val

    return metric_map, metric_cl, sizes, domain_mean(metric_map, lat)


def _get_border_geoms():
    global _LAND_GEOMS, _STATE_GEOMS
    if _LAND_GEOMS is None:
        import cartopy.io.shapereader as shpreader

        shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
        _LAND_GEOMS = list(shpreader.Reader(shp).geometries())
        shp = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces_lakes",
        )
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


def lon_for_plot(lon):
    return lon - 360.0 if float(np.nanmean(lon)) > 180.0 else lon


def setup_map_axes(ax, lat, lon):
    lon_plot = lon_for_plot(lon)
    xlim = (float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5)
    ylim = (float(lat[0]) - 0.5, float(lat[-1]) + 0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    xticks = np.arange(np.ceil(xlim[0] / 10.0) * 10.0, np.floor(xlim[1] / 10.0) * 10.0 + 1, 10)
    yticks = np.arange(np.ceil(ylim[0] / 5.0) * 5.0, np.floor(ylim[1] / 5.0) * 5.0 + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(int(x))}W" for x in xticks], fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}N" for y in yticks], fontsize=8)
    ax.grid(True, linewidth=0.25, color="0.6", alpha=0.35)
    _draw_borders(ax, xlim, ylim, *_get_border_geoms())


def annotate_clusters(ax, labels_2d, values, lat, lon):
    lon_plot = lon_for_plot(lon)
    lon2d, lat2d = np.meshgrid(lon_plot, lat)
    for cluster_id, value in enumerate(values):
        if not np.isfinite(value):
            continue
        mask = labels_2d == cluster_id
        if int(mask.sum()) < 3:
            continue
        ax.text(
            float(lon2d[mask].mean()),
            float(lat2d[mask].mean()),
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            weight="bold",
            color="black",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )


def draw_metric_map(ax, metric_map, labels_2d, metric_cl, domain, lat, lon, title, vlim):
    lon_plot = lon_for_plot(lon)
    extent = [
        float(lon_plot[0]) - 0.5,
        float(lon_plot[-1]) + 0.5,
        float(lat[0]) - 0.5,
        float(lat[-1]) + 0.5,
    ]
    im = ax.imshow(
        metric_map,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        zorder=1,
    )
    setup_map_axes(ax, lat, lon)
    annotate_clusters(ax, labels_2d, metric_cl, lat, lon)
    ax.set_title(title, fontsize=10, pad=4)
    ax.text(
        0.02,
        0.03,
        f"domain avg = {domain:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="0.25"),
        zorder=7,
    )
    return im


def finite_vlim(*arrays, floor=0.05, cap=None):
    vals = []
    for arr in arrays:
        good = np.asarray(arr)[np.isfinite(arr)]
        if good.size:
            vals.append(np.abs(good))
    if not vals:
        return floor
    v = max(float(np.nanpercentile(np.concatenate(vals), 98)), floor)
    return min(v, cap) if cap is not None else v


def render_panel(results, methods, lat, lon, out_png, suptitle):
    ncol = len(methods)
    z_scale = finite_vlim(*(results[m]["z_map"] for m in methods), floor=1.0)
    for method in methods:
        results[method]["z_plot_map"], _ = normalized_z_for_plot(
            results[method]["z_map"], scale=z_scale
        )
        results[method]["z_plot_cl"] = np.clip(
            results[method]["z_cl"] / z_scale, -1.0, 1.0
        ).astype(np.float32)
        results[method]["z_plot_domain"] = domain_mean(results[method]["z_plot_map"], lat)

    fig = plt.figure(figsize=(6.0 * ncol, 10), facecolor="white")
    gs = fig.add_gridspec(
        2,
        ncol,
        left=0.045,
        right=0.985,
        top=0.89,
        bottom=0.12,
        wspace=0.08,
        hspace=0.18,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(2)])
    tau_im = z_im = None
    for c, method in enumerate(methods):
        r = results[method]
        nice = r["label"]
        tau_im = draw_metric_map(
            axes[0, c],
            r["tau_map"],
            r["labels_2d"],
            r["tau_cl"],
            r["tau_domain"],
            lat,
            lon,
            f"{nice}  K={r['n_clusters']}  |  Kendall tau",
            TAU_PLOT_VLIM,
        )
        z_im = draw_metric_map(
            axes[1, c],
            r["z_plot_map"],
            r["labels_2d"],
            r["z_plot_cl"],
            r["z_plot_domain"],
            lat,
            lon,
            f"{nice}  K={r['n_clusters']}  |  normalized modified-Kendall z (top-k={MODK_K})",
            1.0,
        )

    cb_w, cb_h, cb_y, cb_gap = 0.34, 0.022, 0.06, 0.04
    cb_left = (1.0 - (2 * cb_w + cb_gap)) / 2.0
    tau_cax = fig.add_axes([cb_left, cb_y, cb_w, cb_h])
    z_cax = fig.add_axes([cb_left + cb_w + cb_gap, cb_y, cb_w, cb_h])
    tau_cbar = fig.colorbar(tau_im, cax=tau_cax, orientation="horizontal")
    tau_cbar.set_label("Kendall tau", fontsize=9)
    z_cbar = fig.colorbar(z_im, cax=z_cax, orientation="horizontal")
    z_cbar.set_label(f"normalized modified-Kendall z (k={MODK_K})", fontsize=9)
    for cbar in (tau_cbar, z_cbar):
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(suptitle, fontsize=13, y=0.96)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def signed_split(x: np.ndarray) -> np.ndarray:
    """Represent signed features as nonnegative positive/negative channels."""
    pos = np.clip(x, 0.0, None)
    neg = np.clip(-x, 0.0, None)
    return np.concatenate([pos, neg], axis=1).astype(np.float64, copy=False)


def remap_labels(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    out = np.empty_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(ids):
        out[labels == old_id] = new_id
    return out


def labels_from_scores(scores: np.ndarray) -> np.ndarray:
    return remap_labels(np.argmax(scores, axis=1).astype(np.int32))


def pos_part(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0.0)


def neg_part(a: np.ndarray) -> np.ndarray:
    return np.maximum(-a, 0.0)


def normalize_columns(a: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    norms = np.maximum(np.linalg.norm(a, axis=0), eps)
    return a / norms[np.newaxis, :], norms


def rbf_similarity(x: np.ndarray, gamma: float | None = None) -> tuple[np.ndarray, float]:
    """Dense RBF similarity for CONUS-sized grids."""
    x64 = x.astype(np.float64, copy=False)
    if gamma is None:
        gamma = 1.0 / max(1, x64.shape[1])
    sq = np.sum(x64 * x64, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (x64 @ x64.T), 0.0)
    k_mat = np.exp(-gamma * dist2).astype(np.float64, copy=False)
    return k_mat, float(gamma)


def nmf_labels(features_temporal: np.ndarray, k: int, max_iter: int, beta_loss="frobenius"):
    x = signed_split(features_temporal)
    kwargs = {
        "n_components": k,
        "init": "nndsvda",
        "random_state": 42,
        "max_iter": max_iter,
    }
    if beta_loss != "frobenius":
        kwargs.update({"solver": "mu", "beta_loss": beta_loss})
        x = x + EPS
    model = NMF(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        w = model.fit_transform(x)
    h = model.components_

    # Li & Ding's posterior normalization for standard NMF is proportional to
    # sample membership times the L1 mass of each basis vector.
    scores = w * np.maximum(h.sum(axis=1), EPS)[np.newaxis, :]
    return labels_from_scores(scores), {
        "factorization": "NMF",
        "objective": beta_loss,
        "input_transform": "positive/negative signed split",
        "reconstruction_error": float(getattr(model, "reconstruction_err_", np.nan)),
        "n_iter": int(getattr(model, "n_iter_", -1)),
    }


def semi_nmf_labels(features_temporal: np.ndarray, k: int, max_iter: int):
    """Semi-NMF: X ~= W H, W >= 0 and H signed."""
    x = features_temporal.astype(np.float64, copy=False)
    rng = np.random.default_rng(42)
    w = rng.random((x.shape[0], k)) + 0.1
    h = np.zeros((k, x.shape[1]), dtype=np.float64)

    for _ in range(max_iter):
        h = np.linalg.lstsq(w, x, rcond=None)[0]
        xht = x @ h.T
        hht = h @ h.T
        num = pos_part(xht) + w @ neg_part(hht)
        den = neg_part(xht) + w @ pos_part(hht) + EPS
        w *= np.sqrt(num / den)
        w = np.maximum(w, EPS)
        w, norms = normalize_columns(w)
        h *= norms[:, np.newaxis]

    basis_norm = np.maximum(np.linalg.norm(h, axis=1), EPS)
    scores = w * basis_norm[np.newaxis, :]
    return labels_from_scores(scores), {
        "factorization": "Semi-NMF",
        "objective": "squared reconstruction error",
        "input_transform": "signed z-scored temporal features",
        "n_iter": int(max_iter),
    }


def convex_nmf_from_kernel(
    kernel: np.ndarray,
    k: int,
    max_iter: int,
    kernel_name: str,
    gamma: float | None = None,
):
    """Convex-NMF multiplicative updates using a linear or nonlinear kernel."""
    rng = np.random.default_rng(42)
    n = kernel.shape[0]
    w = rng.random((n, k)) + 0.1  # convex basis weights
    g = rng.random((n, k)) + 0.1  # sample memberships

    kp = pos_part(kernel)
    kn = neg_part(kernel)

    for _ in range(max_iter):
        kp_w = kp @ w
        kn_w = kn @ w
        num_g = kp_w + g @ (w.T @ kn_w)
        den_g = kn_w + g @ (w.T @ kp_w) + EPS
        g *= np.sqrt(num_g / den_g)
        g = np.maximum(g, EPS)

        kp_g = kp @ g
        kn_g = kn @ g
        gtg = g.T @ g
        num_w = kp_g + kn_w @ gtg
        den_w = kn_g + kp_w @ gtg + EPS
        w *= np.sqrt(num_w / den_w)
        w = np.maximum(w, EPS)

        g, g_norm = normalize_columns(g)
        w *= g_norm[np.newaxis, :]
        w, w_norm = normalize_columns(w)
        g *= w_norm[np.newaxis, :]

    scores = g * np.maximum(w.sum(axis=0), EPS)[np.newaxis, :]
    info = {
        "factorization": "Convex-NMF",
        "kernel": kernel_name,
        "n_iter": int(max_iter),
    }
    if gamma is not None:
        info["kernel_gamma"] = float(gamma)
    return labels_from_scores(scores), info


def convex_nmf_labels(features_temporal: np.ndarray, k: int, max_iter: int):
    x = features_temporal.astype(np.float64, copy=False)
    kernel = x @ x.T
    return convex_nmf_from_kernel(kernel, k, max_iter, "linear")


def kernel_nmf_labels(features_temporal: np.ndarray, k: int, max_iter: int, gamma: float | None):
    kernel, used_gamma = rbf_similarity(features_temporal, gamma)
    return convex_nmf_from_kernel(kernel, k, max_iter, "rbf", gamma=used_gamma)


def symmetric_nmf_labels(features_temporal: np.ndarray, k: int, max_iter: int, gamma: float | None):
    kernel, used_gamma = rbf_similarity(features_temporal, gamma)
    rng = np.random.default_rng(42)
    h = rng.random((kernel.shape[0], k)) + 0.1

    for _ in range(max_iter):
        num = kernel @ h
        den = h @ (h.T @ h) + EPS
        h *= num / den
        h = np.maximum(h, EPS)
        h, _ = normalize_columns(h)

    return labels_from_scores(h), {
        "factorization": "Symmetric NMF",
        "objective": "K ~= H H^T",
        "kernel": "rbf",
        "kernel_gamma": float(used_gamma),
        "n_iter": int(max_iter),
    }


def tri_nmf_labels(features_temporal: np.ndarray, k: int, feature_k: int, max_iter: int):
    """Orthogonal-style tri-factorization: X ~= F S G^T."""
    x = signed_split(features_temporal)
    rng = np.random.default_rng(42)
    f = rng.random((x.shape[0], k)) + 0.1
    g = rng.random((x.shape[1], feature_k)) + 0.1
    s = rng.random((k, feature_k)) + 0.1

    for _ in range(max_iter):
        xgs = x @ g @ s.T
        f *= np.sqrt(xgs / (f @ (f.T @ xgs) + EPS))
        f = np.maximum(f, EPS)

        xtfs = x.T @ f @ s
        g *= np.sqrt(xtfs / (g @ (g.T @ xtfs) + EPS))
        g = np.maximum(g, EPS)

        num_s = f.T @ x @ g
        den_s = (f.T @ f) @ s @ (g.T @ g) + EPS
        s *= np.sqrt(num_s / den_s)
        s = np.maximum(s, EPS)

        f, f_norm = normalize_columns(f)
        s *= f_norm[:, np.newaxis]
        g, g_norm = normalize_columns(g)
        s *= g_norm[np.newaxis, :]

    return labels_from_scores(f), {
        "factorization": "Tri-factorization",
        "objective": "X ~= F S G^T",
        "input_transform": "positive/negative signed split",
        "row_components": int(k),
        "feature_components": int(feature_k),
        "n_iter": int(max_iter),
    }


def fit_factorization_labels(
    method: str,
    features_temporal: np.ndarray,
    k: int,
    max_iter: int,
    tri_feature_k: int,
    kernel_gamma: float | None,
):
    if method == "nmf":
        return nmf_labels(features_temporal, k, max_iter, beta_loss="frobenius")
    if method == "kl_nmf":
        labels, info = nmf_labels(features_temporal, k, max_iter, beta_loss="kullback-leibler")
        info["factorization"] = "KL-NMF / PLSI-style NMF"
        return labels, info
    if method == "semi_nmf":
        return semi_nmf_labels(features_temporal, k, max_iter)
    if method == "convex_nmf":
        return convex_nmf_labels(features_temporal, k, max_iter)
    if method == "kernel_nmf":
        return kernel_nmf_labels(features_temporal, k, max_iter, kernel_gamma)
    if method == "symmetric_nmf":
        return symmetric_nmf_labels(features_temporal, k, max_iter, kernel_gamma)
    if method == "tri_nmf":
        return tri_nmf_labels(features_temporal, k, tri_feature_k, max_iter)
    raise ValueError(method)


def parse_methods(raw_methods: str) -> list[str]:
    if raw_methods.strip().lower() == "all":
        return list(METHODS)
    methods = [m.strip() for m in raw_methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"Unknown methods {bad}; choices are {list(METHODS)} or all")
    return methods


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Number of grid-cell clusters/components for each method")
    p.add_argument("--methods", default="all",
                   help="Comma-separated subset, or 'all'. Choices: " + ", ".join(METHODS))
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help="Maximum multiplicative-update iterations")
    p.add_argument("--tri-feature-k", type=int, default=DEFAULT_TRI_FEATURE_K,
                   help="Number of feature/year clusters for Tri-NMF; defaults to --k")
    p.add_argument("--kernel-gamma", type=float, default=None,
                   help="RBF gamma for kernel/symmetric NMF; default is 1 / n_features")
    p.add_argument("--metric-k", type=int, default=DEFAULT_MODK_K,
                   help="Top-k truncation value for modified-Kendall z")
    args = p.parse_args()

    methods = parse_methods(args.methods)
    tri_feature_k = args.tri_feature_k if args.tri_feature_k is not None else args.k
    global MODK_K
    MODK_K = args.metric_k

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Matrix-factorization methods: {', '.join(methods)}", flush=True)
    print(f"Cluster components: k={args.k}; Tri-NMF feature components={tri_feature_k}", flush=True)
    print(f"Modified-Kendall top-k truncation: k={MODK_K}", flush=True)

    pred, obs, lat, lon = load_conus_frequency()
    _features_full, features_temporal, valid_mask = build_features(pred, lat, lon)
    n_lat, n_lon = len(lat), len(lon)
    print(f"CONUS frequency fields: pred/obs={pred.shape} temporal_features={features_temporal.shape}", flush=True)

    results = {}
    summary = {
        "input": str(FREQ_NC),
        "clustering_field": "ACE2 JJA sliding-7d seasonal frequency",
        "skill_pair": "ACE2 frequency vs ERA5 frequency",
        "metric": f"modified-Kendall z uses truncation k={MODK_K}",
        "requested_components": int(args.k),
        "max_iter": int(args.max_iter),
        "methods": {},
    }

    for method in methods:
        label = METHODS[method]
        print(f"{label}: fitting k={args.k}", flush=True)
        labels, info = fit_factorization_labels(
            method,
            features_temporal,
            args.k,
            args.max_iter,
            tri_feature_k,
            args.kernel_gamma,
        )
        n_clusters = int(labels.max()) + 1
        labels_2d = expand_labels(labels, valid_mask, n_lat, n_lon)

        tau_map, tau_cl, sizes, tau_domain = eval_cluster_metric(
            pred, obs, labels, valid_mask, lat, n_clusters, "tau",
        )
        z_map, z_cl, _sizes_z, z_domain = eval_cluster_metric(
            pred, obs, labels, valid_mask, lat, n_clusters, "modkendall",
        )

        results[method] = {
            "label": label,
            "n_clusters": n_clusters,
            "labels_2d": labels_2d,
            "tau_map": tau_map,
            "tau_cl": tau_cl,
            "tau_domain": tau_domain,
            "z_map": z_map,
            "z_cl": z_cl,
            "z_domain": z_domain,
            "sizes": sizes,
        }
        summary["methods"][method] = {
            **info,
            "label": label,
            "requested_k": int(args.k),
            "rendered_k": int(n_clusters),
            "domain_avg_tau": round(float(tau_domain), 4),
            "domain_avg_modkendall_z": round(float(z_domain), 4),
            "cluster_sizes": [int(x) for x in sizes],
        }
        print(
            f"  {label}: tau={tau_domain:.3f}  z={z_domain:.3f}  "
            f"sizes={[int(x) for x in sizes]}",
            flush=True,
        )

    tag = "_".join(methods)
    metric_suffix = "" if MODK_K == DEFAULT_MODK_K else f"_mk{MODK_K}"
    out_png = OUT_DIR / f"matrix_factorization_cluster_metric_panel_k{args.k}_{tag}{metric_suffix}.png"
    render_panel(
        results,
        methods,
        lat,
        lon,
        out_png,
        suptitle="CONUS sliding-7d seasonal-frequency matrix-factorization clusters: ACE2 vs ERA5 skill",
    )

    z_scale = finite_vlim(*(results[m]["z_map"] for m in methods), floor=1.0)
    for method in methods:
        summary["methods"][method]["kendall_tau_plot_range"] = [
            -TAU_PLOT_VLIM,
            TAU_PLOT_VLIM,
        ]
        summary["methods"][method]["modkendall_plot_normalization_scale"] = round(float(z_scale), 4)
        summary["methods"][method]["domain_avg_normalized_modkendall_z"] = round(
            float(results[method]["z_plot_domain"]), 4
        )

    out_json = OUT_DIR / f"matrix_factorization_cluster_metric_panel_summary_k{args.k}_{tag}{metric_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
