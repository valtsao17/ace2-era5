#!/usr/bin/env python3
"""One-panel raw-data cluster skill comparison.

Clusters CONUS grid cells by raw JJA-mean Tmax trajectories, then scores the
same labels with:

  top row    : Kendall tau
  bottom row : modified-Kendall z

Requested fixed cluster counts:
  K-means = 5, SOM = 8x1, REDCAP = 5.

Output -> outputs/lag_may/cluster_analysis_rawdata_conus/raw_metric_panel/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom
from scipy.cluster.hierarchy import fcluster
from scipy.stats import kendalltau
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import NMF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cluster_skill_analysis_rawdata as raw  # noqa: E402
from mod_kendall_metric import DEFAULT_K as DEFAULT_MODK_K, mk_z, normalized_z_for_plot  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_rawdata_conus/raw_metric_panel"
MODK_K = DEFAULT_MODK_K

DEFAULT_KMEANS = 5
DEFAULT_SOM = 8
DEFAULT_REDCAP = 5
DEFAULT_NMF = 5
TAU_PLOT_VLIM = 1.0


def load_conus_raw(recompute_raw=False):
    if raw.RAW_NC.exists() and not recompute_raw:
        with xr.open_dataset(raw.RAW_NC) as ds:
            lat = ds["lat"].values
            lon = ds["lon"].values
            pred = ds["ace2_raw_tmax"].values.astype(np.float32)
            obs = ds["era5_raw_tmax"].values.astype(np.float32)
    else:
        pred, obs, lat, lon = raw.compute_jja_raw_means()
        raw.RAW_DIR.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {
                "ace2_raw_tmax": (("year", "lat", "lon"), pred),
                "era5_raw_tmax": (("year", "lat", "lon"), obs),
            },
            coords={"year": raw.YEARS, "lat": lat, "lon": lon},
        ).to_netcdf(raw.RAW_NC)
        print(f"wrote {raw.RAW_NC}", flush=True)

    return (
        pred[:, raw.CONUS_LAT_SLICE, raw.CONUS_LON_SLICE],
        obs[:, raw.CONUS_LAT_SLICE, raw.CONUS_LON_SLICE],
        lat[raw.CONUS_LAT_SLICE],
        lon[raw.CONUS_LON_SLICE],
    )


def remap_labels(labels):
    ids = np.unique(labels)
    out = np.empty_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(ids):
        out[labels == old_id] = new_id
    return out


def kmeans_labels(features_full, k):
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=5,
        batch_size=min(4096, features_full.shape[0]),
    )
    return km.fit_predict(features_full).astype(np.int32)


def som_labels(features_temporal, k):
    som = MiniSom(
        k,
        1,
        features_temporal.shape[1],
        sigma=max(1.0, k / 4.0),
        learning_rate=0.5,
        random_seed=42,
    )
    som.train_random(features_temporal, num_iteration=2000, verbose=False)
    return np.array([som.winner(x)[0] for x in features_temporal], dtype=np.int32)


def nmf_labels(features_temporal, k):
    # NMF requires a non-negative matrix, but the temporal features are z-scored
    # (signed). Use the standard signed-data split: stack the positive and
    # negative parts so the magnitude of each interannual deviation is preserved
    # as a non-negative coordinate, factorize V ~= W H, then label each cell by
    # its dominant component (argmax over the W coefficient row).
    pos = np.clip(features_temporal, 0.0, None)
    neg = np.clip(-features_temporal, 0.0, None)
    x = np.concatenate([pos, neg], axis=1).astype(np.float64)
    model = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=1000)
    w = model.fit_transform(x)
    return remap_labels(np.argmax(w, axis=1).astype(np.int32))


def redcap_labels(features_temporal, valid_mask, k):
    conn = raw.build_queen_connectivity(valid_mask)
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage="ward",
        connectivity=conn,
        compute_full_tree=True,
    )
    agg.fit(features_temporal)
    linkage_mat = raw._sklearn_to_scipy_linkage(agg.children_, agg.distances_, conn.shape[0])
    return remap_labels(fcluster(linkage_mat, t=k, criterion="maxclust") - 1)


def fit_labels(method, k, features_full, features_temporal, valid_mask):
    if method == "kmeans":
        return kmeans_labels(features_full, k)
    if method == "som":
        return som_labels(features_temporal, k)
    if method == "redcap":
        return redcap_labels(features_temporal, valid_mask, k)
    if method == "nmf":
        return nmf_labels(features_temporal, k)
    raise ValueError(method)


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

    domain = raw.domain_mean_tau(metric_map, lat)
    return metric_map, metric_cl, sizes, domain


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
    raw._draw_borders(ax, xlim, ylim, *raw._get_border_geoms())
    return lon_plot


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
        cmap=raw._TAU_CMAP,
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


def render_panel(results, methods, lat, lon, out_png,
                 suptitle="CONUS raw JJA-mean Tmax clusters: ACE2 vs ERA5 skill by cluster"):
    ncol = len(methods)
    tau_vlim = TAU_PLOT_VLIM
    z_scale = finite_vlim(*(results[m]["z_map"] for m in methods), floor=1.0)
    for method in methods:
        results[method]["z_plot_map"], _ = normalized_z_for_plot(
            results[method]["z_map"], scale=z_scale
        )
        results[method]["z_plot_cl"] = np.clip(
            results[method]["z_cl"] / z_scale, -1.0, 1.0
        ).astype(np.float32)
        results[method]["z_plot_domain"] = raw.domain_mean_tau(
            results[method]["z_plot_map"], lat
        )

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
            tau_vlim,
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

    # two equal-width horizontal colorbars, centered with a small gap
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


def main():
    import argparse
    global MODK_K

    p = argparse.ArgumentParser()
    p.add_argument("--kmeans-k", type=int, default=DEFAULT_KMEANS)
    p.add_argument("--som-k", type=int, default=DEFAULT_SOM)
    p.add_argument("--redcap-k", type=int, default=DEFAULT_REDCAP)
    p.add_argument("--nmf-k", type=int, default=DEFAULT_NMF)
    p.add_argument("--metric-k", type=int, default=DEFAULT_MODK_K,
                   help="Top-k truncation value for modified-Kendall z")
    p.add_argument("--recompute-raw", action="store_true")
    args = p.parse_args()
    MODK_K = args.metric_k

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modified-Kendall top-k truncation: k={MODK_K}", flush=True)
    pred, obs, lat, lon = load_conus_raw(recompute_raw=args.recompute_raw)
    features_full, features_temporal, valid_mask = raw.build_features(pred, lat, lon)
    n_lat, n_lon = len(lat), len(lon)
    print(
        f"CONUS raw fields: pred/obs={pred.shape}  features={features_full.shape}",
        flush=True,
    )

    specs = {
        "kmeans": ("K-means", args.kmeans_k),
        "som": ("SOM", args.som_k),
        "redcap": ("REDCAP", args.redcap_k),
        "nmf": ("NMF", args.nmf_k),
    }
    results = {}
    summary = {"metric": f"modified-Kendall z uses truncation k={MODK_K}", "methods": {}}
    for method, (label, k) in specs.items():
        print(f"{label}: fitting K={k}", flush=True)
        labels = fit_labels(method, k, features_full, features_temporal, valid_mask)
        n_clusters = int(labels.max()) + 1
        labels_2d = raw._expand_labels(labels, valid_mask, n_lat, n_lon)
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
            "requested_k": int(k),
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

    methods = list(specs.keys())
    tag = "_".join(f"{m}{results[m]['n_clusters']}" for m in methods)
    metric_suffix = "" if MODK_K == DEFAULT_MODK_K else f"_mk{MODK_K}"
    out_png = OUT_DIR / f"raw_cluster_metric_panel_{tag}{metric_suffix}.png"
    render_panel(results, methods, lat, lon, out_png)
    z_scale = finite_vlim(*(results[m]["z_map"] for m in methods), floor=1.0)
    for method in methods:
        summary["methods"][method]["kendall_tau_plot_range"] = [-TAU_PLOT_VLIM, TAU_PLOT_VLIM]
        summary["methods"][method]["modkendall_plot_normalization_scale"] = round(float(z_scale), 4)
        summary["methods"][method]["domain_avg_normalized_modkendall_z"] = round(
            float(results[method]["z_plot_domain"]), 4
        )
    out_json = OUT_DIR / f"raw_cluster_metric_panel_summary{metric_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
