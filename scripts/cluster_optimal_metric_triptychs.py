#!/usr/bin/env python3
"""Triptychs linking optimal-K selection to raw-extreme cluster skill metrics.

For K-means, SOM, and REDCAP, each output figure has:

  1. the existing within/between optimal-cluster similarity curve,
  2. the cluster map scored with Kendall tau at that selected K,
  3. the same cluster map scored with modified-Kendall z at that selected K.

The clustering labels are fit once per method at the optimal K reported by
optimal_clusters_{kmeans,som,redcap}.json. Kendall tau and modified-Kendall z are
then evaluated on those same labels so the two metric panels are directly
comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cluster_skill_analysis_sliding7d as csa  # noqa: E402
from mod_kendall_metric import DEFAULT_K as DEFAULT_MODK_K, normalized_z_for_plot  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_sliding7d_conus"
TRIPTYCH_DIR = OUT_DIR / "optimal_metric_triptychs"
FREQ_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
FIELD_LABEL = "CONUS JJA raw TMP2m extreme frequency"
SUMMARY_NAME = "optimal_metric_triptych_raw_extremes_summary.json"
MODK_K = DEFAULT_MODK_K

METHODS = {
    "kmeans": {
        "label": "K-means",
        "json": OUT_DIR / "optimal_clusters_kmeans.json",
        "out": TRIPTYCH_DIR / "optimal_metric_triptych_raw_extremes_kmeans.png",
    },
    "som": {
        "label": "SOM",
        "json": OUT_DIR / "optimal_clusters_som.json",
        "out": TRIPTYCH_DIR / "optimal_metric_triptych_raw_extremes_som.png",
    },
    "redcap": {
        "label": "REDCAP",
        "json": OUT_DIR / "optimal_clusters_redcap.json",
        "out": TRIPTYCH_DIR / "optimal_metric_triptych_raw_extremes_redcap.png",
    },
}
SELECTED_K = {
    "kmeans": 5,
    "som": 8,
    "redcap": 5,
}

BLUE = "#1f5fd0"
RED = "#e8202a"


def load_conus_frequency():
    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values[csa.CONUS_LAT_SLICE]
        lon = ds["lon"].values[csa.CONUS_LON_SLICE]
        pred = ds["ace2_freq"].values[:, csa.CONUS_LAT_SLICE, csa.CONUS_LON_SLICE].astype(np.float32)
        obs = ds["era5_freq"].values[:, csa.CONUS_LAT_SLICE, csa.CONUS_LON_SLICE].astype(np.float32)
    return pred, obs, lat, lon


def optimal_k(summary):
    k = summary.get("optimal_k")
    if k is None:
        raise KeyError("optimal_k missing from optimal-cluster summary")
    return int(k)


def kmeans_labels(features_full, k):
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=5,
        batch_size=min(4096, features_full.shape[0]),
    )
    return km.fit_predict(features_full)


def som_labels(features_temporal, k):
    som = MiniSom(
        k, 1, features_temporal.shape[1],
        sigma=max(1.0, k / 4.0),
        learning_rate=0.5,
        random_seed=42,
    )
    som.train_random(features_temporal, num_iteration=2000, verbose=False)
    return np.array([som.winner(x)[0] for x in features_temporal], dtype=np.int32)


def redcap_labels(features_temporal, valid_mask, k):
    conn = csa.build_queen_connectivity(valid_mask)
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage="ward",
        connectivity=conn,
        compute_full_tree=True,
    )
    agg.fit(features_temporal)
    linkage_mat = csa._sklearn_to_scipy_linkage(agg.children_, agg.distances_, conn.shape[0])
    return fcluster(linkage_mat, t=k, criterion="maxclust") - 1


def fit_labels(method, k, features_full, features_temporal, valid_mask):
    if method == "kmeans":
        return kmeans_labels(features_full, k)
    if method == "som":
        return som_labels(features_temporal, k)
    if method == "redcap":
        return redcap_labels(features_temporal, valid_mask, k)
    raise ValueError(f"Unknown method: {method}")


def eval_metric(pred, obs, labels, valid_mask, lat, n_clusters, metric):
    if metric == "tau":
        csa.METRIC = "tau"
        csa.METRIC_SYM = "tau"
        csa.METRIC_NAME = "Kendall tau"
    elif metric == "modkendall":
        csa.METRIC = "modkendall"
        csa.METRIC_K = MODK_K
        csa.METRIC_SYM = "z"
        csa.METRIC_NAME = f"modified-Kendall z (k={MODK_K})"
    else:
        raise ValueError(metric)
    metric_map, _bss_map, metric_cl, _bss_cl, sizes = csa.eval_cluster_skill(
        pred, obs, labels, valid_mask, lat, n_clusters,
    )
    domain = csa.domain_mean_tau(metric_map, lat)
    return metric_map, metric_cl, sizes, domain


def lon_for_plot(lon):
    return lon - 360.0 if float(np.nanmean(lon)) > 180.0 else lon


def draw_borders(ax, xlim, ylim):
    csa._draw_borders(ax, xlim, ylim, *csa._get_border_geoms())


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
    draw_borders(ax, xlim, ylim)
    return lon_plot, xlim, ylim


def annotate_cluster_values(ax, labels_2d, values, lat, lon, fontsize=7):
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
            fontsize=fontsize,
            weight="bold",
            color="black",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )


def draw_metric_map(ax, cax, fig, metric_map, labels_2d, metric_cl, domain, lat, lon,
                    title, label, vlim):
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
        cmap=csa._TAU_CMAP,
        vmin=-vlim,
        vmax=vlim,
        zorder=1,
    )
    setup_map_axes(ax, lat, lon)
    annotate_cluster_values(ax, labels_2d, metric_cl, lat, lon)
    ax.set_title(title, fontsize=10, pad=4)
    ax.text(
        0.02, 0.03,
        f"domain avg {label} = {domain:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="0.25"),
        zorder=7,
    )
    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_anchor("N")


def draw_similarity(ax, summary, method_label):
    k_vals = np.asarray(summary["k_values"], dtype=int)
    within = np.asarray(summary["within_mean"], dtype=float)
    within_sd = np.nan_to_num(np.asarray(summary["within_std"], dtype=float))
    between = np.asarray(summary["between_mean"], dtype=float)
    between_sd = np.nan_to_num(np.asarray(summary["between_std"], dtype=float))

    ax.fill_between(k_vals, within - within_sd, within + within_sd, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(k_vals, between - between_sd, between + between_sd, color=RED, alpha=0.18, lw=0)
    ax.plot(k_vals, within, "-", color=BLUE, lw=2.2, label="Within clusters")
    ax.plot(k_vals, between, "-", color=RED, lw=2.2, label="Between clusters")
    ax.set_title("(A) Cluster similarity", fontsize=10, pad=4)
    ax.set_xlabel("Number of clusters", fontsize=9)
    ax.set_ylabel("Pattern correlation", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.text(0.03, 0.96, method_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, weight="bold")
    ax.set_box_aspect(0.58)
    ax.set_anchor("N")


def finite_vlim(*arrays, floor=0.05, cap=None):
    vals = []
    for arr in arrays:
        good = np.asarray(arr)[np.isfinite(arr)]
        if good.size:
            vals.append(np.abs(good))
    if not vals:
        return floor
    v = max(float(np.nanpercentile(np.concatenate(vals), 98)), floor)
    if cap is not None:
        v = min(v, cap)
    return v


def render_triptych(method, result, tau_vlim, z_vlim):
    label = METHODS[method]["label"]
    k = result["k"]
    fig = plt.figure(figsize=(19.2, 5.7), facecolor="white")
    gs = fig.add_gridspec(
        2, 3,
        left=0.04,
        right=0.985,
        bottom=0.15,
        top=0.84,
        wspace=0.12,
        hspace=0.18,
        height_ratios=[1.0, 0.08],
        width_ratios=[1.12, 1.52, 1.52],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    blank = fig.add_subplot(gs[1, 0])
    blank.axis("off")
    tau_cax = fig.add_subplot(gs[1, 1])
    z_cax = fig.add_subplot(gs[1, 2])
    draw_similarity(axes[0], result["summary"], label)
    draw_metric_map(
        axes[1], tau_cax, fig, result["tau_map"], result["labels_2d"], result["tau_cl"],
        result["tau_domain"], result["lat"], result["lon"],
        "(B) Kendall tau at selected K", "tau", tau_vlim,
    )
    draw_metric_map(
        axes[2], z_cax, fig, result["z_plot_map"], result["labels_2d"], result["z_plot_cl"],
        result["z_plot_domain"], result["lat"], result["lon"],
        f"(C) Normalized modified Kendall z (top-k={MODK_K}) at selected K",
        "normalized z", z_vlim,
    )
    fig.suptitle(
        f"{label}: cluster skill maps  |  K={k}  |  {FIELD_LABEL}",
        fontsize=13,
        y=0.93,
    )
    out = METHODS[method]["out"]
    if MODK_K != DEFAULT_MODK_K:
        out = out.with_name(f"{out.stem}_mk{MODK_K}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    import argparse
    global MODK_K

    p = argparse.ArgumentParser()
    p.add_argument("--metric-k", type=int, default=DEFAULT_MODK_K,
                   help="Top-k truncation value for the modified-Kendall z panel")
    args = p.parse_args()
    MODK_K = args.metric_k

    if not FREQ_NC.exists():
        raise FileNotFoundError(FREQ_NC)
    TRIPTYCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modified-Kendall top-k truncation: k={MODK_K}", flush=True)

    pred, obs, lat, lon = load_conus_frequency()
    features_full, features_temporal, valid_mask = csa.build_features(pred, lat, lon)
    n_lat, n_lon = len(lat), len(lon)
    print(f"CONUS features: full={features_full.shape} temporal={features_temporal.shape}", flush=True)

    results = {}
    summary_out = {}
    for method, meta in METHODS.items():
        if not meta["json"].exists():
            raise FileNotFoundError(meta["json"])
        summary = json.loads(meta["json"].read_text())
        k = SELECTED_K[method]
        print(f"{meta['label']}: fitting requested K={k}", flush=True)
        labels = fit_labels(method, k, features_full, features_temporal, valid_mask)
        n_clusters = int(labels.max()) + 1
        labels_2d = csa._expand_labels(labels, valid_mask, n_lat, n_lon)

        tau_map, tau_cl, sizes, tau_domain = eval_metric(
            pred, obs, labels, valid_mask, lat, n_clusters, "tau",
        )
        z_map, z_cl, _sizes_z, z_domain = eval_metric(
            pred, obs, labels, valid_mask, lat, n_clusters, "modkendall",
        )
        results[method] = {
            "summary": summary,
            "k": k,
            "n_clusters": n_clusters,
            "labels_2d": labels_2d,
            "tau_map": tau_map,
            "tau_cl": tau_cl,
            "tau_domain": tau_domain,
            "z_map": z_map,
            "z_cl": z_cl,
            "z_domain": z_domain,
            "sizes": sizes,
            "lat": lat,
            "lon": lon,
        }
        summary_out[method] = {
            "selected_k": k,
            "optimal_k_from_similarity_json": summary.get("optimal_k"),
            "n_clusters_rendered": n_clusters,
            "modified_kendall_top_k": int(MODK_K),
            "domain_avg_tau": round(float(tau_domain), 4),
            "domain_avg_modkendall_z": round(float(z_domain), 4),
            "cluster_sizes": [int(x) for x in sizes],
        }
        print(
            f"  {meta['label']}: tau={tau_domain:.3f}  z={z_domain:.3f}  "
            f"clusters={n_clusters}",
            flush=True,
        )

    tau_vlim = 1.0
    z_scale = finite_vlim(*(r["z_map"] for r in results.values()), floor=1.0)
    for result in results.values():
        result["z_plot_map"], _ = normalized_z_for_plot(result["z_map"], scale=z_scale)
        result["z_plot_cl"] = np.clip(result["z_cl"] / z_scale, -1.0, 1.0).astype(np.float32)
        result["z_plot_domain"] = csa.domain_mean_tau(result["z_plot_map"], result["lat"])
    for method in summary_out:
        summary_out[method]["modkendall_plot_normalization_scale"] = round(float(z_scale), 4)
        summary_out[method]["domain_avg_normalized_modkendall_z"] = round(
            float(results[method]["z_plot_domain"]), 4
        )
    print(f"Common color limits: tau +/-{tau_vlim:.3f}, normalized z +/-1.000  "
          f"(raw z scale={z_scale:.3f})", flush=True)

    for method, result in results.items():
        render_triptych(method, result, tau_vlim, 1.0)

    summary_path = TRIPTYCH_DIR / SUMMARY_NAME
    if MODK_K != DEFAULT_MODK_K:
        summary_path = summary_path.with_name(f"{summary_path.stem}_mk{MODK_K}{summary_path.suffix}")
    summary_path.write_text(json.dumps(summary_out, indent=2))
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
