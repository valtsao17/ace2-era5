#!/usr/bin/env python3
"""One-panel HHE-frequency cluster skill comparison (CONUS, non-raw).

Twin of raw_cluster_metric_panel.py, but clusters CONUS grid cells by their
37-year ACE2 JJA *HHE seasonal-frequency* trajectories (the "as we did before"
sliding-7d data from seasonal_jja_skill.py) instead of the raw JJA-mean Tmax.

It reuses the exact clustering, scoring and plotting machinery from
raw_cluster_metric_panel, only swapping the underlying data source module
(cluster_skill_analysis_sliding7d) and the input arrays
(outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc).

  top row    : Kendall tau
  bottom row : modified-Kendall z
  methods    : K-means, SOM (Nx1), REDCAP, NMF

Output -> outputs/lag_may/cluster_analysis_sliding7d/conus_metric_panel/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cluster_skill_analysis_sliding7d as src  # noqa: E402
import raw_cluster_metric_panel as panel  # noqa: E402

# Rebind the panel's data-source module so every raw.* helper it calls
# (build_features, build_queen_connectivity, domain_mean_tau, _expand_labels,
# _draw_borders, _get_border_geoms, _TAU_CMAP, _sklearn_to_scipy_linkage, ...)
# resolves against the HHE-frequency module instead of the raw-Tmax one.
panel.raw = src

FREQ_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
OUT_DIR = PROJECT_ROOT / "outputs/lag_may/cluster_analysis_sliding7d/conus_metric_panel"

DEFAULT_KMEANS = 5
DEFAULT_SOM = 8
DEFAULT_REDCAP = 5
DEFAULT_NMF = 5


def load_conus_hhe():
    """CONUS ACE2/ERA5 JJA HHE seasonal-frequency fields (37, lat, lon)."""
    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        pred = ds["ace2_freq"].values.astype(np.float32)
        obs = ds["era5_freq"].values.astype(np.float32)
    la, lo = src.CONUS_LAT_SLICE, src.CONUS_LON_SLICE
    return pred[:, la, lo], obs[:, la, lo], lat[la], lon[lo]


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--kmeans-k", type=int, default=DEFAULT_KMEANS)
    p.add_argument("--som-k", type=int, default=DEFAULT_SOM)
    p.add_argument("--redcap-k", type=int, default=DEFAULT_REDCAP)
    p.add_argument("--nmf-k", type=int, default=DEFAULT_NMF)
    p.add_argument("--metric-k", type=int, default=panel.DEFAULT_MODK_K,
                   help="Top-k truncation value for modified-Kendall z")
    args = p.parse_args()
    panel.MODK_K = args.metric_k

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modified-Kendall top-k truncation: k={panel.MODK_K}", flush=True)
    pred, obs, lat, lon = load_conus_hhe()
    features_full, features_temporal, valid_mask = src.build_features(pred, lat, lon)
    n_lat, n_lon = len(lat), len(lon)
    print(
        f"CONUS HHE-freq fields: pred/obs={pred.shape}  features={features_full.shape}",
        flush=True,
    )

    specs = {
        "kmeans": ("K-means", args.kmeans_k),
        "som": ("SOM", args.som_k),
        "redcap": ("REDCAP", args.redcap_k),
        "nmf": ("NMF", args.nmf_k),
    }
    results = {}
    summary = {"metric": f"modified-Kendall z uses truncation k={panel.MODK_K}", "methods": {}}
    for method, (label, k) in specs.items():
        print(f"{label}: fitting K={k}", flush=True)
        labels = panel.fit_labels(method, k, features_full, features_temporal, valid_mask)
        n_clusters = int(labels.max()) + 1
        labels_2d = src._expand_labels(labels, valid_mask, n_lat, n_lon)
        tau_map, tau_cl, sizes, tau_domain = panel.eval_cluster_metric(
            pred, obs, labels, valid_mask, lat, n_clusters, "tau",
        )
        z_map, z_cl, _sizes_z, z_domain = panel.eval_cluster_metric(
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
    metric_suffix = "" if panel.MODK_K == panel.DEFAULT_MODK_K else f"_mk{panel.MODK_K}"
    out_png = OUT_DIR / f"hhe_cluster_metric_panel_{tag}{metric_suffix}.png"
    panel.render_panel(
        results, methods, lat, lon, out_png,
        suptitle="CONUS HHE-frequency clusters: ACE2 vs ERA5 skill by cluster",
    )
    z_scale = panel.finite_vlim(*(results[m]["z_map"] for m in methods), floor=1.0)
    for method in methods:
        summary["methods"][method]["kendall_tau_plot_range"] = [
            -panel.TAU_PLOT_VLIM,
            panel.TAU_PLOT_VLIM,
        ]
        summary["methods"][method]["modkendall_plot_normalization_scale"] = round(float(z_scale), 4)
        summary["methods"][method]["domain_avg_normalized_modkendall_z"] = round(
            float(results[method]["z_plot_domain"]), 4
        )
    out_json = OUT_DIR / f"hhe_cluster_metric_panel_summary{metric_suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
