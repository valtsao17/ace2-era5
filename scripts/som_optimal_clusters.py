#!/usr/bin/env python3
"""Optimal SOM size via Johnson (2013) statistical distinguishability.

Faithful implementation of the optimal-map-size criterion of Johnson (2013),
"How many ENSO flavors can we distinguish?" (J. Climate,
doi:10.1175/JCLI-D-12-00649.1, Section 2b).

Like Johnson, we cluster *fields* — here the 37 yearly JJA HHE-frequency fields
(samples = years), NOT grid cells. An N×1 SOM groups the 37 years into recurring
HHE-frequency spatial patterns ("flavours" of HHE summers). For each map size N:

  * assign each year to its best-matching node;
  * for every pair of node composites (i, j) run a per-grid-cell two-sample
    (Welch) t-test for difference of means — members are the (independent) YEARS
    assigned to each node — yielding one p-value per cell;
  * apply an FDR field-significance test (Benjamini-Hochberg / Wilks 2006) at
    q=0.05: the pair is DISTINGUISHABLE if >=1 local test survives FDR;
  * count the indistinguishable node pairs.

  N* = the largest N with ZERO indistinguishable node pairs (Johnson's K*).

This replaces the earlier within/between pattern-correlation *intersection*
method (Frontiers 2022), which was run on a transposed cell-regionalization and
is not Johnson's method. See johnson_distinguishability.py.

Outputs → outputs/lag_may/cluster_analysis_sliding7d_conus[/_<domain>]/
            optimal_clusters_som.png
            optimal_clusters_som.json
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from minisom import MiniSom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    SLIDING_DIR, CONUS_LAT_SLICE, CONUS_LON_SLICE, _apply_domain_mask,
)
from johnson_distinguishability import (  # noqa: E402
    count_indistinguishable_pairs, select_max_distinguishable,
    plot_distinguishability,
)


def som_year_labels(fields_z, n_nodes, seed):
    """Train an N×1 SOM on the per-cell-standardized yearly fields and return the
    best-matching-unit (node) label for each YEAR.

    fields_z : (n_years, n_cells) standardized HHE-frequency anomaly fields.
    """
    n_features = fields_z.shape[1]
    som = MiniSom(n_nodes, 1, n_features,
                  sigma=max(1.0, n_nodes / 4.0), learning_rate=0.5,
                  random_seed=seed)
    som.train_random(fields_z, num_iteration=2000, verbose=False)
    return np.array([som.winner(x)[0] for x in fields_z])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--global", dest="globaldomain", action="store_true",
                   help="Full global domain instead of CONUS (default CONUS)")
    p.add_argument("--domain", choices=["all", "land", "ocean"], default="all")
    p.add_argument("--source", choices=["ace2", "era5"], default="ace2",
                   help="Which yearly HHE-frequency fields to cluster")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=12,
                   help="Max nodes (limited by ~37 yearly samples)")
    p.add_argument("--seeds", type=int, default=5,
                   help="SOM trainings per N (stochastic init)")
    p.add_argument("--q", type=float, default=0.05,
                   help="FDR field-significance level (Johnson uses 0.05)")
    p.add_argument("--out-stem", default="optimal_clusters_som",
                   help="Output filename stem for the Johnson plot/json")
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
    var = "ace2_freq" if args.source == "ace2" else "era5_freq"
    data = ds[var].values.astype(np.float32)
    ds.close()

    if conus:
        data = data[:, CONUS_LAT_SLICE, CONUS_LON_SLICE]
        lat = lat[CONUS_LAT_SLICE]
        lon = lon[CONUS_LON_SLICE]
        print(f"CONUS box: lat {lat[0]:.1f}..{lat[-1]:.1f}  lon {lon[0]:.1f}..{lon[-1]:.1f}", flush=True)

    if args.domain != "all":
        data, _ = _apply_domain_mask(args.domain, lat, lon, data, data.copy())
        print(f"Domain: {args.domain}-only", flush=True)

    # ── Johnson field-clustering: samples = years, features = grid cells ──────
    valid = np.all(np.isfinite(data), axis=0)
    fields = data[:, valid]                      # (n_years, n_cells)
    n_years, n_cells = fields.shape
    # standardize each cell across years -> SOM groups by anomaly pattern, not by
    # the climatological hotspot magnitude (Johnson clusters anomaly fields). The
    # per-cell scaling leaves the per-cell two-sample t-statistic unchanged, so
    # the distinguishability test is identical whether run on raw or standardized.
    mu = fields.mean(0, keepdims=True)
    sd = fields.std(0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    fields_z = (fields - mu) / sd
    print(f"field-clustering: samples(years)={n_years}  features(cells)={n_cells}  "
          f"source={args.source}", flush=True)

    k_vals = list(range(args.kmin, args.kmax + 1))
    med_counts, lo_counts, hi_counts = [], [], []
    print(f"\n  N   #indist (median of {args.seeds} seeds)   [min..max]   "
          f"med #non-empty nodes", flush=True)
    for k in k_vals:
        per_seed, nodes_seed = [], []
        for seed in range(args.seeds):
            labels = som_year_labels(fields_z, k, seed=42 + seed)
            n_ind, n_pairs, n_unt = count_indistinguishable_pairs(
                fields_z, labels, q=args.q)
            per_seed.append(n_ind)
            nodes_seed.append(len(np.unique(labels)))
        per_seed = np.asarray(per_seed, float)
        med_counts.append(float(np.median(per_seed)))
        lo_counts.append(float(per_seed.min()))
        hi_counts.append(float(per_seed.max()))
        print(f"  {k:2d}   {med_counts[-1]:5.1f}                      "
              f"[{lo_counts[-1]:.0f}..{hi_counts[-1]:.0f}]    "
              f"{int(np.median(nodes_seed))}", flush=True)

    kstar = select_max_distinguishable(k_vals, med_counts)
    print(f"\nOptimal SOM N* (largest N with zero indistinguishable pairs): {kstar}",
          flush=True)

    out_png = out_dir / f"{args.out_stem}.png"
    plot_distinguishability(
        k_vals, med_counts, kstar, "SOM map size (N×1)", out_png,
        spread=(lo_counts, hi_counts),
        title="SOM optimal size — Johnson (2013) distinguishability "
              "(t-test + FDR, q=%.2f)" % args.q)
    print(f"wrote {out_png}", flush=True)

    summary = {
        "method": "Johnson (2013) maximum number of statistically distinguishable "
                  "clusters: cluster yearly HHE fields; pairwise per-cell Welch "
                  "t-test + FDR field significance; N* = largest N with zero "
                  "indistinguishable node pairs",
        "reference": "Johnson 2013, J. Climate, doi:10.1175/JCLI-D-12-00649.1, Sec 2b",
        "clustering": "SOM (N×1, MiniSom) on yearly HHE-frequency fields",
        "samples": f"{n_years} JJA years",
        "field_dim": int(n_cells),
        "field_meaning": "grid cells (the spatial field the t-test runs over)",
        "member_meaning": "years assigned to each node (independent samples)",
        "source": args.source,
        "fdr_q": args.q,
        "domain": "conus" if conus else "global",
        "subdomain": args.domain,
        "seeds": args.seeds,
        "k_range": [args.kmin, args.kmax],
        "k_values": k_vals,
        "indistinguishable_median": med_counts,
        "indistinguishable_min": lo_counts,
        "indistinguishable_max": hi_counts,
        "optimal_k": kstar,
        "selection_rule": "largest N with zero statistically indistinguishable node pairs",
    }
    out_json = out_dir / f"{args.out_stem}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
