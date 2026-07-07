#!/usr/bin/env python3
"""Single combined CONUS panel of all four raw heat-extreme skill diagnostics:

    [ Kendall τ        ]   [ modified-Kendall z ]
    [ Precision        ]   [ Recall                  ]

Puts the two rank-correlation skill metrics (signed, diverging blue-white-red)
above the two season-level classification metrics (0–1, sequential) so the whole
skill story reads from one figure. All four fields already exist on disk on the
same 1° grid; this script only assembles + renders them (CONUS box).

Sources (CONUS slice):
  Kendall τ      → seasonal_jja_sliding7d/skill_jja_seasonal.nc          (kendall_tau)
  mod-Kendall z  → seasonal_jja_sliding7d_modkendall/skill_jja_seasonal.nc (kendall_tau holds z)
  precision      → seasonal_jja_sliding7d/precision_recall_jja_seasonal_event.nc (precision)
  recall         → seasonal_jja_sliding7d/precision_recall_jja_seasonal_event.nc (recall)

Precision/recall are plus-four/Agresti-Coull binomial estimates:
  precision = (TP + 2) / (TP + FP + 4)
  recall    = (TP + 2) / (TP + FN + 4)

One case is one grid-cell JJA season-year. A positive event is a high-burden
season: seasonal heat-extreme frequency above the grid-cell upper-tail
percentile threshold (75th percentile by default).

Output → outputs/lag_may/seasonal_jja_sliding7d_modkendall/skill_panel_combined.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (
    _plain_map_axes, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)
from seasonal_jja_skill import _SKILL_CMAP, cos_lat_mean, load_land_mask, domain_scores_label
from mod_kendall_metric import normalized_z_for_plot

from copy import copy

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
MODK_DIR    = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall"
SEASONAL_PR_NC = SLIDING_DIR / "precision_recall_jja_seasonal_event.nc"

# Skill colormap that draws UNDEFINED cells (NaN) in a distinct neutral grey
# instead of white. For precision this matters: a cell where the model never
# predicts a positive season has TP+FP=0 -> precision is undefined (0/0), which
# is physically different from a genuine precision of 0.
_SKILL_CMAP_NA = copy(_SKILL_CMAP)
_SKILL_CMAP_NA.set_bad("#bdbdbd")


def _stipple(ax, sig, lat, lon):
    """Black dots at cell centres where `sig` is True (p<0.05 / |z|>1.96)."""
    if sig is None or not np.any(sig):
        return
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    LON2D, LAT2D = np.meshgrid(lon_plot, lat)
    ax.scatter(LON2D[sig], LAT2D[sig], s=1.8, c="k", alpha=0.55,
               linewidths=0, zorder=6)


def _panel(ax, field, lat, lon, title, cmap, vmin, vmax, cbar_label, mean_lbl=None,
           sig=None):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap=cmap, zorder=1,
                   interpolation="nearest")
    _stipple(ax, sig, lat, lon)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=cbar_label)
    if mean_lbl is not None:
        ax.legend([Line2D([], [], linestyle="none")], [mean_lbl],
                  loc="lower left", fontsize=8, handlelength=0, handletextpad=0,
                  framealpha=1.0, borderpad=0.5).set_zorder(7)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sliding-dir", type=Path, default=SLIDING_DIR)
    p.add_argument("--modk-dir", type=Path, default=MODK_DIR)
    p.add_argument("--pr-nc", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--label", default="raw heat-extreme (90th-pct TMP2m)")
    args = p.parse_args()
    sliding_dir = args.sliding_dir
    modk_dir = args.modk_dir
    pr_nc = args.pr_nc or (sliding_dir / "precision_recall_jja_seasonal_event.nc")
    out = args.out or (modk_dir / "skill_panel_combined_raw.png")

    # Kendall τ (+ p-value for stippling)
    with xr.open_dataset(sliding_dir / "skill_jja_seasonal.nc") as ds:
        lat = ds["lat"].values[CONUS_LAT_SLICE]
        lon = ds["lon"].values[CONUS_LON_SLICE]
        tau = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        tau_p = ds["tau_p_value"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        years_label = str(ds.attrs.get("years", "1980-2016")).replace("-", "–")

    # modified-Kendall z (variable name is kendall_tau but holds z)
    with xr.open_dataset(modk_dir / "skill_jja_seasonal.nc") as ds:
        zmap = ds["kendall_tau"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        zk = int(ds["kendall_tau"].attrs.get("truncation_k", 10))

    # season-level precision / recall
    if not pr_nc.exists():
        raise FileNotFoundError(
            f"{pr_nc} not found. Run "
            "`python scripts/seasonal_precision_recall_jja.py --event-percentile 75` "
            "before regenerating the combined skill panel."
        )
    with xr.open_dataset(pr_nc) as ds:
        prec = ds["precision"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        rec  = ds["recall"].values[CONUS_LAT_SLICE, CONUS_LON_SLICE]
        event_pct = float(ds.attrs.get("event_percentile", 75.0))

    # NON-significance masks for the rank-correlation panels (stipple = p>=0.05)
    tau_ns = np.isfinite(tau) & ~(np.isfinite(tau_p) & (tau_p < 0.05))
    z_ns   = np.isfinite(zmap) & (np.abs(zmap) <= 1.96)       # |z|<=1.96 -> p>=0.05

    # symmetric, data-driven limits for tau; modified-Kendall z is normalized
    # to [-1, 1] for plotting so its color scale matches bounded tau.
    z_plot, z_scale = normalized_z_for_plot(zmap)

    land = load_land_mask(lat, lon)         # all / land / sea score breakdown
    tau_m  = cos_lat_mean(tau, lat)
    z_m    = cos_lat_mean(zmap, lat)
    zn_m   = cos_lat_mean(z_plot, lat)
    prec_m = cos_lat_mean(prec, lat)
    rec_m  = cos_lat_mean(rec, lat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    _panel(axes[0, 0], tau, lat, lon, "Kendall τ  (ACE2 vs ERA5)",
           _TAU_CMAP, -1.0, 1.0, "τ", domain_scores_label("mean τ", tau, lat, land),
           sig=tau_ns)
    _panel(axes[0, 1], z_plot, lat, lon, f"Normalized modified-Kendall z  (k={zk})",
           _TAU_CMAP, -1.0, 1.0, "normalized z",
           domain_scores_label("mean norm z", z_plot, lat, land),
           sig=z_ns)
    pr_title = f"seasonal upper-{100.0 - event_pct:.0f}% event, plus-four"
    _panel(axes[1, 0], prec, lat, lon, f"Precision  ({pr_title})",
           _SKILL_CMAP_NA, 0.0, 1.0, "precision", domain_scores_label("precision", prec, lat, land))
    _panel(axes[1, 1], rec, lat, lon, f"Recall  ({pr_title})",
           _SKILL_CMAP_NA, 0.0, 1.0, "recall", domain_scores_label("recall", rec, lat, land))
    # grey = undefined: precision where ACE2 predicted no positive seasons
    # (TP+FP=0); recall where ERA5 observed none (TP+FN=0). Only annotate when
    # such cells exist.
    _bbox = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="0.6")
    n_na_p = int(np.sum(~np.isfinite(prec)))
    if n_na_p:
        axes[1, 0].text(0.98, 0.03, f"grey = undefined (no predicted\npositive seasons): {n_na_p} cells",
                        transform=axes[1, 0].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)
    n_na_r = int(np.sum(~np.isfinite(rec)))
    if n_na_r:
        axes[1, 1].text(0.98, 0.03, f"grey = undefined (no observed\npositive seasons): {n_na_r} cells",
                        transform=axes[1, 1].transAxes, ha="right", va="bottom",
                        fontsize=7, bbox=_bbox, zorder=7)

    fig.suptitle(f"CONUS {args.label} skill — rank-correlation (top, "
                 "stipple = NOT significant: τ p≥0.05 / |z|≤1.96) vs season-level classification "
                 f"(bottom; precision/recall for >p{event_pct:.0f} seasonal-frequency years)  |  "
                 f"JJA {years_label}, seasonal, no-LOO",
                 fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    print(f"  CONUS means: τ={tau_m:.3f}  z={z_m:.3f}  norm_z={zn_m:.3f}  "
          f"z_plot_scale={z_scale:.3f}  prec={prec_m:.3f}  rec={rec_m:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
