#!/usr/bin/env python3
"""Grid-point precision/recall for JJA HHE day-level classification, no clustering.

Positive event = ERA5 day exceeds its LOO ±7-day 90th-pct threshold (the same
HHE definition used everywhere else in this project, see seasonal_jja_skill.py).
ACE2's predicted class for that day = majority vote across the 25-member
ensemble (>=50% of members exceed their own LOO threshold that day).

Pooled across all 37 years x 92 JJA days per grid cell:
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)

Reuses the load/threshold machinery from seasonal_jja_skill.py so the event
definition is identical to era5_freq/ace2_freq already in jja_seasonal_freqs.nc.
This is the heavy path: full (37, 92, lat, lon) ERA5 array and (37, 25, 92,
lat, lon) ACE2 array (~22 GB) loaded into memory, same scale as that script.

Outputs -> outputs/lag_may/seasonal_jja_sliding7d/
  precision_recall_jja_seasonal.nc
  precision_jja_seasonal_global.png
  recall_jja_seasonal_global.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seasonal_jja_skill import (
    YEARS, COMBINED_DIR,
    _load_all_era5_jja, _load_all_ace2_jja, compute_daywise_thresholds_loo,
    cos_lat_mean, _roll_to_180, _draw_coast, _SKILL_CMAP,
)

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"


def plot_pr_map(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                title: str, out_path: Path, metric_label: str) -> None:
    field_r, lon_r = _roll_to_180(field, lon)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#d0e8f0")
    LON2D, LAT2D = np.meshgrid(lon_r, lat)
    mesh = ax.pcolormesh(LON2D, LAT2D, field_r, cmap=_SKILL_CMAP, vmin=0.0, vmax=1.0,
                         shading="nearest", zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    _draw_coast(ax)

    fig.colorbar(mesh, ax=ax, shrink=0.7, label=metric_label)
    full_mean = cos_lat_mean(field, lat)
    ax.text(0.01, 0.03, f"cos-lat mean {metric_label} = {full_mean:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))

    ax.set_xticks(range(-180, 181, 60))
    ax.set_xticklabels(["180°", "120°W", "60°W", "0°", "60°E", "120°E", "180°"], fontsize=8)
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"], fontsize=8)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}", flush=True)


def main():
    with xr.open_dataset(COMBINED_DIR / f"tmax_jja_{YEARS[0]}.nc") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
    nlat, nlon = len(lat), len(lon)
    print(f"Grid: {nlat} lat x {nlon} lon (global)", flush=True)

    print("Loading all ERA5 JJA data ...", flush=True)
    era5_all = _load_all_era5_jja(nlat, nlon)
    print(f"  era5_all: {era5_all.shape}  ({era5_all.nbytes/1e6:.0f} MB)", flush=True)

    print("Loading all ACE2 JJA data ...", flush=True)
    ace2_all = _load_all_ace2_jja(nlat, nlon)
    print(f"  ace2_all: {ace2_all.shape}  ({ace2_all.nbytes/1e9:.1f} GB)", flush=True)

    print("Computing ERA5 LOO +/-7d sliding thresholds ...", flush=True)
    era5_thresh = compute_daywise_thresholds_loo(era5_all)
    print("  ERA5 done.", flush=True)

    print("Computing ACE2 LOO +/-7d sliding thresholds ...", flush=True)
    ace2_thresh = compute_daywise_thresholds_loo(ace2_all)
    print("  ACE2 done.", flush=True)

    tp = np.zeros((nlat, nlon), dtype=np.int64)
    fp = np.zeros((nlat, nlon), dtype=np.int64)
    fn = np.zeros((nlat, nlon), dtype=np.int64)
    tn = np.zeros((nlat, nlon), dtype=np.int64)

    print("Pooling day-level confusion matrix across years ...", flush=True)
    for i, yr in enumerate(YEARS):
        valid = (np.isfinite(era5_all[i]) & np.isfinite(era5_thresh[i])
                & np.isfinite(ace2_thresh[i]) & np.isfinite(ace2_all[i]).all(axis=0))  # (92,lat,lon)

        obs_ext   = era5_all[i] > era5_thresh[i]                              # (92,lat,lon)
        ace2_prob = (ace2_all[i] > ace2_thresh[i][np.newaxis]).mean(axis=0)    # (92,lat,lon)
        pred_ext  = ace2_prob >= 0.5

        tp += np.sum(valid & obs_ext & pred_ext, axis=0)
        fp += np.sum(valid & (~obs_ext) & pred_ext, axis=0)
        fn += np.sum(valid & obs_ext & (~pred_ext), axis=0)
        tn += np.sum(valid & (~obs_ext) & (~pred_ext), axis=0)
        print(f"  {yr} done  (TP+FP+FN+TN so far, cell[90,180]="
              f"{tp[90,180]+fp[90,180]+fn[90,180]+tn[90,180]})", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan).astype(np.float32)
        recall    = np.where((tp + fn) > 0, tp / (tp + fn), np.nan).astype(np.float32)

    xr.Dataset(
        {
            "precision": (("lat", "lon"), precision),
            "recall":    (("lat", "lon"), recall),
            "tp": (("lat", "lon"), tp), "fp": (("lat", "lon"), fp),
            "fn": (("lat", "lon"), fn), "tn": (("lat", "lon"), tn),
        },
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "Day-level HHE precision/recall, ERA5 obs vs ACE2 majority-vote (>=13/25 members), JJA 1980-2016"},
    ).to_netcdf(OUT_DIR / "precision_recall_jja_seasonal.nc")
    print(f"wrote: {OUT_DIR / 'precision_recall_jja_seasonal.nc'}", flush=True)

    yr_range = f"{YEARS[0]}-{YEARS[-1]}"
    plot_pr_map(precision, lat, lon,
                f"ACE2 day-level HHE precision  |  JJA {yr_range}  |  ±7d LOO, majority vote",
                OUT_DIR / "precision_jja_seasonal_global.png", "Precision")
    plot_pr_map(recall, lat, lon,
                f"ACE2 day-level HHE recall  |  JJA {yr_range}  |  ±7d LOO, majority vote",
                OUT_DIR / "recall_jja_seasonal_global.png", "Recall")
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
