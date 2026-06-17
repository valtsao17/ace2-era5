#!/usr/bin/env python3
"""Postprocess the lag ensemble combined prediction file.

Produces the same figures and metrics as the stochastic pipeline for the
2014-04-01 target date so results are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Pipeline src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.extremes import compute_target_date_extremes
from hiro_ace_pipeline.era5 import observed_target_date, monthly_baseline_threshold
from hiro_ace_pipeline.figures import targetdate_diagnostics
from hiro_ace_pipeline.io import to_celsius, subset_bbox

# Paths
COMBINED = PROJECT_ROOT / "outputs/lag_experiment/predictions/ace2_TMP2m_lag_target20140401.nc"
OUT_ROOT = PROJECT_ROOT / "outputs/lag_experiment"
CACHE_DIR = OUT_ROOT / "era5_cache"
PROCESSED_DIR = OUT_ROOT / "processed"
FIGURES_DIR = OUT_ROOT / "figures"
METRICS_DIR = OUT_ROOT / "metrics"

for d in [CACHE_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_DATE = pd.Timestamp("2014-04-01")
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BBOX = (-90.0, 90.0, 0.0, 360.0)    # global 1-degree grid
BASELINE_START = 2012
BASELINE_END = 2013
EXPECTED_MEMBERS = 25
FIGURE_FORMATS = ["png"]
MAX_MAP_MEMBERS = 6


def load_lag_prediction(path: Path, bbox: tuple) -> xr.DataArray:
    """Load the combined lag prediction file.

    The file has time=[2014-04-01] as proper datetime64 (written by
    combine_lag_predictions.py with xarray datetime encoding), so
    decode_times=True gives a usable time coordinate directly.
    """
    ds = xr.open_dataset(path)   # decode_times=True
    da = ds["TMP2m"]

    # Convert K -> C if needed
    probe = float(da.isel(sample=0, time=0).mean())
    if probe > 100:
        da = da - 273.15
        da.attrs["units"] = "degC"

    da = subset_bbox(da, bbox)

    # Daily max: with a single 00:00 time step, resample returns that value.
    return da.resample(time="1D").max().rename("predicted_daily_tmax_C")


def compute_metrics(ds: xr.Dataset, lead_days: int) -> dict:
    prob = ds["predicted_hot_extreme_probability"].values.ravel().astype(float)
    obs = ds["observed_hot_extreme"].values.ravel().astype(float)
    pred_tmax = ds["predicted_daily_tmax_C_ensmean"].values.ravel().astype(float)
    obs_tmax = ds["observed_daily_tmax_C"].values.ravel().astype(float)

    mask = np.isfinite(prob) & np.isfinite(obs) & np.isfinite(pred_tmax) & np.isfinite(obs_tmax)
    prob, obs, pred_tmax, obs_tmax = prob[mask], obs[mask], pred_tmax[mask], obs_tmax[mask]

    brier = float(np.mean((prob - obs) ** 2))
    frac_obs = float(np.mean(obs))
    mean_prob = float(np.mean(prob))
    corr = float(np.corrcoef(prob, obs)[0, 1]) if obs.std() > 0 else float("nan")
    tmax_bias = float(np.mean(pred_tmax - obs_tmax))
    tmax_rmse = float(np.sqrt(np.mean((pred_tmax - obs_tmax) ** 2)))
    brier_ref = float(np.mean((frac_obs - obs) ** 2))
    bss = 1.0 - brier / brier_ref if brier_ref > 0 else float("nan")

    return {
        "ensemble_type": "lag",
        "target_date": str(TARGET_DATE.date()),
        "lead_days": lead_days,
        "init_date": "2014-02-01 (centre)",
        "n_members": EXPECTED_MEMBERS,
        "frac_obs_extreme": round(frac_obs, 4),
        "mean_pred_prob": round(mean_prob, 4),
        "brier_score": round(brier, 4),
        "brier_skill_score": round(bss, 4),
        "spatial_corr_prob_obs": round(corr, 4),
        "pred_tmax_bias_C": round(tmax_bias, 3),
        "pred_tmax_rmse_C": round(tmax_rmse, 3),
    }


def main():
    if not COMBINED.exists():
        print(f"ERROR: combined prediction file not found: {COMBINED}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading lag prediction: {COMBINED}", flush=True)
    pred_daily = load_lag_prediction(COMBINED, BBOX)
    print(f"  pred_daily dims: {pred_daily.sizes}", flush=True)

    print("Fetching ERA5 observation at target date...", flush=True)
    obs_daily = observed_target_date(ERA5_ZARR, TARGET_DATE, BBOX, pred_daily, CACHE_DIR, force=True)

    print("Computing baseline threshold...", flush=True)
    baseline, threshold = monthly_baseline_threshold(
        ERA5_ZARR, TARGET_DATE.month, BASELINE_START, BASELINE_END,
        BBOX, pred_daily, CACHE_DIR, PROCESSED_DIR, force=True,
    )

    out_path = PROCESSED_DIR / "TMP2m_hot_extremes_lag_target20140401.nc"
    print("Computing extremes...", flush=True)
    ds = compute_target_date_extremes(
        pred_daily, obs_daily, baseline, threshold, TARGET_DATE, out_path, force=True
    )

    # Lead days = Feb 1 centre to Apr 1 = 59 days
    metrics = compute_metrics(ds, lead_days=59)
    import json
    metrics_path = METRICS_DIR / "lag_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics}", flush=True)
    print(f"wrote: {metrics_path}", flush=True)

    title = (
        "ACE2-ERA5 lag ensemble | target 2014-04-01 | "
        "centre init 2014-02-01 ±3 days (25 members, 6 h spacing)"
    )
    stem = "TMP2m_hot_extreme_lag_target20140401_init20140201center"
    images = targetdate_diagnostics(
        ds, title, FIGURES_DIR, stem, FIGURE_FORMATS, MAX_MAP_MEMBERS, force=True
    )
    print(f"wrote {len(images)} figure(s) to {FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
