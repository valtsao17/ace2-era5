#!/usr/bin/env python3
"""Aggregate compact per-year AMIP-TNA evaluations.

The evaluator treats the year as the independent unit: the 25 lag members are
paired cases within each year.  This script therefore averages the annual
regional effects and reports across-year uncertainty without needing raw
autoregressive prediction files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":", 1))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True, type=Path)
    p.add_argument("--years", default="1980:2022")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    files = []
    missing = []
    for year in parse_years(args.years):
        path = args.evaluation_dir / f"amip_global_minus_tna_{year}.nc"
        if path.exists():
            files.append((year, path))
        else:
            missing.append(year)
    if not files:
        raise FileNotFoundError("no per-year AMIP-TNA evaluation files found")

    years, effects, n_members = [], [], []
    maps_high, maps_low, maps_diff = [], [], []
    lat = lon = None
    for year, path in files:
        with xr.open_dataset(path) as ds:
            years.append(year)
            effects.append(float(ds["regional_effect_by_year"].values[0]))
            n_members.append(int(ds["n_paired_members"].values[0]))
            maps_high.append(np.asarray(ds["high_hhe_frequency"], dtype=np.float32))
            maps_low.append(np.asarray(ds["low_hhe_frequency"], dtype=np.float32))
            maps_diff.append(np.asarray(ds["high_minus_low_hhe_frequency"], dtype=np.float32))
            if lat is None:
                lat = np.asarray(ds["lat"], dtype=float)
                lon = np.asarray(ds["lon"], dtype=float)

    annual = np.asarray(effects, dtype=np.float64)
    t = stats.ttest_1samp(annual, 0.0) if len(annual) >= 2 else None
    rng = np.random.default_rng(20260717)
    if len(annual) >= 2:
        boot = rng.choice(annual, size=(10000, len(annual)), replace=True).mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
        signs = rng.choice((-1.0, 1.0), size=(10000, len(annual)))
        observed = abs(annual.mean())
        flip_p = (np.count_nonzero(np.abs((signs * annual).mean(axis=1)) >= observed) + 1) / 10001
        t_stat, t_p = float(t.statistic), float(t.pvalue)
    else:
        ci = np.array([np.nan, np.nan])
        flip_p = t_stat = t_p = np.nan

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        {
            "high_hhe_frequency": (("lat", "lon"), np.nanmean(maps_high, axis=0)),
            "low_hhe_frequency": (("lat", "lon"), np.nanmean(maps_low, axis=0)),
            "high_minus_low_hhe_frequency": (("lat", "lon"), np.nanmean(maps_diff, axis=0)),
            "regional_effect_by_year": (("year",), annual.astype(np.float32)),
            "n_paired_members": (("year",), np.asarray(n_members, dtype=np.int16)),
        },
        coords={"year": years, "lat": lat, "lon": lon},
        attrs={
            "contrast": "Observed global SST/SIC minus observed TNA SST/SIC plus climatology elsewhere",
            "inference_unit": "independent base year; lag members averaged within year",
            "missing_years": json.dumps(missing),
        },
    )
    out.to_netcdf(args.out)
    summary = {
        "years": years,
        "missing_years": missing,
        "n_years": len(years),
        "mean_regional_contrast_percentage_points": 100 * float(annual.mean()),
        "median_regional_contrast_percentage_points": 100 * float(np.median(annual)),
        "bootstrap_95pct_ci_percentage_points": [100 * float(x) for x in ci],
        "paired_year_t_statistic": None if not np.isfinite(t_stat) else t_stat,
        "paired_year_t_p_value": None if not np.isfinite(t_p) else t_p,
        "year_block_sign_flip_p_value": None if not np.isfinite(flip_p) else float(flip_p),
        "positive_years": int(np.count_nonzero(annual > 0)),
        "output": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
