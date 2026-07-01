#!/usr/bin/env python3
"""Choose a modified-Kendall truncation k for the 1980-2016 year ranking.

This script makes two diagnostics:

1. A score scatterplot, similar in spirit to Zheng and Lo's Figure 1, with the
   top-ranked years under the selected k highlighted.
2. A sweep of modified-Kendall z over candidate k values, mirroring the logic
   in the original seqn_agg_func.R helper.

The k convention follows the paper and the R formulas: ranks are truncated as
min(rank, k), so ranks 1, ..., k - 1 are individually ordered and ranks k, ...,
n are tied together. For n = 37, valid candidate k values are 2 through 37.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

mpl_cache = Path(tempfile.gettempdir()) / "ace2_matplotlib_cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mod_kendall_metric import (  # noqa: E402
    modified_kendall_from_ranks,
    rank_scores,
)

DEFAULT_FREQ_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/lag_may/modkendall_k_diagnostic"


def _parse_bounds(text: Optional[str], default: Optional[Tuple[float, float]] = None):
    if text is None:
        return default
    lo, hi = text.split(",", 1)
    return float(lo), float(hi)


def _years_from_coord(coord, n: int) -> np.ndarray:
    if coord is None:
        return np.arange(1, n + 1)
    vals = np.asarray(coord)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[Y]").astype(int) + 1970
    return vals.astype(int) if np.issubdtype(vals.dtype, np.integer) else vals


def load_csv_series(path: Path, pred_col: str, obs_col: str, year_col: Optional[str]):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in (pred_col, obs_col) if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    years = (
        np.asarray([row[year_col] for row in rows])
        if year_col and year_col in rows[0]
        else np.arange(1, len(rows) + 1)
    )
    pred = np.asarray([float(row[pred_col]) for row in rows], dtype=float)
    obs = np.asarray([float(row[obs_col]) for row in rows], dtype=float)
    return years, pred, obs, f"{path.name}: {pred_col} vs {obs_col}"


def _subset_lon(da, lon_name: str, bounds: Optional[Tuple[float, float]]):
    if bounds is None or lon_name not in da.coords:
        return da
    lo, hi = bounds
    lon = da[lon_name]
    if float(lon.max()) <= 180.0 and lo > 180.0:
        lo -= 360.0
        hi -= 360.0
    if lo <= hi:
        return da.sel({lon_name: slice(lo, hi)})
    return da.where((lon >= lo) | (lon <= hi), drop=True)


def _spatial_mean(da):
    spatial_dims = [d for d in da.dims if d not in ("year", "time")]
    if not spatial_dims:
        return da
    lat_name = next((d for d in da.dims if d.lower() in ("lat", "latitude")), None)
    if lat_name is None:
        return da.mean(dim=spatial_dims, skipna=True)
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=spatial_dims, skipna=True)


def load_nc_series(
    path: Path,
    pred_var: str,
    obs_var: str,
    domain: str,
    lat_bounds: Optional[Tuple[float, float]],
    lon_bounds: Optional[Tuple[float, float]],
):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required for --freq-nc input") from exc

    if domain == "conus":
        lat_bounds = lat_bounds or (25.0, 53.0)
        lon_bounds = lon_bounds or (235.0, 295.0)

    with xr.open_dataset(path) as ds:
        if pred_var not in ds or obs_var not in ds:
            raise ValueError(f"{path} must contain variables {pred_var!r} and {obs_var!r}")
        pred_da = ds[pred_var]
        obs_da = ds[obs_var]
        lat_name = next((d for d in pred_da.dims if d.lower() in ("lat", "latitude")), None)
        lon_name = next((d for d in pred_da.dims if d.lower() in ("lon", "longitude")), None)
        if lat_bounds and lat_name:
            pred_da = pred_da.sel({lat_name: slice(*lat_bounds)})
            obs_da = obs_da.sel({lat_name: slice(*lat_bounds)})
        if lon_name:
            pred_da = _subset_lon(pred_da, lon_name, lon_bounds)
            obs_da = _subset_lon(obs_da, lon_name, lon_bounds)
        pred_s = _spatial_mean(pred_da)
        obs_s = _spatial_mean(obs_da)
        dim = "year" if "year" in pred_s.dims else ("time" if "time" in pred_s.dims else pred_s.dims[0])
        years = _years_from_coord(pred_s.coords.get(dim), pred_s.sizes[dim])
        pred = np.asarray(pred_s.values, dtype=float).ravel()
        obs = np.asarray(obs_s.values, dtype=float).ravel()

    label = f"{path.name}: {pred_var} vs {obs_var}"
    if domain == "conus":
        label += " (CONUS weighted mean)"
    elif lat_bounds or lon_bounds:
        label += " (bounded weighted mean)"
    else:
        label += " (global weighted mean)"
    return years, pred, obs, label


def sweep_k(pred, obs, k_min: int, k_max: int, tie_method: str):
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred = np.asarray(pred[ok], dtype=float)
    obs = np.asarray(obs[ok], dtype=float)
    if pred.size < 4:
        raise ValueError("need at least 4 finite paired values")
    if pred.std() < 1e-12 or obs.std() < 1e-12:
        raise ValueError("predicted and observed series must both vary")

    rank_pred = rank_scores(pred, descending=True, tie_method=tie_method)
    rank_obs = rank_scores(obs, descending=True, tie_method=tie_method)
    n = pred.size
    k_min = max(2, int(k_min))
    k_max = min(n, int(k_max))
    if k_min > k_max:
        raise ValueError(f"empty k range after clipping to n={n}: {k_min}>{k_max}")
    rows = []
    for k in range(k_min, k_max + 1):
        res = modified_kendall_from_ranks(rank_pred, rank_obs, k)
        rows.append(
            {
                "k": k,
                "top_years_individually_ordered": k - 1,
                "z": res.z_statistic,
                "agreement_rate": res.agreement_rate,
                "null_mean": res.null_mean,
                "null_variance": res.null_variance,
                "agreement_count": res.agreement_count,
            }
        )
    return rows, rank_pred, rank_obs, ok


def choose_k(rows):
    z = np.asarray([r["z"] for r in rows], dtype=float)
    k = np.asarray([r["k"] for r in rows], dtype=int)
    finite = np.isfinite(z)
    if not finite.any():
        raise ValueError("all modified-Kendall z values are NaN")
    peak_idx = int(np.nanargmax(z))
    k_peak = int(k[peak_idx])
    z_peak = float(z[peak_idx])
    if z_peak > 0:
        near = np.where(z >= 0.95 * z_peak)[0]
        k_plateau = int(k[near[0]])
    else:
        k_plateau = k_peak
    return k_peak, z_peak, k_plateau


def write_sweep_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_points_csv(path: Path, years, pred, obs, rank_pred, rank_obs, selected_k):
    top_pred = rank_pred < selected_k
    top_obs = rank_obs < selected_k
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "pred",
                "obs",
                "pred_rank",
                "obs_rank",
                "in_pred_top",
                "in_obs_top",
                "in_both_top",
            ],
        )
        writer.writeheader()
        for vals in zip(years, pred, obs, rank_pred, rank_obs, top_pred, top_obs):
            year, p, o, rp, ro, itp, ito = vals
            writer.writerow(
                {
                    "year": year,
                    "pred": p,
                    "obs": o,
                    "pred_rank": rp,
                    "obs_rank": ro,
                    "in_pred_top": bool(itp),
                    "in_obs_top": bool(ito),
                    "in_both_top": bool(itp and ito),
                }
            )


def plot_diagnostic(
    path: Path,
    years,
    pred,
    obs,
    rank_pred,
    rank_obs,
    rows,
    selected_k: int,
    plateau_k: int,
    label: str,
    pred_label: str,
    obs_label: str,
):
    k = np.asarray([r["k"] for r in rows])
    z = np.asarray([r["z"] for r in rows], dtype=float)
    top_pred = rank_pred < selected_k
    top_obs = rank_obs < selected_k
    both = top_pred & top_obs
    one = top_pred ^ top_obs

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.scatter(pred[~(both | one)], obs[~(both | one)], s=42, color="#4C78A8", alpha=0.75,
               label="lower in both")
    ax.scatter(pred[one], obs[one], s=58, color="#F58518", alpha=0.9, label="top in one ranking")
    ax.scatter(pred[both], obs[both], s=72, color="#D62728", alpha=0.95, label="top in both")
    annotate = both | one
    for year, x, y in zip(years[annotate], pred[annotate], obs[annotate]):
        ax.annotate(str(year), (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color="0.25")
    if top_pred.any():
        ax.axvline(float(np.nanmin(pred[top_pred])), color="#D62728", linestyle="--", linewidth=1)
    if top_obs.any():
        ax.axhline(float(np.nanmin(obs[top_obs])), color="#D62728", linestyle="--", linewidth=1)
    ax.set_xlabel(pred_label)
    ax.set_ylabel(obs_label)
    ax.set_title(f"Top-year scatter at k={selected_k}  (top {selected_k - 1} distinct ranks)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(k, z, color="#1F77B4", linewidth=1.6)
    ax.scatter(k, z, color="#1F77B4", s=30)
    ax.axhline(0.0, color="0.55", linewidth=1)
    ax.axhline(1.96, color="#2CA02C", linestyle=":", linewidth=1.2, label="z=1.96")
    ax.axvline(selected_k, color="#D62728", linestyle="--", linewidth=1.2,
               label=f"peak k={selected_k}")
    if plateau_k != selected_k:
        ax.axvline(plateau_k, color="#9467BD", linestyle="--", linewidth=1.2,
                   label=f"95% peak k={plateau_k}")
    ax.set_xlabel("candidate truncation k")
    ax.set_ylabel("modified-Kendall z")
    ax.set_title("k sweep over the 37-year ranking")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", type=Path, help="CSV with one row per year")
    src.add_argument("--freq-nc", type=Path, default=DEFAULT_FREQ_NC,
                     help="NetCDF with annual pred/obs arrays")
    parser.add_argument("--pred-col", default="ace2", help="CSV predicted column")
    parser.add_argument("--obs-col", default="era5", help="CSV observed column")
    parser.add_argument("--year-col", default="year", help="CSV year column")
    parser.add_argument("--pred-var", default="ace2_freq", help="NetCDF predicted variable")
    parser.add_argument("--obs-var", default="era5_freq", help="NetCDF observed variable")
    parser.add_argument("--domain", choices=("global", "conus"), default="global")
    parser.add_argument("--lat-bounds", help="lat bounds as min,max")
    parser.add_argument("--lon-bounds", help="lon bounds as min,max, using dataset convention")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=37)
    parser.add_argument("--tie-method", choices=("average", "ordinal"), default="average",
                        help="average matches R rank(); ordinal forces permutation ranks")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        years, pred, obs, label = load_csv_series(args.csv, args.pred_col, args.obs_col, args.year_col)
        pred_label = args.pred_col
        obs_label = args.obs_col
    else:
        if not args.freq_nc.exists():
            raise FileNotFoundError(
                f"{args.freq_nc} not found. Pass --csv with columns "
                f"{args.pred_col!r}/{args.obs_col!r}, or generate the seasonal NetCDF first."
            )
        lat_bounds = _parse_bounds(args.lat_bounds)
        lon_bounds = _parse_bounds(args.lon_bounds)
        years, pred, obs, label = load_nc_series(
            args.freq_nc, args.pred_var, args.obs_var, args.domain, lat_bounds, lon_bounds
        )
        pred_label = args.pred_var
        obs_label = args.obs_var

    rows, rank_pred, rank_obs, ok = sweep_k(pred, obs, args.k_min, args.k_max, args.tie_method)
    years = np.asarray(years)[ok]
    pred = np.asarray(pred, dtype=float)[ok]
    obs = np.asarray(obs, dtype=float)[ok]
    k_peak, z_peak, k_plateau = choose_k(rows)

    stem = "modkendall_k_diagnostic"
    fig_path = args.out_dir / f"{stem}.png"
    sweep_csv = args.out_dir / f"{stem}_sweep.csv"
    points_csv = args.out_dir / f"{stem}_points.csv"
    plot_diagnostic(
        fig_path, years, pred, obs, rank_pred, rank_obs, rows,
        k_peak, k_plateau, label, pred_label, obs_label
    )
    write_sweep_csv(sweep_csv, rows)
    write_points_csv(points_csv, years, pred, obs, rank_pred, rank_obs, k_peak)

    msg = (
        f"peak k={k_peak} (z={z_peak:.3f}); "
        f"top years individually ordered = {k_peak - 1}; "
        f"smallest k within 95% of peak = {k_plateau}"
    )
    if z_peak < 1.96:
        msg += "; peak is below the usual one-sided 5% z threshold"
    print(msg)
    print(f"wrote {fig_path}")
    print(f"wrote {sweep_csv}")
    print(f"wrote {points_csv}")


if __name__ == "__main__":
    main()
