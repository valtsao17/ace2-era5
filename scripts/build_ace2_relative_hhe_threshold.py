#!/usr/bin/env python3
"""Build one frozen ACE2 relative-HHE threshold for causal evaluations.

Definition
----------
An event is daily Heat Index greater than the grid-cell/calendar-day 90th
percentile.  For each JJA calendar day, the percentile pool contains that day
plus/minus seven days, all requested years, and all available ACE2 members.
The threshold is full-climatology/no-LOYO and is saved once, without a year
dimension, so exactly the same field can be applied to control and perturbation
arms.

By default, daily Tmax and RHmin receive the existing Jia-style additive ACE2
bias correction before Heat Index is computed.  ``--model-hi-inputs raw`` is
available for a fully uncorrected sensitivity test.

This is the memory-intensive step.  For 1980-2022 on the 1-degree grid, the
historical daily-HI cube is roughly 26 GB.  Use the accompanying full-node
Slurm script rather than a login node.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from heat_index_era5 import heat_index


JJA_DATES: list[tuple[int, int]] = (
    [(6, day) for day in range(1, 31)]
    + [(7, day) for day in range(1, 32)]
    + [(8, day) for day in range(1, 32)]
)
JJA_POSITION = {date: i for i, date in enumerate(JJA_DATES)}
N_MEMBERS = 25


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        start, end = (int(value) for value in spec.split(":", 1))
        return list(range(start, end + 1))
    if "-" in spec and "," not in spec:
        start, end = (int(value) for value in spec.split("-", 1))
        return list(range(start, end + 1))
    return [int(value) for value in spec.split(",") if value.strip()]


def atomic_netcdf(dataset: xr.Dataset, path: Path, encoding: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, encoding=encoding or {})
    os.replace(temporary, path)


def load_bias(
    path: Path,
    model_hi_inputs: str,
    years: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_hi_inputs == "raw":
        return None, None
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ACE2 bias fields: {path}\n"
            "Run scripts/hhe_ace2_biascorr.py first or use --model-hi-inputs raw."
        )
    with xr.open_dataset(path) as dataset:
        bias_tmax = np.asarray(dataset["bias_Tmax"], dtype=np.float32)
        bias_rhmin = np.asarray(dataset["bias_RHmin"], dtype=np.float32)
        method = str(dataset.attrs.get("method", ""))
    expected_period = f"{years[0]}-{years[-1]}"
    if method and expected_period not in method:
        raise ValueError(
            f"Bias file {path} does not document the requested {expected_period} period. "
            "Rebuild it with hhe_ace2_biascorr.py --force-bias."
        )
    return bias_tmax, bias_rhmin


def first_cache_file(cache_dir: Path, years: list[int]) -> Path:
    for year in years:
        for member in range(N_MEMBERS):
            path = cache_dir / f"daily_y{year}_mem{member:02d}.nc"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No ACE2 daily cache files found under {cache_dir}")


def load_historical_hi(
    cache_dir: Path,
    years: list[int],
    bias_tmax: np.ndarray | None,
    bias_rhmin: np.ndarray | None,
    model_hi_inputs: str,
    min_members: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    template = first_cache_file(cache_dir, years)
    with xr.open_dataset(template) as dataset:
        lat = np.asarray(dataset["lat"], dtype=np.float32)
        lon = np.asarray(dataset["lon"], dtype=np.float32) % 360.0
    nlat, nlon = lat.size, lon.size
    if bias_tmax is not None and bias_tmax.shape != (nlat, nlon):
        raise ValueError(f"bias_Tmax shape {bias_tmax.shape} != cache grid {(nlat, nlon)}")
    if bias_rhmin is not None and bias_rhmin.shape != (nlat, nlon):
        raise ValueError(f"bias_RHmin shape {bias_rhmin.shape} != cache grid {(nlat, nlon)}")

    shape = (len(years), N_MEMBERS, len(JJA_DATES), nlat, nlon)
    gib = np.prod(shape, dtype=np.int64) * np.dtype(np.float32).itemsize / 1024**3
    print(f"allocating historical HI cube {shape}: {gib:.1f} GiB", flush=True)
    historical = np.full(shape, np.nan, dtype=np.float32)
    counts: list[int] = []

    for year_index, year in enumerate(years):
        used = 0
        for member in range(N_MEMBERS):
            path = cache_dir / f"daily_y{year}_mem{member:02d}.nc"
            if not path.is_file():
                continue
            with xr.open_dataset(path) as dataset:
                tmax_c = np.asarray(dataset["tmax_C"], dtype=np.float32)
                rhmin = np.asarray(dataset["rhmin_pct"], dtype=np.float32)
                times = pd.DatetimeIndex(dataset["time"].values)
                file_lat = np.asarray(dataset["lat"], dtype=float)
                file_lon = np.asarray(dataset["lon"], dtype=float) % 360.0
            if not np.allclose(file_lat, lat) or not np.allclose(file_lon, lon):
                raise ValueError(f"Grid mismatch in {path}")
            if model_hi_inputs == "bias-corrected":
                tmax_c = tmax_c - bias_tmax
                rhmin = np.clip(rhmin - bias_rhmin, 0.0, 100.0)
            hi = heat_index(tmax_c * 9.0 / 5.0 + 32.0, rhmin).astype(np.float32)
            for time_index, timestamp in enumerate(times):
                position = JJA_POSITION.get((int(timestamp.month), int(timestamp.day)))
                if position is not None:
                    historical[year_index, member, position] = hi[time_index]
            used += 1
        counts.append(used)
        print(f"loaded {year}: {used}/{N_MEMBERS} members", flush=True)
        if used < min_members:
            raise RuntimeError(
                f"Historical year {year} has only {used} daily-cache members; "
                f"--min-members={min_members}"
            )
    return historical, lat, lon, counts


def compute_frozen_threshold(
    historical_hi: np.ndarray,
    percentile: float,
    window: int,
    jobs: int,
) -> np.ndarray:
    """Exact daywise percentile without broadcasting a duplicate year axis."""
    _, _, n_days, nlat, nlon = historical_hi.shape

    def one_day(day_index: int) -> np.ndarray:
        start = max(0, day_index - window)
        stop = min(n_days, day_index + window + 1)
        # This contiguous copy is intentionally local to a worker.  Limiting
        # --jobs bounds peak memory while retaining exact numpy percentiles.
        pool = np.ascontiguousarray(
            historical_hi[:, :, start:stop, :, :]
        ).reshape(-1, nlat, nlon)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            result = np.nanpercentile(pool, percentile, axis=0).astype(np.float32)
        print(
            f"threshold day {day_index + 1:02d}/{n_days}: "
            f"pool days {start + 1}-{stop}",
            flush=True,
        )
        return result

    results = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(one_day)(day_index) for day_index in range(n_days)
    )
    return np.stack(results, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="1980:2022")
    parser.add_argument(
        "--daily-cache-dir",
        type=Path,
        default=Path("outputs/lag_may/heat_index_era5/ace2_daily_cache"),
    )
    parser.add_argument(
        "--bias-nc",
        type=Path,
        default=Path("outputs/lag_may/heat_index_era5/bias_fields.nc"),
    )
    parser.add_argument(
        "--model-hi-inputs",
        choices=("bias-corrected", "raw"),
        default="bias-corrected",
    )
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--min-members", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    years = parse_years(args.years)
    if not years:
        raise ValueError("No historical years were requested")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not 0.0 < args.percentile < 100.0:
        raise ValueError("--percentile must be between 0 and 100")

    bias_tmax, bias_rhmin = load_bias(args.bias_nc, args.model_hi_inputs, years)
    historical, lat, lon, member_counts = load_historical_hi(
        args.daily_cache_dir,
        years,
        bias_tmax,
        bias_rhmin,
        args.model_hi_inputs,
        args.min_members,
    )
    threshold = compute_frozen_threshold(
        historical,
        percentile=args.percentile,
        window=args.window,
        jobs=args.jobs,
    )
    del historical

    threshold_id = (
        f"ace2_hi_p{args.percentile:g}_pm{args.window}d_"
        f"{years[0]}_{years[-1]}_{args.model_hi_inputs}"
    )
    dataset = xr.Dataset(
        {
            "relative_hhe_threshold": (
                ("jja_day", "lat", "lon"),
                threshold,
            ),
            "historical_member_count": (
                ("year",),
                np.asarray(member_counts, dtype=np.int16),
            ),
        },
        coords={
            "jja_day": np.arange(len(JJA_DATES), dtype=np.int16),
            "month": ("jja_day", np.asarray([month for month, _ in JJA_DATES], dtype=np.int8)),
            "day": ("jja_day", np.asarray([day for _, day in JJA_DATES], dtype=np.int8)),
            "lat": lat,
            "lon": lon,
            "year": np.asarray(years, dtype=np.int16),
        },
        attrs={
            "threshold_id": threshold_id,
            "definition": "daily Heat Index > grid-cell/calendar-day percentile",
            "percentile": args.percentile,
            "window_days_each_side": args.window,
            "window_days_total": 2 * args.window + 1,
            "threshold_method": "full historical ACE2 climatology; no LOYO; pooled years and members",
            "historical_years": f"{years[0]}-{years[-1]}",
            "model_hi_inputs": args.model_hi_inputs,
            "daily_cache_dir": str(args.daily_cache_dir.resolve()),
            "bias_file": str(args.bias_nc.resolve()) if args.model_hi_inputs == "bias-corrected" else "none",
        },
    )
    dataset["relative_hhe_threshold"].attrs.update(units="degF", long_name="Frozen ACE2 relative-HHE threshold")
    encoding = {
        "relative_hhe_threshold": {
            "zlib": True,
            "complevel": 2,
            "chunksizes": (1, len(lat), len(lon)),
        }
    }
    atomic_netcdf(dataset, args.out, encoding=encoding)
    summary = {
        "threshold_id": threshold_id,
        "output": str(args.out.resolve()),
        "years": years,
        "member_counts": member_counts,
        "percentile": args.percentile,
        "window_days_each_side": args.window,
        "model_hi_inputs": args.model_hi_inputs,
        "finite_fraction": float(np.isfinite(threshold).mean()),
        "mean_threshold_degF": float(np.nanmean(threshold)),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
