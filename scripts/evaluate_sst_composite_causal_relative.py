#!/usr/bin/env python3
"""Re-evaluate completed paired SST experiments with a fixed relative-HHE event.

No ACE2 simulation is launched.  Daily Heat Index is reconstructed from each
existing trajectory, compared with one frozen historical ACE2 grid-cell/day
percentile threshold, and summarized as the fraction of JJA days exceeding the
threshold.  The same threshold and bias correction are applied to both arms.

Inference remains across independent atmospheric base years.  The 25 lagged
initializations are averaged within a base year and are descriptive, not 25
independent climate replicates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from heat_index_era5 import heat_index  # noqa: E402
from hhe_ace2 import NEEDED, assign_times, rh_from_q  # noqa: E402
from sst_teleconnection_jja_sliding7d import _draw_coast_and_borders  # noqa: E402


N_MEMBERS = 25


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        start, end = (int(value) for value in spec.split(":", 1))
        return list(range(start, end + 1))
    return [int(value) for value in spec.split(",") if value.strip()]


def parse_members(spec: str) -> list[int]:
    return list(range(N_MEMBERS)) if spec == "all" else [int(value) for value in spec.split(",")]


def lag_times(year: int, month: int, day: int) -> list[datetime]:
    center = datetime(year, month, day)
    return [center + timedelta(hours=6 * (index - 12)) for index in range(N_MEMBERS)]


def atomic_netcdf(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def load_threshold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen relative-HHE threshold: {path}")
    with xr.open_dataset(path) as dataset:
        threshold = np.asarray(dataset["relative_hhe_threshold"], dtype=np.float32)
        lat = np.asarray(dataset["lat"], dtype=float)
        lon = np.asarray(dataset["lon"], dtype=float) % 360.0
        months = np.asarray(dataset["month"], dtype=int)
        days = np.asarray(dataset["day"], dtype=int)
        attrs = dict(dataset.attrs)
    lookup = {(int(month), int(day)): index for index, (month, day) in enumerate(zip(months, days))}
    if threshold.shape != (len(lookup), lat.size, lon.size):
        raise ValueError(f"Unexpected threshold shape {threshold.shape}")
    required = {"threshold_id", "percentile", "window_days_each_side", "model_hi_inputs"}
    missing = required.difference(attrs)
    if missing:
        raise ValueError(f"Threshold file is missing attributes: {sorted(missing)}")
    return threshold, lat, lon, lookup, attrs


def load_bias(
    path: Path | None,
    model_hi_inputs: str,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_hi_inputs == "raw":
        return None, None
    if path is None or not path.is_file():
        raise FileNotFoundError("Bias-corrected relative HHE requires --bias-nc")
    with xr.open_dataset(path) as dataset:
        bias_tmax = np.asarray(dataset["bias_Tmax"], dtype=np.float32)
        bias_rhmin = np.asarray(dataset["bias_RHmin"], dtype=np.float32)
        bias_lat = np.asarray(dataset["lat"], dtype=float)
        bias_lon = np.asarray(dataset["lon"], dtype=float) % 360.0
    if (
        bias_tmax.shape != (lat.size, lon.size)
        or not np.allclose(bias_lat, lat)
        or not np.allclose(bias_lon, lon)
    ):
        raise ValueError(f"Bias grid in {path} does not match the frozen threshold grid")
    return bias_tmax, bias_rhmin


def load_land(path: Path | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    with xr.open_dataset(path) as dataset:
        land = dataset["land_fraction"]
        if "time" in land.dims:
            land = land.isel(time=0)
        result = np.asarray(land) > 0.5
    if result.shape != expected_shape:
        raise ValueError(f"Land mask shape {result.shape} != threshold grid {expected_shape}")
    return result


def cache_is_valid(
    path: Path,
    threshold_id: str,
    model_hi_inputs: str,
    init_time: datetime,
) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as dataset:
            attrs = dataset.attrs
            return (
                "relative_hhe_frequency" in dataset
                and attrs.get("threshold_id") == threshold_id
                and attrs.get("model_hi_inputs") == model_hi_inputs
                and attrs.get("initialization") == init_time.isoformat()
            )
    except Exception:
        return False


def member_relative_frequency(
    prediction_path: Path,
    init_time: datetime,
    threshold: np.ndarray,
    threshold_lookup: dict[tuple[int, int], int],
    threshold_attrs: dict,
    bias_tmax: np.ndarray | None,
    bias_rhmin: np.ndarray | None,
    model_hi_inputs: str,
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
    min_jja_days: int,
    cache_path: Path | None,
    force_cache: bool,
) -> np.ndarray | None:
    threshold_id = str(threshold_attrs["threshold_id"])
    if cache_path is not None and not force_cache and cache_is_valid(
        cache_path, threshold_id, model_hi_inputs, init_time
    ):
        with xr.open_dataset(cache_path) as dataset:
            return np.asarray(dataset["relative_hhe_frequency"], dtype=np.float32)

    if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
        return None
    with xr.open_dataset(prediction_path, decode_times=False) as raw:
        if not set(NEEDED).issubset(raw.data_vars):
            return None
        dataset = assign_times(raw, init_time)
        temperature = dataset["TMP2m"]
        humidity = dataset["Q2m"]
        pressure = dataset["PRESsfc"]
        if "sample" in temperature.dims:
            temperature = temperature.isel(sample=0)
            humidity = humidity.isel(sample=0)
            pressure = pressure.isel(sample=0)
        relative_humidity = xr.apply_ufunc(
            rh_from_q,
            temperature,
            humidity,
            pressure,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        tmax_c = (temperature.resample(time="1D").max() - 273.15)
        rhmin = relative_humidity.resample(time="1D").min()
        jja = tmax_c.time.dt.month.isin([6, 7, 8])
        tmax_c = tmax_c.sel(time=jja).load()
        rhmin = rhmin.sel(time=jja).load()

    lat = np.asarray(tmax_c["lat"], dtype=float)
    lon = np.asarray(tmax_c["lon"], dtype=float) % 360.0
    if not np.allclose(lat, expected_lat) or not np.allclose(lon, expected_lon):
        raise ValueError(f"Prediction grid in {prediction_path} does not match threshold grid")
    times = pd.DatetimeIndex(tmax_c["time"].values)
    positions = [threshold_lookup.get((int(stamp.month), int(stamp.day))) for stamp in times]
    keep = np.array([position is not None for position in positions], dtype=bool)
    if int(keep.sum()) < min_jja_days:
        print(
            f"WARNING: {prediction_path} contains only {int(keep.sum())} usable JJA days; skipped",
            file=sys.stderr,
        )
        return None
    positions_array = np.asarray([position for position in positions if position is not None], dtype=int)
    tmax_values = np.asarray(tmax_c, dtype=np.float32)[keep]
    rhmin_values = np.asarray(rhmin, dtype=np.float32)[keep]
    if model_hi_inputs == "bias-corrected":
        tmax_values = tmax_values - bias_tmax
        rhmin_values = np.clip(rhmin_values - bias_rhmin, 0.0, 100.0)
    hi = heat_index(tmax_values * 9.0 / 5.0 + 32.0, rhmin_values).astype(np.float32)
    daily_threshold = threshold[positions_array]
    valid = np.isfinite(hi) & np.isfinite(daily_threshold)
    hits = (hi > daily_threshold) & valid
    denominator = valid.sum(axis=0)
    frequency = np.divide(
        hits.sum(axis=0),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32)

    if cache_path is not None:
        cache = xr.Dataset(
            {"relative_hhe_frequency": (("lat", "lon"), frequency)},
            coords={"lat": expected_lat, "lon": expected_lon},
            attrs={
                "threshold_id": threshold_id,
                "model_hi_inputs": model_hi_inputs,
                "initialization": init_time.isoformat(),
                "n_jja_days": int(keep.sum()),
                "source_prediction": str(prediction_path.resolve()),
                "definition": "fraction of JJA days with daily HI > fixed historical ACE2 p90 threshold",
            },
        )
        atomic_netcdf(cache, cache_path)
    return frequency


def weighted_region(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    box: list[float],
    land: np.ndarray | None,
) -> float:
    south, north, west, east = box
    lon360 = lon % 360.0
    latitude_mask = (lat[:, None] >= south) & (lat[:, None] <= north)
    if west % 360.0 <= east % 360.0:
        longitude_mask = (lon360[None, :] >= west % 360.0) & (lon360[None, :] <= east % 360.0)
    else:
        longitude_mask = (lon360[None, :] >= west % 360.0) | (lon360[None, :] <= east % 360.0)
    mask = latitude_mask & longitude_mask & np.isfinite(field)
    if land is not None:
        mask &= land
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    denominator = np.sum(weights)
    if denominator <= 0:
        raise ValueError("Regional mask contains no valid grid cells")
    return float(np.nansum(field * weights) / denominator)


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    count = len(values)
    if count <= 20:
        codes = np.arange(1 << count, dtype=np.uint64)[:, None]
        bits = ((codes >> np.arange(count, dtype=np.uint64)) & 1).astype(float)
        means = ((2 * bits - 1) * values).mean(axis=1)
    else:
        signs = rng.choice((-1.0, 1.0), size=(draws, count))
        means = (signs * values).mean(axis=1)
    return float((np.count_nonzero(np.abs(means) >= observed) + 1) / (len(means) + 1))


def json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def render_figure(
    path: Path,
    high: np.ndarray,
    low: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land: np.ndarray | None,
    box: list[float],
    years: list[int],
    year_effects: np.ndarray,
    case_members: list[int],
    case_effects: list[float],
    high_label: str,
    low_label: str,
    percentile: float,
    window: int,
) -> None:
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon180)
    x = lon180[order]
    high_pct = 100.0 * high[:, order]
    low_pct = 100.0 * low[:, order]
    difference_pct = high_pct - low_pct
    if land is not None:
        plot_land = land[:, order]
        high_pct = np.where(plot_land, high_pct, np.nan)
        low_pct = np.where(plot_land, low_pct, np.nan)
        difference_pct = np.where(plot_land, difference_pct, np.nan)

    finite_frequency = np.concatenate(
        [high_pct[np.isfinite(high_pct)], low_pct[np.isfinite(low_pct)]]
    )
    frequency_max = max(
        float(np.nanpercentile(finite_frequency, 99)) if finite_frequency.size else 1.0,
        1.0,
    )
    finite_difference = np.abs(difference_pct[np.isfinite(difference_pct)])
    difference_max = max(
        float(np.nanpercentile(finite_difference, 99)) if finite_difference.size else 1.0,
        0.25,
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    extent_x, extent_y = (-130.0, -60.0), (20.0, 52.0)
    south, north, west, east = box
    west180 = west - 360.0 if west > 180.0 else west
    east180 = east - 360.0 if east > 180.0 else east

    def map_panel(ax, field, title, cmap, vmin, vmax, colorbar_label):
        mesh = ax.pcolormesh(x, lat, field, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim(*extent_x)
        ax.set_ylim(*extent_y)
        try:
            _draw_coast_and_borders(ax, extent_x, extent_y)
        except Exception as exc:
            print(f"WARNING: coastlines unavailable while rendering ({exc})", file=sys.stderr)
        ax.add_patch(
            Rectangle(
                (west180, south),
                east180 - west180,
                north - south,
                fill=False,
                edgecolor="#00A65A",
                linewidth=2.0,
                zorder=6,
            )
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.84, label=colorbar_label)

    map_panel(
        axes[0, 0], high_pct, f"a  {high_label} arm", "YlOrRd", 0, frequency_max,
        "% of JJA days above fixed ACE2 p90",
    )
    map_panel(
        axes[0, 1], low_pct, f"b  {low_label} arm", "YlOrRd", 0, frequency_max,
        "% of JJA days above fixed ACE2 p90",
    )
    map_panel(
        axes[1, 0], difference_pct, f"c  Relative-HHE contrast: {high_label} minus {low_label}",
        "RdBu_r", -difference_max, difference_max, "Relative-HHE change (percentage points)",
    )

    ax = axes[1, 1]
    if len(years) == 1:
        values = 100.0 * np.asarray(case_effects)
        labels = [f"m{member:02d}" for member in case_members]
        title = f"d  Paired member effects, base year {years[0]}"
        xlabel = "lag member (descriptive, not independent replicates)"
    else:
        values = 100.0 * year_effects
        labels = [str(year) for year in years]
        title = "d  Regional relative-HHE effect by independent base year"
        xlabel = "base year"
    colors = np.where(values >= 0.0, "#C43C39", "#3478B8")
    ax.bar(np.arange(len(values)), values, color=colors, width=0.78)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(
        float(np.mean(values)), color="#6A3D9A", linewidth=1.8, linestyle="--",
        label=f"mean = {np.mean(values):.2f} pp",
    )
    ax.set_xticks(np.arange(len(values)), labels, rotation=60 if len(values) > 12 else 0)
    ax.set_ylabel("regional relative-HHE effect (percentage points)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"ACE2 SST sensitivity using relative HHE | daily HI > fixed ACE2 p{percentile:g} "
        f"(±{window}-day window) | {len(case_effects)} paired cases",
        fontsize=14,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-runs", type=Path, required=True)
    parser.add_argument("--low-runs", type=Path, required=True)
    parser.add_argument("--high-cache-tag", required=True)
    parser.add_argument("--low-cache-tag", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--threshold-nc", type=Path, required=True)
    parser.add_argument("--bias-nc", type=Path, default=None)
    parser.add_argument(
        "--model-hi-inputs", choices=("bias-corrected", "raw"), default="bias-corrected"
    )
    parser.add_argument("--years", required=True)
    parser.add_argument("--members", default="all")
    parser.add_argument("--init-month", type=int, default=5)
    parser.add_argument("--init-day", type=int, default=1)
    parser.add_argument("--high-label", default="SST perturbation")
    parser.add_argument("--low-label", default="Climatological SST control")
    parser.add_argument("--index-box", nargs=4, type=float, default=[23, 38, 260, 283])
    parser.add_argument("--land-mask-forcing", type=Path, default=None)
    parser.add_argument("--min-members", type=int, default=20)
    parser.add_argument("--min-jja-days", type=int, default=90)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    threshold, lat, lon, threshold_lookup, threshold_attrs = load_threshold(args.threshold_nc)
    if threshold_attrs["model_hi_inputs"] != args.model_hi_inputs:
        raise ValueError(
            f"Threshold was built using {threshold_attrs['model_hi_inputs']}, "
            f"but evaluator requested {args.model_hi_inputs}"
        )
    bias_tmax, bias_rhmin = load_bias(
        args.bias_nc, args.model_hi_inputs, lat, lon
    )
    land = load_land(args.land_mask_forcing, (lat.size, lon.size))
    years, members = parse_years(args.years), parse_members(args.members)

    high_sum = np.zeros((lat.size, lon.size), dtype=np.float64)
    low_sum = np.zeros((lat.size, lon.size), dtype=np.float64)
    case_count = 0
    effects_by_year: list[float] = []
    kept_years: list[int] = []
    members_by_year: list[int] = []
    case_years: list[int] = []
    case_members: list[int] = []
    case_effects: list[float] = []

    for year in years:
        times = lag_times(year, args.init_month, args.init_day)
        year_high: list[np.ndarray] = []
        year_low: list[np.ndarray] = []
        regional: list[float] = []
        paired_member_ids: list[int] = []
        for member in members:
            high_prediction = (
                args.high_runs / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc"
            )
            low_prediction = (
                args.low_runs / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc"
            )
            high_cache = (
                args.cache_dir / args.high_cache_tag / str(year) / f"member_{member:02d}.nc"
            )
            low_cache = (
                args.cache_dir / args.low_cache_tag / str(year) / f"member_{member:02d}.nc"
            )
            high_frequency = member_relative_frequency(
                high_prediction,
                times[member],
                threshold,
                threshold_lookup,
                threshold_attrs,
                bias_tmax,
                bias_rhmin,
                args.model_hi_inputs,
                lat,
                lon,
                args.min_jja_days,
                high_cache,
                args.force_cache,
            )
            low_frequency = member_relative_frequency(
                low_prediction,
                times[member],
                threshold,
                threshold_lookup,
                threshold_attrs,
                bias_tmax,
                bias_rhmin,
                args.model_hi_inputs,
                lat,
                lon,
                args.min_jja_days,
                low_cache,
                args.force_cache,
            )
            if high_frequency is None or low_frequency is None:
                continue
            year_high.append(high_frequency)
            year_low.append(low_frequency)
            regional.append(
                weighted_region(
                    high_frequency - low_frequency,
                    lat,
                    lon,
                    args.index_box,
                    land,
                )
            )
            paired_member_ids.append(member)

        if len(regional) < args.min_members:
            print(
                f"{year}: only {len(regional)} paired members; excluded from inference",
                flush=True,
            )
            continue
        for high_frequency, low_frequency in zip(year_high, year_low):
            high_sum += high_frequency
            low_sum += low_frequency
            case_count += 1
        case_years.extend([year] * len(regional))
        case_members.extend(paired_member_ids)
        case_effects.extend(regional)
        effects_by_year.append(float(np.mean(regional)))
        kept_years.append(year)
        members_by_year.append(len(regional))
        print(
            f"{year}: n={len(regional)} relative-HHE regional contrast="
            f"{100.0 * np.mean(regional):+.3f} percentage points",
            flush=True,
        )

    if not kept_years or case_count == 0:
        raise RuntimeError("No complete paired cases found")
    effects = np.asarray(effects_by_year, dtype=float)
    rng = np.random.default_rng(args.seed)
    if len(effects) >= 2:
        t_result = stats.ttest_1samp(effects, 0.0)
        bootstrap = rng.choice(
            effects, size=(args.bootstrap_draws, len(effects)), replace=True
        ).mean(axis=1)
        confidence_interval = np.quantile(bootstrap, [0.025, 0.975])
        flip_p = sign_flip_p(effects, rng, args.bootstrap_draws)
        t_statistic, t_p_value = float(t_result.statistic), float(t_result.pvalue)
    else:
        t_statistic = t_p_value = flip_p = float("nan")
        confidence_interval = np.array([np.nan, np.nan])

    mean_high = (high_sum / case_count).astype(np.float32)
    mean_low = (low_sum / case_count).astype(np.float32)
    case_effect_array = np.asarray(case_effects, dtype=np.float32)
    output = xr.Dataset(
        {
            "high_hhe_frequency": (("lat", "lon"), mean_high),
            "low_hhe_frequency": (("lat", "lon"), mean_low),
            "high_minus_low_hhe_frequency": (("lat", "lon"), mean_high - mean_low),
            "regional_effect_by_year": (("year",), effects.astype(np.float32)),
            "n_paired_members": (("year",), np.asarray(members_by_year, dtype=np.int16)),
            "regional_effect_by_case": (("case",), case_effect_array),
            "case_year": (("case",), np.asarray(case_years, dtype=np.int16)),
            "case_member": (("case",), np.asarray(case_members, dtype=np.int16)),
        },
        coords={"lat": lat, "lon": lon, "year": kept_years},
        attrs={
            "frequency_units": "fraction of JJA days",
            "event_definition": "daily Heat Index > fixed historical ACE2 grid-cell/day percentile",
            "threshold_id": str(threshold_attrs["threshold_id"]),
            "threshold_file": str(args.threshold_nc.resolve()),
            "threshold_percentile": float(threshold_attrs["percentile"]),
            "threshold_window_days_each_side": int(threshold_attrs["window_days_each_side"]),
            "model_hi_inputs": args.model_hi_inputs,
            "regional_index_box": json.dumps(args.index_box),
            "inference_unit": "base year; lag members averaged within year",
            "n_map_member_year_pairs": case_count,
            "high_arm_label": args.high_label,
            "low_arm_label": args.low_label,
            "contrast": f"{args.high_label} minus {args.low_label}",
        },
    )
    atomic_netcdf(output, args.out)
    figure_path = args.figure or args.out.with_suffix(".png")
    render_figure(
        figure_path,
        mean_high,
        mean_low,
        lat,
        lon,
        land,
        args.index_box,
        kept_years,
        effects,
        case_members,
        case_effects,
        args.high_label,
        args.low_label,
        float(threshold_attrs["percentile"]),
        int(threshold_attrs["window_days_each_side"]),
    )

    mean_pp = 100.0 * float(effects.mean())
    median_pp = 100.0 * float(np.median(case_effect_array))
    summary = {
        "years": kept_years,
        "n_years": len(kept_years),
        "n_map_member_year_pairs": case_count,
        "regional_contrast_percentage_points": mean_pp,
        "regional_median_case_effect_percentage_points": median_pp,
        "positive_case_count": int(np.count_nonzero(case_effect_array > 0.0)),
        "bootstrap_95pct_ci_percentage_points": [
            json_number(value) for value in 100.0 * confidence_interval
        ],
        "paired_year_t_statistic": json_number(t_statistic),
        "paired_year_t_p_value": json_number(t_p_value),
        "year_block_sign_flip_p_value": json_number(flip_p),
        "index_box": args.index_box,
        "initialization": f"{args.init_month:02d}-{args.init_day:02d}",
        "event_definition": "daily HI > frozen historical ACE2 grid-cell/day p90",
        "threshold_id": str(threshold_attrs["threshold_id"]),
        "model_hi_inputs": args.model_hi_inputs,
        "contrast": f"{args.high_label} minus {args.low_label}",
        "figure": str(figure_path.resolve()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
