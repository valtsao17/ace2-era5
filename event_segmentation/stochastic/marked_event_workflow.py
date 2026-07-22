"""Archive-facing implementation of Workflow 5 marked-event evaluation."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/ace2_era5_mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import genpareto

SCRIPT_DIR = Path(__file__).resolve().parent
EVENT_SEGMENTATION_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVENT_SEGMENTATION_DIR))

from marked_event_core import (
    Event,
    Match,
    MatchConfig,
    allowed_circular_shifts,
    arrival_scores,
    bootstrap_year_interval,
    circular_shift_event_path,
    count_crps,
    dual_track_masks,
    event_from_record,
    event_support,
    event_to_record,
    extract_3d_events,
    fractions_skill_score,
    match_events_one_to_one,
    match_events_over_configs,
    marginal_preserving_independence_surrogate,
    percentile_thresholds,
    tail_weighted_crps,
)
from run_daily_first_shape_sweep import (
    DailyCase,
    DailyFirstPaths,
    ace2_daily_path,
    daily_cases_for_year,
    draw_borders,
    era5_daily_path,
    lon_for_plot,
)
from weekly_event_metrics import cell_area_km2


EVENT_RECORD_COLUMNS = [
    "event_id",
    "source",
    "track",
    "member",
    "year",
    "start",
    "peak",
    "end",
    "centroid_lat",
    "centroid_lon",
    "duration_days",
    "daily_area_km2",
    "footprint_area_km2",
    "max_daily_area_km2",
    "spacetime_volume_km2_days",
    "peak_temperature_C",
    "mean_temperature_C",
    "peak_threshold_excess_C",
    "mean_threshold_excess_C",
    "standardized_quantile_intensity",
    "region_id",
    "mask_encoding",
    "mask_rle",
    "nlat",
    "nlon",
]
MATCH_RECORD_COLUMNS = [
    "row_type",
    "observed_event_id",
    "forecast_event_id",
    "member",
    "track",
    "year",
    "match_radius_km",
    "match_tolerance_days",
    "cost",
    "timing_error_days",
    "centroid_distance_km",
    "tolerant_iou",
    "best_shift_days",
    "duration_ratio",
    "area_ratio",
    "intensity_error",
]
SUPPORT_RECORD_COLUMNS = [
    "observed_event_id",
    "track",
    "year",
    "region_id",
    "n_supporting_members",
    "n_valid_members",
    "member_support_probability",
    "supporting_members",
    "match_radius_km",
    "match_tolerance_days",
]


@dataclass(frozen=True)
class MarkedPaths:
    root: Path
    thresholds: Path
    catalogs: Path
    matching: Path
    process: Path
    nulls: Path
    coverage: Path
    paper: Path
    figures: Path
    metadata: Path
    logs: Path


def make_marked_paths(root: Path) -> MarkedPaths:
    paths = MarkedPaths(
        root=root,
        thresholds=root / "thresholds",
        catalogs=root / "catalogs",
        matching=root / "matching",
        process=root / "process",
        nulls=root / "nulls",
        coverage=root / "coverage",
        paper=root / "paper_metrics",
        figures=root / "figures",
        metadata=root / "metadata",
        logs=root / "logs",
    )
    for path in paths.__dict__.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def _log(message: str) -> None:
    print(f"[workflow5] {message}", flush=True)


def input_paths(root: Path) -> DailyFirstPaths:
    return DailyFirstPaths(
        out_root=root,
        daily_era5=root / "daily_fields" / "era5",
        daily_ace2=root / "daily_fields" / "ace2",
        thresholds=root / "thresholds",
        masks=root / "masks",
        tables=root / "tables",
        figures=root / "figures",
        logs=root / "logs",
    )


def period_slug(args) -> str:
    number = lambda value: f"{float(value):g}".replace(".", "p")
    lag = (
        f"lag{int(getattr(args, 'lag_group_size', 0))}"
        if int(getattr(args, "lag_group_size", 0)) > 0
        else "lagall"
    )
    operator = "ge" if getattr(args, "comparison_operator", ">") == ">=" else "gt"
    quantiles = getattr(args, "paper_quantiles_parsed", ())
    quantile_slug = (
        "u" + "-".join(number(100.0 * float(value)) for value in quantiles)
        if quantiles
        else "udefault"
    )
    gpd = "_gpd" if bool(getattr(args, "fit_gpd_pot", False)) else ""
    return (
        f"q{number(args.percentile)}_sigma{number(getattr(args, 'gaussian_sigma', 1.0))}"
        f"_fit{args.fit_years.replace(':', '-')}"
        f"_val{getattr(args, 'validation_years', 'validation').replace(':', '-')}"
        f"_test{args.test_years.replace(':', '-')}_w{int(args.calendar_window_days)}"
        f"_{lag}_{operator}_{quantile_slug}_{args.field_variant}{gpd}"
    )


def resolved_process_region_mode(args) -> str:
    mode = str(getattr(args, "process_region_mode", "auto"))
    if mode == "auto":
        return "catalog" if getattr(args, "frozen_region_map", None) is not None else "grid"
    return mode


def process_slug(args) -> str:
    mode = resolved_process_region_mode(args)
    if mode == "grid":
        width = f"{float(getattr(args, 'process_grid_degrees', 10.0)):g}".replace(".", "p")
        region = f"grid{width}deg"
    else:
        region = "catalog"
    partial = (
        "-partial"
        if (
            getattr(args, "process_windows", "weekly") == "weekly"
            and bool(getattr(args, "include_partial_final_week", False))
        )
        else ""
    )
    return (
        f"{period_slug(args)}_proc-{getattr(args, 'process_windows', 'weekly')}"
        f"-{region}{partial}"
    )


def threshold_artifact_path(paths: MarkedPaths, args) -> Path:
    return paths.thresholds / f"frozen_dual_track_thresholds_{period_slug(args)}.nc"


def event_catalog_path(paths: MarkedPaths, args) -> Path:
    return paths.catalogs / f"marked_event_catalog_{period_slug(args)}.csv"


def era5_reference_catalog_path(paths: MarkedPaths, args) -> Path:
    return paths.catalogs / f"era5_loyo_reference_event_catalog_{period_slug(args)}.csv"


def valid_members_path(paths: MarkedPaths, args) -> Path:
    return paths.catalogs / f"valid_members_by_year_{period_slug(args)}.csv"


def validation_selection_path(paths: MarkedPaths, args) -> Path:
    return paths.metadata / f"validation_event_filter_selection_{period_slug(args)}.csv"


def match_catalog_path(paths: MarkedPaths, args) -> Path:
    return paths.matching / f"one_to_one_event_matches_{period_slug(args)}.csv"


def member_scores_path(paths: MarkedPaths, args) -> Path:
    return paths.matching / f"member_event_scores_{period_slug(args)}.csv"


def event_support_path(paths: MarkedPaths, args) -> Path:
    return paths.matching / f"observed_event_member_support_{period_slug(args)}.csv"


def tolerance_surface_path(paths: MarkedPaths, args) -> Path:
    return paths.matching / f"spatiotemporal_tolerance_surface_{period_slug(args)}.csv"


def process_scores_path(paths: MarkedPaths, args) -> Path:
    return paths.process / f"count_arrival_scores_{process_slug(args)}.csv"


def arrival_curves_path(paths: MarkedPaths, args) -> Path:
    return paths.process / f"arrival_survival_curves_{process_slug(args)}.csv"


def reliability_path(paths: MarkedPaths, args) -> Path:
    return paths.process / f"occurrence_reliability_{process_slug(args)}.csv"


def null_scores_path(paths: MarkedPaths, args) -> Path:
    return paths.nulls / f"circular_shift_null_scores_{period_slug(args)}.csv"


def coverage_scores_path(paths: MarkedPaths, args) -> Path:
    return paths.coverage / f"spatiotemporal_fractions_skill_coverage_{period_slug(args)}.csv"


def bootstrap_path(paths: MarkedPaths, args) -> Path:
    return paths.metadata / f"year_block_bootstrap_intervals_{process_slug(args)}.csv"


def manifest_path(paths: MarkedPaths, args) -> Path:
    return paths.metadata / f"workflow5_marked_event_manifest_{period_slug(args)}.json"


def captions_path(paths: MarkedPaths, args) -> Path:
    return paths.figures / f"workflow5_figure_captions_{period_slug(args)}.md"


def methods_path(paths: MarkedPaths, args) -> Path:
    return paths.metadata / f"workflow5_marked_event_methods_{period_slug(args)}.md"


def run_log_path(paths: MarkedPaths, args) -> Path:
    return paths.logs / f"workflow5_run_{period_slug(args)}.json"


def paper_path(paths: MarkedPaths, args, stem: str) -> Path:
    return paths.paper / f"{stem}_{period_slug(args)}.csv"


def _args_without_case_limit(args):
    values = vars(args).copy()
    values["max_cases"] = None
    return SimpleNamespace(**values)


def cases_for_years(years: Sequence[int], args, *, apply_debug_limit: bool = False) -> list[DailyCase]:
    case_args = _args_without_case_limit(args)
    cases: list[DailyCase] = []
    for year in years:
        year_cases = daily_cases_for_year(int(year), case_args)
        if apply_debug_limit and args.marked_max_cases is not None:
            year_cases = year_cases[: int(args.marked_max_cases)]
        cases.extend(year_cases)
    return cases


def _field_variable(source: str, variant: str) -> str:
    if source == "era5":
        return "era5_smoothed_daily_tmax_C" if variant == "smoothed" else "era5_daily_tmax_C"
    if source == "ace2":
        return "ace2_smoothed_daily_tmax_C" if variant == "smoothed" else "ace2_daily_tmax_C"
    raise ValueError(source)


def validate_field_inputs(cases: Sequence[DailyCase], daily_paths: DailyFirstPaths, args) -> None:
    missing: list[Path] = []
    for case in cases:
        for path in (era5_daily_path(daily_paths, case, args), ace2_daily_path(daily_paths, case, args)):
            if not path.is_file():
                missing.append(path)
    if missing:
        shown = "\n".join(str(path) for path in missing[:8])
        extra = f"\n... and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise FileNotFoundError(
            "Workflow 5 requires the existing Workflow 3 daily fields. Missing:\n"
            + shown
            + extra
        )


def load_period_fields(
    cases: Sequence[DailyCase],
    daily_paths: DailyFirstPaths,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    era_rows: list[np.ndarray] = []
    ace_rows: list[np.ndarray] = []
    lat = lon = members = None
    era_var = _field_variable("era5", args.field_variant)
    ace_var = _field_variable("ace2", args.field_variant)
    for case in cases:
        with xr.open_dataset(era5_daily_path(daily_paths, case, args)) as ds:
            era_rows.append(ds[era_var].values.astype(np.float32))
            if lat is None:
                lat = ds["lat"].values.astype(float)
                lon = ds["lon"].values.astype(float)
        with xr.open_dataset(ace2_daily_path(daily_paths, case, args)) as ds:
            ace_rows.append(ds[ace_var].values.astype(np.float32))
            if members is None:
                members = (
                    ds["member"].values.astype(int)
                    if "member" in ds.coords
                    else np.arange(ds[ace_var].shape[0], dtype=int)
                )
    if lat is None or lon is None or members is None:
        raise ValueError("No daily fields were loaded.")
    return (
        np.stack(era_rows, axis=0),
        np.stack(ace_rows, axis=0),
        np.asarray(lat),
        np.asarray(lon),
        np.asarray(members),
    )


def lag_groups(n_members: int, group_size: int) -> list[tuple[int, ...]]:
    if group_size <= 0 or group_size >= n_members:
        return [tuple(range(n_members))]
    return [
        tuple(range(start, min(start + int(group_size), n_members)))
        for start in range(0, n_members, int(group_size))
    ]


def member_group_index(groups: Sequence[Sequence[int]], n_members: int) -> np.ndarray:
    output = np.full(n_members, -1, dtype=int)
    for group_index, group in enumerate(groups):
        output[np.asarray(group, dtype=int)] = group_index
    if np.any(output < 0):
        raise ValueError("Lag groups do not cover every ACE2 member.")
    return output


def _quantiles(args) -> list[float]:
    values = {float(args.percentile) / 100.0}
    values.update(float(value) for value in args.paper_quantiles_parsed)
    values.update((0.90, 0.95, 0.99))
    return sorted(value for value in values if 0.0 < value < 1.0)


def _decluster_excess_peaks(
    excess: np.ndarray,
    dates: pd.DatetimeIndex,
    run_length_days: int,
) -> np.ndarray:
    indices = np.flatnonzero(np.isfinite(excess) & (excess > 0.0))
    if indices.size == 0:
        return np.asarray([], dtype=float)
    peaks: list[float] = []
    current = float(excess[indices[0]])
    previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        gap = int((dates[index] - dates[previous]).days)
        if 0 < gap <= int(run_length_days):
            current = max(current, float(excess[index]))
        else:
            peaks.append(current)
            current = float(excess[index])
        previous = index
    peaks.append(current)
    return np.asarray(peaks, dtype=float)


def _fit_gpd_sample(peaks: np.ndarray, minimum: int) -> tuple[float, float]:
    values = np.asarray(peaks, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < int(minimum) or np.ptp(values) <= 1.0e-10:
        return np.nan, np.nan
    try:
        shape, _, scale = genpareto.fit(values, floc=0.0)
    except (ValueError, RuntimeError, FloatingPointError):
        return np.nan, np.nan
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return np.nan, np.nan
    return float(shape), float(scale)


def _fit_optional_gpd(
    era5: np.ndarray,
    ace2: np.ndarray,
    dates: pd.DatetimeIndex,
    date_md: np.ndarray,
    era_threshold_by_calendar: np.ndarray,
    ace_threshold_by_group_calendar: np.ndarray,
    groups: Sequence[Sequence[int]],
    args,
) -> dict[str, np.ndarray]:
    nlat, nlon = era5.shape[1:]
    era_shape = np.full((nlat, nlon), np.nan, dtype=np.float32)
    era_scale = np.full((nlat, nlon), np.nan, dtype=np.float32)
    era_count = np.zeros((nlat, nlon), dtype=np.int32)
    ace_shape = np.full((len(groups), nlat, nlon), np.nan, dtype=np.float32)
    ace_scale = np.full((len(groups), nlat, nlon), np.nan, dtype=np.float32)
    ace_count = np.zeros((len(groups), nlat, nlon), dtype=np.int32)
    era_threshold_time = era_threshold_by_calendar[date_md]
    for lat_index in range(nlat):
        for lon_index in range(nlon):
            peaks = _decluster_excess_peaks(
                era5[:, lat_index, lon_index] - era_threshold_time[:, lat_index, lon_index],
                dates,
                int(args.gpd_run_length_days),
            )
            era_count[lat_index, lon_index] = len(peaks)
            shape, scale = _fit_gpd_sample(peaks, int(args.gpd_min_peaks))
            era_shape[lat_index, lon_index] = shape
            era_scale[lat_index, lon_index] = scale
            for group_index, group in enumerate(groups):
                pooled: list[np.ndarray] = []
                threshold_time = ace_threshold_by_group_calendar[
                    group_index,
                    date_md,
                    lat_index,
                    lon_index,
                ]
                for member in group:
                    pooled.append(
                        _decluster_excess_peaks(
                            ace2[:, int(member), lat_index, lon_index] - threshold_time,
                            dates,
                            int(args.gpd_run_length_days),
                        )
                    )
                member_peaks = np.concatenate(pooled) if pooled else np.asarray([])
                ace_count[group_index, lat_index, lon_index] = len(member_peaks)
                shape, scale = _fit_gpd_sample(member_peaks, int(args.gpd_min_peaks))
                ace_shape[group_index, lat_index, lon_index] = shape
                ace_scale[group_index, lat_index, lon_index] = scale
    return {
        "era5_gpd_shape_xi": era_shape,
        "era5_gpd_scale_beta_C": era_scale,
        "era5_gpd_declustered_peak_count": era_count,
        "ace2_gpd_shape_xi": ace_shape,
        "ace2_gpd_scale_beta_C": ace_scale,
        "ace2_gpd_declustered_peak_count": ace_count,
    }


def fit_frozen_thresholds(
    fit_cases: Sequence[DailyCase],
    daily_paths: DailyFirstPaths,
    paths: MarkedPaths,
    args,
) -> Path:
    output = threshold_artifact_path(paths, args)
    if output.exists() and not args.force_marked:
        return output
    era5, ace2, lat, lon, members = load_period_fields(fit_cases, daily_paths, args)
    dates = pd.DatetimeIndex([case.date for case in fit_cases])
    month_days = sorted({timestamp.strftime("%m-%d") for timestamp in dates})
    md_index = {month_day: index for index, month_day in enumerate(month_days)}
    date_md = np.asarray([md_index[timestamp.strftime("%m-%d")] for timestamp in dates])
    groups = lag_groups(ace2.shape[1], int(args.lag_group_size))
    quantiles = _quantiles(args)
    era_thresholds: list[np.ndarray] = []
    ace_thresholds: list[np.ndarray] = []
    n_fit_days: list[int] = []
    for target_index in range(len(month_days)):
        selected = np.abs(date_md - target_index) <= int(args.calendar_window_days)
        n_fit_days.append(int(selected.sum()))
        era_q, ace_q = percentile_thresholds(
            era5[selected],
            ace2[selected],
            quantiles,
            lag_groups=groups,
        )
        era_thresholds.append(era_q)
        ace_thresholds.append(ace_q)
    era_q_all = np.stack(era_thresholds, axis=1)
    ace_q_all = np.stack(ace_thresholds, axis=2)
    q_values = np.asarray(quantiles, dtype=float)
    primary_index = int(np.argmin(np.abs(q_values - float(args.percentile) / 100.0)))
    upper_candidates = np.flatnonzero(q_values > q_values[primary_index])
    lower_candidates = np.flatnonzero(q_values < q_values[primary_index])
    if upper_candidates.size:
        scale_index = int(upper_candidates[0])
        era_scale = era_q_all[scale_index] - era_q_all[primary_index]
        ace_scale = ace_q_all[:, scale_index] - ace_q_all[:, primary_index]
        scale_definition = f"q{q_values[scale_index]:g}-q{q_values[primary_index]:g}"
    elif lower_candidates.size:
        scale_index = int(lower_candidates[-1])
        era_scale = era_q_all[primary_index] - era_q_all[scale_index]
        ace_scale = ace_q_all[:, primary_index] - ace_q_all[:, scale_index]
        scale_definition = f"q{q_values[primary_index]:g}-q{q_values[scale_index]:g}"
    else:
        era_scale = np.ones_like(era_q_all[primary_index])
        ace_scale = np.ones_like(ace_q_all[:, primary_index])
        scale_definition = "unit"
    era_scale = np.where(era_scale > 1.0e-6, era_scale, 1.0).astype(np.float32)
    ace_scale = np.where(ace_scale > 1.0e-6, ace_scale, 1.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        era_mean = np.nanmean(era5, axis=0)
        ace_mean = np.nanmean(ace2, axis=(0, 1))
    dataset = xr.Dataset(
        {
            "era5_quantile_C": (
                ("quantile", "calendar_day", "lat", "lon"),
                era_q_all.astype(np.float32),
            ),
            "ace2_quantile_C": (
                ("lag_group", "quantile", "calendar_day", "lat", "lon"),
                ace_q_all.astype(np.float32),
            ),
            "era5_tail_scale_C": (
                ("calendar_day", "lat", "lon"),
                era_scale,
            ),
            "ace2_tail_scale_C": (
                ("lag_group", "calendar_day", "lat", "lon"),
                ace_scale,
            ),
            "ace2_minus_era5_threshold_C": (
                ("lag_group", "calendar_day", "lat", "lon"),
                ace_q_all[:, primary_index] - era_q_all[primary_index][np.newaxis, ...],
            ),
            "ace2_minus_era5_mean_bias_C": (
                ("lat", "lon"),
                (ace_mean - era_mean).astype(np.float32),
            ),
            "n_era5_fit_days": (("calendar_day",), np.asarray(n_fit_days, dtype=np.int32)),
            "member_lag_group": (("member",), member_group_index(groups, len(members))),
        },
        coords={
            "quantile": q_values,
            "calendar_day": np.asarray(month_days, dtype=str),
            "lag_group": np.arange(len(groups), dtype=int),
            "member": members,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "frozen": 1,
            "fit_years": args.fit_years,
            "validation_years_reserved": args.validation_years,
            "test_years_reserved": args.test_years,
            "percentile": float(args.percentile),
            "calendar_window_half_width_days": int(args.calendar_window_days),
            "comparison_operator": args.comparison_operator,
            "field_variant": args.field_variant,
            "gaussian_sigma_grid_cells": float(args.gaussian_sigma),
            "ace2_pooling": (
                "members pooled within configured lag groups"
                if int(args.lag_group_size) > 0
                else "all ACE2 members pooled"
            ),
            "lag_groups_json": json.dumps([list(group) for group in groups]),
            "tail_scale_definition": scale_definition,
            "leakage_guard": "Only fit_years files were opened to construct this artifact.",
            "gpd_pot_enabled": int(bool(args.fit_gpd_pot)),
        },
    )
    if args.fit_gpd_pot:
        gpd = _fit_optional_gpd(
            era5,
            ace2,
            dates,
            date_md,
            era_q_all[primary_index],
            ace_q_all[:, primary_index],
            groups,
            args,
        )
        for name, values in gpd.items():
            dims = ("lag_group", "lat", "lon") if name.startswith("ace2_") else ("lat", "lon")
            dataset[name] = (dims, values)
        dataset.attrs.update(
            {
                "gpd_definition": (
                    "GPD fit to runs-declustered positive daily threshold excess peaks; "
                    "location fixed at zero. No GEV is fit to daily exceedances."
                ),
                "gpd_run_length_days": int(args.gpd_run_length_days),
                "gpd_min_declustered_peaks": int(args.gpd_min_peaks),
            }
        )
    encoding = {
        variable: {"zlib": True, "complevel": 4}
        for variable in dataset.data_vars
        if dataset[variable].ndim >= 2
    }
    dataset.to_netcdf(output, encoding=encoding)
    return output


def load_region_map(path: Path | None, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    if path is None:
        return np.zeros((len(lat), len(lon)), dtype=np.int32), {
            "regionalization": "predeclared full evaluation domain",
            "frozen": 1,
            "region_ids": [0],
        }
    project_src = Path(__file__).resolve().parents[1] / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from hiro_ace_pipeline.exchangeable_hierarchy import (
        load_frozen_exchangeable_region_map,
    )

    verified = load_frozen_exchangeable_region_map(path)
    with verified as ds:
        if "region_id" not in ds:
            raise ValueError(f"Frozen region artifact has no region_id: {path}")
        frozen = ds["region_id"].attrs.get("frozen", ds.attrs.get("frozen", 0))
        if str(frozen).strip().lower() not in {"1", "true", "yes"}:
            raise ValueError(
                f"Region map is not frozen and cannot be used for scoring: {path}"
            )
        if not np.array_equal(ds["lat"].values, lat) or not np.array_equal(ds["lon"].values, lon):
            raise ValueError("Frozen region map grid does not match Workflow 3 daily fields.")
        region = ds["region_id"].values.astype(np.int32)
        metadata = {
            "regionalization": str(path),
            "frozen": 1,
            "regionalization_id": ds.attrs.get(
                "regionalization_id",
                ds["region_id"].attrs.get("regionalization_id", "unknown"),
            ),
            "region_ids": sorted(int(value) for value in np.unique(region) if value >= 0),
        }
    return region, metadata


def _threshold_arrays_for_dates(
    threshold_ds: xr.Dataset,
    dates: Sequence[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_values = threshold_ds["quantile"].values.astype(float)
    primary_q = float(threshold_ds.attrs["percentile"]) / 100.0
    q_index = int(np.argmin(np.abs(q_values - primary_q)))
    calendar_days = [str(value) for value in threshold_ds["calendar_day"].values]
    lookup = {value: index for index, value in enumerate(calendar_days)}
    indices = np.asarray([lookup[pd.Timestamp(day).strftime("%m-%d")] for day in dates], dtype=int)
    era_threshold = threshold_ds["era5_quantile_C"].isel(quantile=q_index, calendar_day=indices).values
    ace_threshold = threshold_ds["ace2_quantile_C"].isel(quantile=q_index, calendar_day=indices).values
    era_scale = threshold_ds["era5_tail_scale_C"].isel(calendar_day=indices).values
    ace_scale = threshold_ds["ace2_tail_scale_C"].isel(calendar_day=indices).values
    member_groups = threshold_ds["member_lag_group"].values.astype(int)
    return era_threshold, ace_threshold, era_scale, ace_scale, member_groups


def _extract_year_catalog(
    year: int,
    cases: Sequence[DailyCase],
    daily_paths: DailyFirstPaths,
    threshold_ds: xr.Dataset,
    region_map: np.ndarray,
    args,
    *,
    include_ace2: bool,
) -> tuple[list[Event], list[int], np.ndarray, np.ndarray]:
    era5, ace2, lat, lon, members = load_period_fields(cases, daily_paths, args)
    dates = [case.date for case in cases]
    era_threshold, ace_threshold_groups, era_scale, ace_scale_groups, member_groups = (
        _threshold_arrays_for_dates(threshold_ds, dates)
    )
    ace_threshold = np.stack(
        [ace_threshold_groups[member_groups[index], :, :, :] for index in range(len(members))],
        axis=1,
    )
    ace_scale = np.stack(
        [ace_scale_groups[member_groups[index], :, :, :] for index in range(len(members))],
        axis=1,
    )
    tracks = dual_track_masks(
        era5,
        ace2,
        era_threshold,
        ace_threshold,
        comparison=args.comparison_operator,
    )
    area = cell_area_km2(lat, lon)
    era_standardized = (era5 - era_threshold) / era_scale
    events: list[Event] = []
    member_day_valid = np.any(np.isfinite(ace2), axis=(2, 3))
    valid_members = [
        int(members[index])
        for index in range(len(members))
        if np.all(member_day_valid[:, index])
    ]
    for track, (obs_mask, member_masks) in tracks.items():
        events.extend(
            extract_3d_events(
                obs_mask,
                era5,
                era_threshold,
                era_standardized,
                dates,
                lat,
                lon,
                area,
                source="era5",
                track=track,
                member=None,
                connectivity=int(args.connectivity),
                min_duration_days=int(args.min_duration_days),
                min_max_area_km2=float(args.min_event_area_km2),
                region_map=region_map,
            )
        )
        if not include_ace2:
            continue
        for member_index, member in enumerate(members):
            if int(member) not in valid_members:
                continue
            if track == "relative":
                threshold = ace_threshold[:, member_index]
                scale = ace_scale[:, member_index]
            else:
                threshold = era_threshold
                scale = era_scale
            standardized = (ace2[:, member_index] - threshold) / scale
            events.extend(
                extract_3d_events(
                    member_masks[:, member_index],
                    ace2[:, member_index],
                    threshold,
                    standardized,
                    dates,
                    lat,
                    lon,
                    area,
                    source="ace2",
                    track=track,
                    member=int(member),
                    connectivity=int(args.connectivity),
                    min_duration_days=int(args.min_duration_days),
                    min_max_area_km2=float(args.min_event_area_km2),
                    region_map=region_map,
                )
            )
    return events, valid_members, lat, lon


def select_event_filters_on_validation(
    validation_years: Sequence[int],
    daily_paths: DailyFirstPaths,
    paths: MarkedPaths,
    args,
) -> Path:
    """Select component filters from validation years without opening test files."""
    output = validation_selection_path(paths, args)
    if output.exists() and not args.force_marked:
        frame = pd.read_csv(output)
        selected = frame[frame["selected"] == 1]
        if len(selected) != 1:
            raise ValueError(f"Validation selection artifact is malformed: {output}")
        row = selected.iloc[0]
        args.connectivity = int(row["connectivity"])
        args.min_duration_days = int(row["min_duration_days"])
        args.min_event_area_km2 = float(row["min_event_area_km2"])
        return output
    threshold_ds = xr.load_dataset(threshold_artifact_path(paths, args))
    lat = threshold_ds["lat"].values.astype(float)
    lon = threshold_ds["lon"].values.astype(float)
    region_map, _ = load_region_map(args.frozen_region_map, lat, lon)
    area = cell_area_km2(lat, lon)
    spacing = _grid_spacing_km(lat, lon)
    config = _match_config(
        args,
        float(args.primary_match_radius_km),
        int(args.primary_match_tolerance_days),
        spacing,
    )
    original = (
        int(args.connectivity),
        int(args.min_duration_days),
        float(args.min_event_area_km2),
    )
    rows: list[dict[str, object]] = []
    for connectivity in args.validation_connectivities_parsed:
        for minimum_duration in args.validation_min_duration_days_parsed:
            for minimum_area in args.validation_min_event_area_km2_parsed:
                args.connectivity = int(connectivity)
                args.min_duration_days = int(minimum_duration)
                args.min_event_area_km2 = float(minimum_area)
                member_f1: list[float] = []
                n_observed = 0
                n_forecast = 0
                for year in validation_years:
                    cases = cases_for_years([int(year)], args, apply_debug_limit=True)
                    events, valid_members, _, _ = _extract_year_catalog(
                        int(year),
                        cases,
                        daily_paths,
                        threshold_ds,
                        region_map,
                        args,
                        include_ace2=True,
                    )
                    for track in ("relative", "absolute"):
                        observed = [
                            event
                            for event in events
                            if event.source == "era5" and event.track == track
                        ]
                        n_observed += len(observed)
                        for member in valid_members:
                            forecast = [
                                event
                                for event in events
                                if event.source == "ace2"
                                and event.track == track
                                and event.member == member
                            ]
                            n_forecast += len(forecast)
                            matches, misses, false_alarms = match_events_one_to_one(
                                observed,
                                forecast,
                                area,
                                config,
                                member=member,
                                track=track,
                            )
                            member_f1.append(
                                _event_score_row(matches, misses, false_alarms)["event_f1"]
                            )
                rows.append(
                    {
                        "connectivity": int(connectivity),
                        "min_duration_days": int(minimum_duration),
                        "min_event_area_km2": float(minimum_area),
                        "validation_mean_member_event_f1": (
                            float(np.mean(member_f1)) if member_f1 else np.nan
                        ),
                        "validation_n_observed_events_summed_across_tracks": n_observed,
                        "validation_n_forecast_events_summed_across_members": n_forecast,
                        "fit_years_never_used_for_selection": args.fit_years,
                        "validation_years": args.validation_years,
                        "test_years_unopened": args.test_years,
                    }
                )
    frame = pd.DataFrame(rows)
    finite = frame["validation_mean_member_event_f1"].fillna(-np.inf)
    best_index = int(finite.idxmax()) if len(frame) else 0
    if not np.isfinite(float(finite.iloc[best_index])):
        args.connectivity, args.min_duration_days, args.min_event_area_km2 = original
        fallback = (
            (frame["connectivity"] == original[0])
            & (frame["min_duration_days"] == original[1])
            & np.isclose(frame["min_event_area_km2"], original[2])
        )
        best_index = int(np.flatnonzero(fallback)[0]) if np.any(fallback) else 0
    row = frame.loc[best_index]
    args.connectivity = int(row["connectivity"])
    args.min_duration_days = int(row["min_duration_days"])
    args.min_event_area_km2 = float(row["min_event_area_km2"])
    frame["selected"] = 0
    frame.loc[best_index, "selected"] = 1
    frame["selection_rule"] = "maximum validation mean member event F1; first configured candidate breaks ties"
    frame.to_csv(output, index=False)
    threshold_ds.close()
    return output


def build_event_catalogs(
    all_reference_years: Sequence[int],
    test_years: Sequence[int],
    daily_paths: DailyFirstPaths,
    paths: MarkedPaths,
    args,
) -> tuple[Path, Path, Path, dict[str, object]]:
    catalog_output = event_catalog_path(paths, args)
    reference_output = era5_reference_catalog_path(paths, args)
    valid_output = valid_members_path(paths, args)
    if (
        catalog_output.exists()
        and reference_output.exists()
        and valid_output.exists()
        and not args.force_marked
    ):
        with xr.open_dataset(threshold_artifact_path(paths, args)) as threshold_ds:
            lat = threshold_ds["lat"].values.astype(float)
            lon = threshold_ds["lon"].values.astype(float)
        _, region_meta = load_region_map(args.frozen_region_map, lat, lon)
        return catalog_output, reference_output, valid_output, region_meta
    threshold_ds = xr.load_dataset(threshold_artifact_path(paths, args))
    lat = threshold_ds["lat"].values.astype(float)
    lon = threshold_ds["lon"].values.astype(float)
    region_map, region_meta = load_region_map(args.frozen_region_map, lat, lon)
    test_set = set(int(year) for year in test_years)
    test_events: list[Event] = []
    era_reference: list[Event] = []
    valid_rows: list[dict[str, object]] = []
    for year in all_reference_years:
        cases = cases_for_years([int(year)], args, apply_debug_limit=True)
        events, valid_members, _, _ = _extract_year_catalog(
            int(year),
            cases,
            daily_paths,
            threshold_ds,
            region_map,
            args,
            include_ace2=int(year) in test_set,
        )
        era_reference.extend(event for event in events if event.source == "era5")
        if int(year) in test_set:
            test_events.extend(events)
            valid_rows.append(
                {
                    "year": int(year),
                    "n_valid_members": len(valid_members),
                    "valid_members": json.dumps(valid_members, separators=(",", ":")),
                }
            )
    pd.DataFrame(
        [event_to_record(event) for event in test_events],
        columns=EVENT_RECORD_COLUMNS,
    ).to_csv(catalog_output, index=False)
    pd.DataFrame(
        [event_to_record(event) for event in era_reference],
        columns=EVENT_RECORD_COLUMNS,
    ).to_csv(reference_output, index=False)
    pd.DataFrame(valid_rows).to_csv(valid_output, index=False)
    threshold_ds.close()
    return catalog_output, reference_output, valid_output, region_meta


def read_events(path: Path) -> list[Event]:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if frame.empty:
        return []
    return [event_from_record(record) for record in frame.to_dict(orient="records")]


def _parse_json_ints(value: object) -> list[int]:
    return [int(item) for item in json.loads(str(value))]


def read_valid_members(path: Path) -> dict[int, list[int]]:
    frame = pd.read_csv(path)
    return {
        int(row["year"]): _parse_json_ints(row["valid_members"])
        for row in frame.to_dict(orient="records")
    }


def _tolerances(args) -> list[tuple[float, int]]:
    return [
        (float(radius), int(tau))
        for radius in args.match_radii_km_parsed
        for tau in args.match_tolerances_days_parsed
    ]


def _grid_spacing_km(lat: np.ndarray, lon: np.ndarray) -> float:
    lat_step = float(np.nanmedian(np.abs(np.diff(lat)))) * 111.0 if len(lat) > 1 else 111.0
    lon_step = (
        float(np.nanmedian(np.abs(np.diff(lon))))
        * 111.0
        * math.cos(math.radians(float(np.nanmedian(lat))))
        if len(lon) > 1
        else 111.0
    )
    return max(1.0, min(lat_step, lon_step))


def _match_config(args, radius_km: float, tau_days: int, grid_spacing_km: float) -> MatchConfig:
    return MatchConfig(
        max_distance_km=max(float(radius_km), 1.0e-9),
        max_timing_days=int(tau_days),
        tolerant_iou_radius_pixels=max(0, int(round(float(radius_km) / grid_spacing_km))),
        timing_weight=float(args.match_weight_timing),
        distance_weight=float(args.match_weight_distance),
        iou_weight=float(args.match_weight_iou),
        duration_weight=float(args.match_weight_duration),
        area_weight=float(args.match_weight_area),
        intensity_weight=float(args.match_weight_intensity),
    )


def _match_record(match: Match, *, radius_km: float, tau_days: int) -> dict[str, object]:
    record = asdict(match)
    record.update(
        {
            "row_type": "match",
            "match_radius_km": float(radius_km),
            "match_tolerance_days": int(tau_days),
        }
    )
    return record


def run_event_matching(
    events: Sequence[Event],
    valid_members_by_year: dict[int, list[int]],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> tuple[Path, Path, Path, Path]:
    match_output = match_catalog_path(paths, args)
    score_output = member_scores_path(paths, args)
    support_output = event_support_path(paths, args)
    surface_output = tolerance_surface_path(paths, args)
    if all(path.exists() for path in (match_output, score_output, support_output, surface_output)) and not args.force_marked:
        return match_output, score_output, support_output, surface_output
    area = cell_area_km2(lat, lon)
    spacing = _grid_spacing_km(lat, lon)
    match_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    support_frames: list[pd.DataFrame] = []
    tracks = ("relative", "absolute")
    years = sorted(valid_members_by_year)
    tolerance_pairs = _tolerances(args)
    configurations = [
        _match_config(args, radius_km, tau_days, spacing)
        for radius_km, tau_days in tolerance_pairs
    ]
    observed_index: dict[tuple[str, int], list[Event]] = {}
    forecast_index: dict[tuple[str, int, int], list[Event]] = {}
    for event in events:
        if event.source == "era5":
            observed_index.setdefault((event.track, event.year), []).append(event)
        elif event.source == "ace2" and event.member is not None:
            forecast_index.setdefault(
                (event.track, event.year, int(event.member)),
                [],
            ).append(event)
    total_member_cases = sum(
        len(valid_members_by_year[year]) for year in years
    ) * len(tracks)
    completed_member_cases = 0
    matching_started = time.perf_counter()
    for track in tracks:
        for year in years:
            observed = observed_index.get((track, year), [])
            members = valid_members_by_year[year]
            forecast_counts = [
                len(forecast_index.get((track, year, member), []))
                for member in members
            ]
            observed_voxels = sum(len(event.voxels) for event in observed)
            maximum_forecast_voxels = max(
                (
                    sum(
                        len(event.voxels)
                        for event in forecast_index.get((track, year, member), [])
                    )
                    for member in members
                ),
                default=0,
            )
            _log(
                "matching "
                f"{track} {year}: {len(observed)} ERA5 events/"
                f"{observed_voxels:,} voxels; ACE2 events per member "
                f"min/median/max={min(forecast_counts, default=0)}/"
                f"{np.median(forecast_counts) if forecast_counts else 0:g}/"
                f"{max(forecast_counts, default=0)}, max member voxels="
                f"{maximum_forecast_voxels:,}"
            )
            matches_by_tolerance: list[dict[int, list[Match]]] = [
                {} for _ in tolerance_pairs
            ]
            observed_geometry_cache: dict[tuple[int, int], object] = {}
            for member_index, member in enumerate(members, start=1):
                forecast = forecast_index.get((track, year, member), [])
                if member_index == 1:
                    _log(
                        f"matching {track} {year}: starting member {member:02d} "
                        f"with {len(forecast)} events/{sum(len(event.voxels) for event in forecast):,} voxels"
                    )
                results = match_events_over_configs(
                    observed,
                    forecast,
                    area,
                    configurations,
                    member=member,
                    track=track,
                    observed_geometry_cache=observed_geometry_cache,
                )
                for tolerance_index, (
                    (radius_km, tau_days),
                    (matches, misses, false_alarms),
                ) in enumerate(zip(tolerance_pairs, results)):
                    matches_by_tolerance[tolerance_index][member] = matches
                    for match in matches:
                        row = _match_record(match, radius_km=radius_km, tau_days=tau_days)
                        row["year"] = year
                        match_rows.append(row)
                    for event in misses:
                        match_rows.append(
                            {
                                "row_type": "miss",
                                "observed_event_id": event.event_id,
                                "forecast_event_id": "",
                                "member": member,
                                "track": track,
                                "year": year,
                                "match_radius_km": radius_km,
                                "match_tolerance_days": tau_days,
                            }
                        )
                    for event in false_alarms:
                        match_rows.append(
                            {
                                "row_type": "false_alarm",
                                "observed_event_id": "",
                                "forecast_event_id": event.event_id,
                                "member": member,
                                "track": track,
                                "year": year,
                                "match_radius_km": radius_km,
                                "match_tolerance_days": tau_days,
                            }
                        )
                    hits = len(matches)
                    n_misses = len(misses)
                    n_false = len(false_alarms)
                    score_rows.append(
                        {
                            "score_type": "member",
                            "track": track,
                            "year": year,
                            "member": member,
                            "match_radius_km": radius_km,
                            "match_tolerance_days": tau_days,
                            "hits": hits,
                            "misses": n_misses,
                            "false_alarms": n_false,
                            "event_precision": hits / (hits + n_false) if hits + n_false else np.nan,
                            "event_recall": hits / (hits + n_misses) if hits + n_misses else np.nan,
                            "event_f1": (
                                2 * hits / (2 * hits + n_misses + n_false)
                                if 2 * hits + n_misses + n_false
                                else 1.0
                            ),
                            "mean_tolerant_iou": (
                                float(np.mean([match.tolerant_iou for match in matches]))
                                if matches
                                else np.nan
                            ),
                            "mean_timing_error_days": (
                                float(np.mean([match.timing_error_days for match in matches]))
                                if matches
                                else np.nan
                            ),
                            "mean_centroid_distance_km": (
                                float(np.mean([match.centroid_distance_km for match in matches]))
                                if matches
                                else np.nan
                            ),
                        }
                    )
                completed_member_cases += 1
                if (
                    member_index == len(members)
                    or member_index % 5 == 0
                ):
                    elapsed = time.perf_counter() - matching_started
                    rate = completed_member_cases / elapsed if elapsed > 0 else 0.0
                    remaining = (
                        (total_member_cases - completed_member_cases) / rate
                        if rate > 0
                        else np.nan
                    )
                    _log(
                        f"matching progress {completed_member_cases}/{total_member_cases} "
                        f"member-years; elapsed={elapsed / 60:.1f} min, "
                        f"ETA={remaining / 60:.1f} min"
                    )
            for tolerance_index, (radius_km, tau_days) in enumerate(tolerance_pairs):
                support = event_support(
                    observed,
                    matches_by_tolerance[tolerance_index],
                    members,
                )
                support["match_radius_km"] = radius_km
                support["match_tolerance_days"] = tau_days
                support_frames.append(support)
    matches_df = pd.DataFrame(match_rows, columns=MATCH_RECORD_COLUMNS)
    scores_df = pd.DataFrame(score_rows)
    support_df = (
        pd.concat(support_frames, ignore_index=True)
        if support_frames
        else pd.DataFrame(columns=SUPPORT_RECORD_COLUMNS)
    )
    support_df = support_df.reindex(columns=SUPPORT_RECORD_COLUMNS)
    primary = scores_df[
        np.isclose(scores_df["match_radius_km"], float(args.primary_match_radius_km))
        & (scores_df["match_tolerance_days"] == int(args.primary_match_tolerance_days))
    ]
    oracle_rows: list[dict[str, object]] = []
    for (track, year), group in primary.groupby(["track", "year"]):
        best = group.sort_values("event_f1", ascending=False).iloc[0].to_dict()
        best["score_type"] = "oracle_best_member_upper_bound"
        oracle_rows.append(best)
    if oracle_rows:
        scores_df = pd.concat([scores_df, pd.DataFrame(oracle_rows)], ignore_index=True)
    surface = (
        scores_df[scores_df["score_type"] == "member"]
        .groupby(["track", "match_radius_km", "match_tolerance_days"], as_index=False)
        .agg(
            observed_event_f1=("event_f1", "mean"),
            observed_mean_iou=("mean_tolerant_iou", "mean"),
            n_member_years=("event_f1", "size"),
        )
    )
    matches_df.to_csv(match_output, index=False)
    scores_df.to_csv(score_output, index=False)
    support_df.to_csv(support_output, index=False)
    surface.to_csv(surface_output, index=False)
    return match_output, score_output, support_output, surface_output


def _catalog_mask_cube(
    events: Sequence[Event],
    ordinals: Sequence[int],
    nlat: int,
    nlon: int,
) -> np.ndarray:
    cube = np.zeros((len(ordinals), nlat, nlon), dtype=bool)
    lookup = {int(ordinal): index for index, ordinal in enumerate(ordinals)}
    for event in events:
        for ordinal_raw, cell_raw in event.voxels:
            time_index = lookup.get(int(ordinal_raw))
            if time_index is not None:
                cube[time_index].reshape(-1)[int(cell_raw)] = True
    return cube


def run_coverage_scores(
    events: Sequence[Event],
    valid_members_by_year: dict[int, list[int]],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> Path:
    """Write FSS as neighborhood coverage, never as matched-event probability."""
    output = coverage_scores_path(paths, args)
    if output.exists() and not args.force_marked:
        return output
    spacing = _grid_spacing_km(lat, lon)
    rows: list[dict[str, object]] = []
    for track in ("relative", "absolute"):
        for year, members in valid_members_by_year.items():
            ordinals = [
                case.date.date().toordinal()
                for case in cases_for_years([year], args, apply_debug_limit=True)
            ]
            observed_events = [
                event
                for event in events
                if event.source == "era5" and event.track == track and event.year == year
            ]
            observed_mask = _catalog_mask_cube(observed_events, ordinals, len(lat), len(lon))
            for radius_km, tau_days in _tolerances(args):
                radius_pixels = max(0, int(round(radius_km / spacing)))
                for member in members:
                    forecast_events = [
                        event
                        for event in events
                        if event.source == "ace2"
                        and event.track == track
                        and event.year == year
                        and event.member == member
                    ]
                    forecast_mask = _catalog_mask_cube(
                        forecast_events,
                        ordinals,
                        len(lat),
                        len(lon),
                    )
                    rows.append(
                        {
                            "diagnostic_type": "spatiotemporal_neighborhood_coverage_not_event_probability",
                            "track": track,
                            "year": year,
                            "member": member,
                            "spatial_radius_km": radius_km,
                            "spatial_radius_pixels": radius_pixels,
                            "temporal_radius_days": tau_days,
                            "fractions_skill_score": fractions_skill_score(
                                observed_mask,
                                forecast_mask,
                                radius_pixels,
                                temporal_radius_days=tau_days,
                            ),
                        }
                    )
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _window_definitions(
    year: int,
    cases: Sequence[DailyCase],
    mode: str,
    *,
    include_partial_final_week: bool = False,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex([case.date for case in cases])
    if dates.empty:
        return []
    if mode == "seasonal":
        return [("JJA", dates.min(), dates.max())]
    if mode == "weekly":
        windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        for start_index in range(0, len(dates), 7):
            selected = dates[start_index : start_index + 7]
            if len(selected) == 7 or (
                len(selected) and include_partial_final_week
            ):
                windows.append(
                    (
                        f"W{start_index // 7 + 1:02d}",
                        selected.min(),
                        selected.max(),
                    )
                )
        return windows
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for month in sorted(dates.month.unique()):
        selected = dates[dates.month == month]
        windows.append((pd.Timestamp(year=year, month=int(month), day=1).strftime("%b"), selected.min(), selected.max()))
    return windows


@dataclass(frozen=True)
class ProcessRegion:
    region_id: int
    label: str
    mode: str
    latitude_min: float | None = None
    latitude_max: float | None = None
    longitude_min: float | None = None
    longitude_max: float | None = None


def _process_regions(
    catalog_region_ids: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    args,
) -> list[ProcessRegion]:
    mode = resolved_process_region_mode(args)
    if mode == "catalog":
        return [
            ProcessRegion(
                region_id=int(region_id),
                label=f"frozen region {int(region_id)}",
                mode="catalog",
            )
            for region_id in sorted(set(int(value) for value in catalog_region_ids))
        ]
    width = float(getattr(args, "process_grid_degrees", 10.0))
    lat_values = np.asarray(lat, dtype=float)
    lon_values = np.asarray(lon, dtype=float)
    lat_start = math.floor(float(np.nanmin(lat_values)) / width) * width
    lat_stop = math.ceil(float(np.nanmax(lat_values)) / width) * width
    lon_start = math.floor(float(np.nanmin(lon_values)) / width) * width
    lon_stop = math.ceil(float(np.nanmax(lon_values)) / width) * width
    lat_edges = np.arange(lat_start, lat_stop + width * 0.5, width)
    lon_edges = np.arange(lon_start, lon_stop + width * 0.5, width)
    regions: list[ProcessRegion] = []
    for lat_index, (lat_min, lat_max) in enumerate(zip(lat_edges[:-1], lat_edges[1:])):
        latitude_max = (
            float(np.nextafter(lat_max, np.inf))
            if lat_index == len(lat_edges) - 2
            else float(lat_max)
        )
        if not np.any((lat_values >= lat_min) & (lat_values < latitude_max)):
            continue
        for lon_index, (lon_min, lon_max) in enumerate(zip(lon_edges[:-1], lon_edges[1:])):
            longitude_max = (
                float(np.nextafter(lon_max, np.inf))
                if lon_index == len(lon_edges) - 2
                else float(lon_max)
            )
            if not np.any((lon_values >= lon_min) & (lon_values < longitude_max)):
                continue
            regions.append(
                ProcessRegion(
                    region_id=len(regions),
                    label=(
                        f"{lat_min:g}–{lat_max:g}°N, "
                        f"{lon_min:g}–{lon_max:g}°E"
                    ),
                    mode="predeclared_grid",
                    latitude_min=float(lat_min),
                    latitude_max=latitude_max,
                    longitude_min=float(lon_min),
                    longitude_max=longitude_max,
                )
            )
    if not regions:
        raise ValueError("No predeclared process grid blocks intersect the evaluation grid.")
    return regions


def _event_in_process_region(event: Event, region: ProcessRegion) -> bool:
    if region.mode == "catalog":
        return event.region_id == region.region_id
    assert region.latitude_min is not None and region.latitude_max is not None
    assert region.longitude_min is not None and region.longitude_max is not None
    latitude_inside = region.latitude_min <= event.centroid_lat < region.latitude_max
    longitude_inside = region.longitude_min <= event.centroid_lon < region.longitude_max
    return bool(latitude_inside and longitude_inside)


def _events_in_unit(
    events: Sequence[Event],
    *,
    source: str,
    track: str,
    year: int,
    region: ProcessRegion,
    start: pd.Timestamp,
    end: pd.Timestamp,
    member: int | None = None,
) -> list[Event]:
    start_ordinal = start.date().toordinal()
    end_ordinal = end.date().toordinal()
    return [
        event
        for event in events
        if event.source == source
        and event.track == track
        and event.year == year
        and _event_in_process_region(event, region)
        and start_ordinal <= event.start_ordinal <= end_ordinal
        and (member is None or event.member == member)
    ]


def _first_arrival_day(events: Sequence[Event], start: pd.Timestamp) -> float:
    if not events:
        return float("nan")
    return float(min(event.start_ordinal for event in events) - start.date().toordinal())


def _index_process_events(
    events: Sequence[Event],
    regions: Sequence[ProcessRegion],
    windows_by_year: dict[int, list[tuple[str, pd.Timestamp, pd.Timestamp]]],
) -> dict[tuple[str, str, int, int, str, int | None], list[Event]]:
    region_by_catalog_id = {
        region.region_id: region for region in regions if region.mode == "catalog"
    }
    indexed: dict[
        tuple[str, str, int, int, str, int | None],
        list[Event],
    ] = {}
    for event in events:
        windows = windows_by_year.get(event.year)
        if not windows:
            continue
        if region_by_catalog_id:
            region = region_by_catalog_id.get(event.region_id)
        else:
            region = next(
                (
                    candidate
                    for candidate in regions
                    if _event_in_process_region(event, candidate)
                ),
                None,
            )
        if region is None:
            continue
        window_name = next(
            (
                name
                for name, start, end in windows
                if start.date().toordinal()
                <= event.start_ordinal
                <= end.date().toordinal()
            ),
            None,
        )
        if window_name is None:
            continue
        key = (
            event.source,
            event.track,
            event.year,
            region.region_id,
            window_name,
            event.member,
        )
        indexed.setdefault(key, []).append(event)
    return indexed


def run_count_arrival_process(
    events: Sequence[Event],
    reference_events: Sequence[Event],
    valid_members_by_year: dict[int, list[int]],
    catalog_region_ids: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> tuple[Path, Path, Path]:
    score_output = process_scores_path(paths, args)
    curves_output = arrival_curves_path(paths, args)
    reliability_output = reliability_path(paths, args)
    if all(path.exists() for path in (score_output, curves_output, reliability_output)) and not args.force_marked:
        return score_output, curves_output, reliability_output
    score_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    tracks = ("relative", "absolute")
    regions = _process_regions(catalog_region_ids, lat, lon, args)
    reference_years = sorted({event.year for event in reference_events})
    all_window_years = sorted(set(valid_members_by_year) | set(reference_years))
    windows_by_year = {
        year: _window_definitions(
            year,
            cases_for_years([year], args, apply_debug_limit=True),
            args.process_windows,
            include_partial_final_week=bool(
                getattr(args, "include_partial_final_week", False)
            ),
        )
        for year in all_window_years
    }
    event_index = _index_process_events(events, regions, windows_by_year)
    reference_index = _index_process_events(
        reference_events,
        regions,
        windows_by_year,
    )
    for year in sorted(valid_members_by_year):
        for window_name, start, end in windows_by_year[year]:
            horizon = int((end - start).days)
            for region in regions:
                for track in tracks:
                    observed = event_index.get(
                        ("era5", track, year, region.region_id, window_name, None),
                        [],
                    )
                    observed_count = len(observed)
                    observed_arrival = _first_arrival_day(observed, start)
                    counts: list[int] = []
                    arrivals: list[float] = []
                    for member in valid_members_by_year[year]:
                        member_events = event_index.get(
                            (
                                "ace2",
                                track,
                                year,
                                region.region_id,
                                window_name,
                                member,
                            ),
                            [],
                        )
                        counts.append(len(member_events))
                        arrivals.append(_first_arrival_day(member_events, start))
                    count_array = np.asarray(counts, dtype=float)
                    occurrence_probability = float(np.mean(count_array > 0))
                    count_mean = float(np.mean(count_array))
                    count_variance = float(np.var(count_array, ddof=1)) if len(count_array) > 1 else np.nan
                    arrival = arrival_scores(
                        arrivals,
                        None if not np.isfinite(observed_arrival) else observed_arrival,
                        horizon,
                    )
                    climatology_counts: list[int] = []
                    for reference_year in reference_years:
                        if reference_year == year:
                            continue
                        reference_windows = {
                            name: (ref_start, ref_end)
                            for name, ref_start, ref_end in windows_by_year[reference_year]
                        }
                        if window_name not in reference_windows:
                            continue
                        climatology_counts.append(
                            len(
                                reference_index.get(
                                    (
                                        "era5",
                                        track,
                                        reference_year,
                                        region.region_id,
                                        window_name,
                                        None,
                                    ),
                                    [],
                                )
                            )
                        )
                    climatology_array = np.asarray(climatology_counts, dtype=float)
                    climatology_probability = (
                        float(np.mean(climatology_array > 0)) if climatology_array.size else np.nan
                    )
                    row: dict[str, object] = {
                        "year": year,
                        "track": track,
                        "region_id": int(region.region_id),
                        "region_label": region.label,
                        "region_mode": region.mode,
                        "region_latitude_min": region.latitude_min,
                        "region_latitude_max": region.latitude_max,
                        "region_longitude_min": region.longitude_min,
                        "region_longitude_max": region.longitude_max,
                        "window": window_name,
                        "window_start": str(start.date()),
                        "window_end": str(end.date()),
                        "n_valid_members": len(valid_members_by_year[year]),
                        "member_event_counts": json.dumps(counts, separators=(",", ":")),
                        "member_first_arrival_days": json.dumps(
                            [None if not np.isfinite(value) else float(value) for value in arrivals],
                            separators=(",", ":"),
                        ),
                        "era5_event_count": observed_count,
                        "ace2_mean_event_count": count_mean,
                        "count_crps": count_crps(count_array, observed_count),
                        "count_bias": count_mean - observed_count,
                        "count_rate_ratio": count_mean / observed_count if observed_count > 0 else np.nan,
                        "ace2_count_variance": count_variance,
                        "dispersion_variance_over_mean": (
                            count_variance / count_mean if count_mean > 0 else np.nan
                        ),
                        "negative_binomial_alpha_mom": (
                            max(0.0, (count_variance - count_mean) / (count_mean**2))
                            if count_mean > 0 and np.isfinite(count_variance)
                            else np.nan
                        ),
                        "era5_occurrence": int(observed_count > 0),
                        "ace2_occurrence_probability": occurrence_probability,
                        "occurrence_brier_score": (
                            occurrence_probability - float(observed_count > 0)
                        )
                        ** 2,
                        "era5_first_arrival_day": observed_arrival,
                        "loyo_era5_climatology_n_years": int(climatology_array.size),
                        "loyo_era5_count_crps": count_crps(climatology_array, observed_count),
                        "loyo_era5_occurrence_probability": climatology_probability,
                        "loyo_era5_occurrence_brier": (
                            (climatology_probability - float(observed_count > 0)) ** 2
                            if np.isfinite(climatology_probability)
                            else np.nan
                        ),
                    }
                    row.update(arrival)
                    score_rows.append(row)
                    arrival_array = np.asarray(arrivals, dtype=float)
                    for day_index in range(horizon + 1):
                        probability = float(np.mean(np.isfinite(arrival_array) & (arrival_array <= day_index)))
                        curve_rows.append(
                            {
                                "year": year,
                                "track": track,
                                "region_id": int(region.region_id),
                                "region_label": region.label,
                                "window": window_name,
                                "day": day_index,
                                "probability_arrived_by_day": probability,
                                "survival_probability": 1.0 - probability,
                                "era5_arrived_by_day": int(
                                    np.isfinite(observed_arrival) and observed_arrival <= day_index
                                ),
                            }
                        )
    scores = pd.DataFrame(score_rows)
    curves = pd.DataFrame(curve_rows)
    reliability_rows: list[dict[str, object]] = []
    if not scores.empty:
        bins = np.linspace(0.0, 1.0, int(args.reliability_bins) + 1)
        for track, group in scores.groupby("track"):
            bin_index = np.clip(
                np.digitize(group["ace2_occurrence_probability"], bins, right=True) - 1,
                0,
                len(bins) - 2,
            )
            for index in range(len(bins) - 1):
                selected = group.iloc[np.flatnonzero(bin_index == index)]
                if selected.empty:
                    continue
                reliability_rows.append(
                    {
                        "track": track,
                        "probability_bin_lower": bins[index],
                        "probability_bin_upper": bins[index + 1],
                        "n_region_time_units": len(selected),
                        "mean_forecast_probability": selected[
                            "ace2_occurrence_probability"
                        ].mean(),
                        "observed_frequency": selected["era5_occurrence"].mean(),
                        "mean_brier_score": selected["occurrence_brier_score"].mean(),
                    }
                )
    scores.to_csv(score_output, index=False)
    curves.to_csv(curves_output, index=False)
    pd.DataFrame(reliability_rows).to_csv(reliability_output, index=False)
    return score_output, curves_output, reliability_output


def _event_score_row(
    matches: Sequence[Match],
    misses: Sequence[Event],
    false_alarms: Sequence[Event],
) -> dict[str, float]:
    hits = len(matches)
    denominator = 2 * hits + len(misses) + len(false_alarms)
    return {
        "hits": float(hits),
        "misses": float(len(misses)),
        "false_alarms": float(len(false_alarms)),
        "event_f1": 2.0 * hits / denominator if denominator else 1.0,
        "mean_tolerant_iou": (
            float(np.mean([match.tolerant_iou for match in matches])) if matches else np.nan
        ),
    }


def run_circular_shift_nulls(
    events: Sequence[Event],
    valid_members_by_year: dict[int, list[int]],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> Path:
    output = null_scores_path(paths, args)
    if output.exists() and not args.force_marked:
        return output
    if int(args.null_replicates) <= 0:
        pd.DataFrame().to_csv(output, index=False)
        return output
    area = cell_area_km2(lat, lon)
    spacing = _grid_spacing_km(lat, lon)
    rng = np.random.default_rng(int(args.random_seed) + 5005)
    rows: list[dict[str, object]] = []
    tolerance_pairs = _tolerances(args)
    configurations = [
        _match_config(args, radius_km, tau_days, spacing)
        for radius_km, tau_days in tolerance_pairs
    ]
    observed_index: dict[tuple[str, int], list[Event]] = {}
    forecast_index: dict[tuple[str, int, int], list[Event]] = {}
    for event in events:
        if event.source == "era5":
            observed_index.setdefault((event.track, event.year), []).append(event)
        elif event.source == "ace2" and event.member is not None:
            forecast_index.setdefault(
                (event.track, event.year, int(event.member)),
                [],
            ).append(event)
    for track in ("relative", "absolute"):
        for year, members in valid_members_by_year.items():
            observed = observed_index.get((track, year), [])
            season_ordinals = [
                case.date.date().toordinal()
                for case in cases_for_years([year], args, apply_debug_limit=True)
            ]
            allowed = allowed_circular_shifts(
                len(season_ordinals),
                min(
                    int(args.minimum_null_shift_days),
                    max(1, len(season_ordinals) // 3),
                ),
            )
            observed_geometry_cache: dict[tuple[int, int], object] = {}
            _log(
                f"circular null {track} {year}: "
                f"{int(args.null_replicates)} replicates × {len(members)} members"
            )
            for replicate in range(int(args.null_replicates)):
                for member in members:
                    forecast = forecast_index.get((track, year, member), [])
                    shift = int(rng.choice(allowed))
                    shifted = circular_shift_event_path(
                        forecast,
                        season_ordinals,
                        shift,
                    )
                    results = match_events_over_configs(
                        observed,
                        shifted,
                        area,
                        configurations,
                        member=member,
                        track=track,
                        observed_geometry_cache=observed_geometry_cache,
                    )
                    for (
                        (radius_km, tau_days),
                        (matches, misses, false_alarms),
                    ) in zip(tolerance_pairs, results):
                        row: dict[str, object] = {
                            "replicate": replicate,
                            "year": year,
                            "track": track,
                            "member": member,
                            "circular_shift_days": shift,
                            "match_radius_km": radius_km,
                            "match_tolerance_days": tau_days,
                            "null_type": "seasonal_circular_complete_member_path",
                        }
                        row.update(_event_score_row(matches, misses, false_alarms))
                        rows.append(row)
                if (
                    replicate == 0
                    or (replicate + 1) % 10 == 0
                    or replicate + 1 == int(args.null_replicates)
                ):
                    _log(
                        f"circular null {track} {year}: "
                        f"{replicate + 1}/{int(args.null_replicates)} replicates"
                    )
    for replicate in range(int(args.independence_surrogate_replicates)):
        for track in ("relative", "absolute"):
            for year, members in valid_members_by_year.items():
                season_ordinals = [
                    case.date.date().toordinal()
                    for case in cases_for_years([year], args, apply_debug_limit=True)
                ]
                dates = pd.to_datetime([date.fromordinal(value) for value in season_ordinals])
                observed = [
                    event
                    for event in events
                    if event.source == "era5" and event.track == track and event.year == year
                ]
                for member in members:
                    forecast = [
                        event
                        for event in events
                        if event.source == "ace2"
                        and event.track == track
                        and event.year == year
                        and event.member == member
                    ]
                    member_mask = _catalog_mask_cube(
                        forecast,
                        season_ordinals,
                        len(lat),
                        len(lon),
                    )
                    surrogate_mask = marginal_preserving_independence_surrogate(
                        member_mask,
                        minimum_shift_days=min(
                            int(args.minimum_null_shift_days),
                            max(1, len(season_ordinals) // 3),
                        ),
                        rng=rng,
                    )
                    surrogate = extract_3d_events(
                        surrogate_mask,
                        surrogate_mask.astype(float),
                        np.zeros_like(surrogate_mask, dtype=float),
                        np.zeros_like(surrogate_mask, dtype=float),
                        dates,
                        lat,
                        lon,
                        area,
                        source="ace2",
                        track=track,
                        member=member,
                        connectivity=int(args.connectivity),
                        min_duration_days=int(args.min_duration_days),
                        min_max_area_km2=float(args.min_event_area_km2),
                        region_map=None,
                    )
                    if forecast:
                        shuffled_marks = rng.choice(
                            np.asarray(
                                [
                                    event.standardized_quantile_intensity
                                    for event in forecast
                                ],
                                dtype=float,
                            ),
                            size=len(surrogate),
                            replace=True,
                        )
                        surrogate = [
                            replace(
                                event,
                                event_id=f"{event.event_id}_indrep{replicate}",
                                standardized_quantile_intensity=float(mark),
                            )
                            for event, mark in zip(surrogate, shuffled_marks)
                        ]
                    results = match_events_over_configs(
                        observed,
                        surrogate,
                        area,
                        configurations,
                        member=member,
                        track=track,
                    )
                    for (
                        (radius_km, tau_days),
                        (matches, misses, false_alarms),
                    ) in zip(tolerance_pairs, results):
                        row = {
                            "replicate": replicate,
                            "year": year,
                            "track": track,
                            "member": member,
                            "circular_shift_days": np.nan,
                            "match_radius_km": radius_km,
                            "match_tolerance_days": tau_days,
                            "null_type": "marginal_preserving_cellwise_independence",
                        }
                        row.update(_event_score_row(matches, misses, false_alarms))
                        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    surface_file = tolerance_surface_path(paths, args)
    if surface_file.exists() and not frame.empty:
        observed_surface = pd.read_csv(surface_file)
        circular = frame[
            frame["null_type"] == "seasonal_circular_complete_member_path"
        ]
        null_summary = (
            circular.groupby(
                ["track", "match_radius_km", "match_tolerance_days"],
                as_index=False,
            )
            .agg(
                shuffled_null_event_f1=("event_f1", "mean"),
                shuffled_null_event_f1_std=("event_f1", "std"),
                null_replicates=("replicate", "nunique"),
            )
        )
        merged = observed_surface.merge(
            null_summary,
            on=["track", "match_radius_km", "match_tolerance_days"],
            how="left",
        )
        merged["observed_minus_null_event_f1"] = (
            merged["observed_event_f1"] - merged["shuffled_null_event_f1"]
        )
        merged.to_csv(surface_file, index=False)
    return output


def run_bootstrap(paths: MarkedPaths, args) -> Path:
    output = bootstrap_path(paths, args)
    member_file = member_scores_path(paths, args)
    process_file = process_scores_path(paths, args)
    expected_artifacts: set[str] = set()
    if member_file.exists():
        expected_artifacts.add("primary_member_event_scores")
    if process_file.exists():
        expected_artifacts.add("count_arrival_process")
    if output.exists() and not args.force_marked:
        try:
            existing = pd.read_csv(output)
            existing_artifacts = (
                set(existing["artifact"].dropna().astype(str))
                if "artifact" in existing
                else set()
            )
        except pd.errors.EmptyDataError:
            existing_artifacts = set()
        if expected_artifacts <= existing_artifacts:
            return output
    frames: list[pd.DataFrame] = []
    if member_file.exists():
        member_scores = pd.read_csv(member_file)
        selected = member_scores[
            (member_scores["score_type"] == "member")
            & np.isclose(
                member_scores["match_radius_km"],
                float(args.primary_match_radius_km),
            )
            & (
                member_scores["match_tolerance_days"]
                == int(args.primary_match_tolerance_days)
            )
        ]
        intervals = bootstrap_year_interval(
            selected,
            ["event_f1", "mean_tolerant_iou", "mean_timing_error_days", "mean_centroid_distance_km"],
            n_replicates=int(args.bootstrap_replicates),
            random_seed=int(args.random_seed) + 51,
        )
        if not intervals.empty:
            intervals["artifact"] = "primary_member_event_scores"
            frames.append(intervals)
    if process_file.exists():
        process = pd.read_csv(process_file)
        intervals = bootstrap_year_interval(
            process,
            [
                "count_crps",
                "occurrence_brier_score",
                "count_bias",
                "first_arrival_crps_days",
                "integrated_arrival_brier_score",
            ],
            n_replicates=int(args.bootstrap_replicates),
            random_seed=int(args.random_seed) + 52,
        )
        if not intervals.empty:
            intervals["artifact"] = "count_arrival_process"
            frames.append(intervals)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False) if frames else pd.DataFrame().to_csv(output, index=False)
    return output


def _haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlon = np.deg2rad((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    value = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def _sample_distance_pairs(
    lat: np.ndarray,
    lon: np.ndarray,
    edges: Sequence[float],
    max_per_bin: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    flat_lat = lat2d.reshape(-1)
    flat_lon = lon2d.reshape(-1)
    ncell = flat_lat.size
    counts = np.zeros(len(edges) - 1, dtype=int)
    rows: list[dict[str, object]] = []
    attempts = 0
    maximum_attempts = max(100_000, int(max_per_bin) * len(counts) * 500)
    while np.any(counts < int(max_per_bin)) and attempts < maximum_attempts:
        batch = min(50_000, maximum_attempts - attempts)
        first = rng.integers(0, ncell, size=batch)
        second = rng.integers(0, ncell, size=batch)
        keep = first != second
        first = first[keep]
        second = second[keep]
        distance = _haversine_vectorized(
            flat_lat[first],
            flat_lon[first],
            flat_lat[second],
            flat_lon[second],
        )
        bins = np.searchsorted(np.asarray(edges), distance, side="right") - 1
        for bin_index in range(len(counts)):
            needed = int(max_per_bin) - counts[bin_index]
            selected = np.flatnonzero(bins == bin_index)[:needed]
            for index in selected:
                rows.append(
                    {
                        "cell_i": int(first[index]),
                        "cell_j": int(second[index]),
                        "distance_km": float(distance[index]),
                        "distance_bin_start_km": float(edges[bin_index]),
                        "distance_bin_end_km": float(edges[bin_index + 1]),
                    }
                )
            counts[bin_index] += len(selected)
        attempts += batch
    return pd.DataFrame(rows)


def _threshold_for_quantile(
    threshold_ds: xr.Dataset,
    dates: Sequence[pd.Timestamp],
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_values = threshold_ds["quantile"].values.astype(float)
    q_index = int(np.argmin(np.abs(q_values - float(quantile))))
    if not np.isclose(q_values[q_index], float(quantile)):
        raise ValueError(f"Frozen threshold artifact does not contain quantile {quantile}.")
    calendar = {str(value): index for index, value in enumerate(threshold_ds["calendar_day"].values)}
    indices = np.asarray([calendar[pd.Timestamp(day).strftime("%m-%d")] for day in dates], dtype=int)
    era = threshold_ds["era5_quantile_C"].isel(quantile=q_index, calendar_day=indices).values
    ace_group = threshold_ds["ace2_quantile_C"].isel(quantile=q_index, calendar_day=indices).values
    member_group = threshold_ds["member_lag_group"].values.astype(int)
    ace = np.stack(
        [ace_group[member_group[index]] for index in range(len(member_group))],
        axis=1,
    )
    return era, ace, member_group


def _bootstrap_year_means(
    values_by_year: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values_by_year, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    n_years = values.shape[0]
    if n_years == 0:
        return np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    estimates = np.empty((int(replicates), values.shape[1]), dtype=float)
    for replicate in range(int(replicates)):
        selected = rng.integers(0, n_years, size=n_years)
        sample = values[selected]
        for column in range(values.shape[1]):
            finite = sample[:, column][np.isfinite(sample[:, column])]
            estimates[replicate, column] = float(np.mean(finite)) if finite.size else np.nan
    lower = np.full(values.shape[1], np.nan, dtype=float)
    upper = np.full(values.shape[1], np.nan, dtype=float)
    for column in range(values.shape[1]):
        finite = estimates[:, column][np.isfinite(estimates[:, column])]
        if finite.size:
            lower[column] = np.quantile(finite, 0.025)
            upper[column] = np.quantile(finite, 0.975)
    return lower, upper


def _finite_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_row_means(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray([_finite_mean(row) for row in array], dtype=float)


def run_paper_metrics(
    test_years: Sequence[int],
    daily_paths: DailyFirstPaths,
    paths: MarkedPaths,
    args,
) -> dict[str, Path]:
    outputs = {
        "twcrps": paper_path(paths, args, "tail_weighted_crps"),
        "chi": paper_path(paths, args, "chi_distance"),
        "are": paper_path(paths, args, "are_threshold"),
        "qq": paper_path(paths, args, "upper_tail_qq"),
        "local_are": paths.paper / f"local_era5_are_{period_slug(args)}.nc",
    }
    if all(path.exists() for path in outputs.values()) and not args.force_marked:
        return outputs
    threshold_ds = xr.load_dataset(threshold_artifact_path(paths, args))
    lat = threshold_ds["lat"].values.astype(float)
    lon = threshold_ds["lon"].values.astype(float)
    area = cell_area_km2(lat, lon).reshape(-1)
    edges = args.chi_distance_bins_km_parsed
    rng = np.random.default_rng(int(args.random_seed) + 551)
    pairs = _sample_distance_pairs(
        lat,
        lon,
        edges,
        int(args.chi_max_pairs_per_bin),
        rng,
    )
    pair_i = pairs["cell_i"].to_numpy(dtype=int)
    pair_j = pairs["cell_j"].to_numpy(dtype=int)
    reference_count = min(int(args.are_max_reference_points), len(area))
    reference_cells = np.sort(
        rng.choice(len(area), size=reference_count, replace=False)
        if reference_count < len(area)
        else np.arange(len(area))
    )
    quantiles = list(args.paper_quantiles_parsed)
    chi_year: dict[tuple[str, float], list[np.ndarray]] = {
        (source, quantile): [] for source in ("era5", "ace2") for quantile in quantiles
    }
    are_year: dict[tuple[str, float], list[np.ndarray]] = {
        (source, quantile): [] for source in ("era5", "ace2") for quantile in quantiles
    }
    tw_rows: list[dict[str, object]] = []
    era_samples: list[np.ndarray] = []
    ace_samples: list[np.ndarray] = []
    local_quantile = float(args.local_are_quantile)
    local_joint = np.zeros((len(area), len(area)), dtype=np.float64)
    local_denom = np.zeros(len(area), dtype=np.float64)
    for year in test_years:
        cases = cases_for_years([int(year)], args, apply_debug_limit=True)
        era5, ace2, _, _, _ = load_period_fields(cases, daily_paths, args)
        dates = [case.date for case in cases]
        tail_era, _, _ = _threshold_for_quantile(
            threshold_ds,
            dates,
            float(args.twcrps_tail_quantile),
        )
        crps = tail_weighted_crps(
            np.moveaxis(ace2, 1, 0),
            era5,
            tail_era,
        )
        valid = np.isfinite(crps)
        area3 = np.broadcast_to(area.reshape(1, len(lat), len(lon)), crps.shape)
        tw_rows.append(
            {
                "year": int(year),
                "tail_quantile": float(args.twcrps_tail_quantile),
                "area_weighted_twcrps": (
                    float(np.nansum(crps * area3) / np.nansum(np.where(valid, area3, 0.0)))
                    if np.any(valid)
                    else np.nan
                ),
                "n_valid_grid_days": int(np.count_nonzero(valid)),
            }
        )
        flat_era = era5.reshape(-1)
        flat_ace = ace2.reshape(-1)
        finite_era = flat_era[np.isfinite(flat_era)]
        finite_ace = flat_ace[np.isfinite(flat_ace)]
        if finite_era.size:
            era_samples.append(
                rng.choice(finite_era, size=min(50_000, finite_era.size), replace=False)
            )
        if finite_ace.size:
            ace_samples.append(
                rng.choice(finite_ace, size=min(100_000, finite_ace.size), replace=False)
            )
        for quantile in quantiles:
            era_threshold, ace_threshold, _ = _threshold_for_quantile(
                threshold_ds,
                dates,
                quantile,
            )
            source_exceed = {
                "era5": (np.isfinite(era5) & (era5 > era_threshold)).reshape(-1, len(area)),
                "ace2": (
                    np.isfinite(ace2) & (ace2 > ace_threshold)
                ).reshape(-1, len(area)),
            }
            for source, exceed in source_exceed.items():
                denom = exceed.sum(axis=0).astype(float)
                both = (exceed[:, pair_i] & exceed[:, pair_j]).sum(axis=0).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    directional_chi = np.where(
                        denom[pair_i] > 0,
                        both / denom[pair_i],
                        np.nan,
                    )
                chi_year[(source, quantile)].append(directional_chi)
                are_values = np.full(len(reference_cells), np.nan, dtype=float)
                for ref_index, cell in enumerate(reference_cells):
                    ref_mask = exceed[:, cell]
                    n_ref = int(ref_mask.sum())
                    if n_ref:
                        joint_counts = exceed[ref_mask].sum(axis=0)
                        joint_area = float(np.sum(joint_counts * area))
                        are_values[ref_index] = math.sqrt(joint_area / (math.pi * n_ref))
                are_year[(source, quantile)].append(are_values)
                if source == "era5" and np.isclose(quantile, local_quantile):
                    numeric = exceed.astype(np.float32)
                    local_joint += numeric.T @ numeric
                    local_denom += numeric.sum(axis=0)
    chi_rows: list[dict[str, object]] = []
    for (source, quantile), yearly in chi_year.items():
        values = np.stack(yearly, axis=0)
        for (bin_start, bin_end), group in pairs.groupby(
            ["distance_bin_start_km", "distance_bin_end_km"]
        ):
            indices = group.index.to_numpy(dtype=int)
            yearly_mean = _finite_row_means(values[:, indices])
            lower, upper = _bootstrap_year_means(
                yearly_mean,
                replicates=int(args.bootstrap_replicates),
                rng=rng,
            )
            chi_rows.append(
                {
                    "source": source,
                    "quantile": quantile,
                    "distance_bin_start_km": bin_start,
                    "distance_bin_end_km": bin_end,
                    "mean_distance_km": group["distance_km"].mean(),
                    "chi_mean": _finite_mean(yearly_mean),
                    "chi_ci95_lower": float(lower[0]),
                    "chi_ci95_upper": float(upper[0]),
                    "n_pairs": len(indices),
                    "n_years": len(yearly_mean),
                }
            )
    are_rows: list[dict[str, object]] = []
    for (source, quantile), yearly in are_year.items():
        values = np.stack(yearly, axis=0)
        yearly_mean = _finite_row_means(values)
        lower, upper = _bootstrap_year_means(
            yearly_mean,
            replicates=int(args.bootstrap_replicates),
            rng=rng,
        )
        are_rows.append(
            {
                "source": source,
                "quantile": quantile,
                "are_km_mean": _finite_mean(yearly_mean),
                "are_km_ci95_lower": float(lower[0]),
                "are_km_ci95_upper": float(upper[0]),
                "n_reference_cells": len(reference_cells),
                "n_years": len(yearly_mean),
            }
        )
    qq_probabilities = np.unique(
        np.concatenate(
            [
                np.linspace(0.90, 0.99, 10),
                np.asarray([0.995, 0.9975, 0.999]),
            ]
        )
    )
    era_pool = np.concatenate(era_samples) if era_samples else np.asarray([])
    ace_pool = np.concatenate(ace_samples) if ace_samples else np.asarray([])
    qq_rows = [
        {
            "probability": probability,
            "era5_quantile_C": float(np.quantile(era_pool, probability)) if era_pool.size else np.nan,
            "ace2_quantile_C": float(np.quantile(ace_pool, probability)) if ace_pool.size else np.nan,
            "ace2_minus_era5_C": (
                float(np.quantile(ace_pool, probability) - np.quantile(era_pool, probability))
                if era_pool.size and ace_pool.size
                else np.nan
            ),
        }
        for probability in qq_probabilities
    ]
    pd.DataFrame(tw_rows).to_csv(outputs["twcrps"], index=False)
    pd.DataFrame(chi_rows).to_csv(outputs["chi"], index=False)
    pd.DataFrame(are_rows).to_csv(outputs["are"], index=False)
    pd.DataFrame(qq_rows).to_csv(outputs["qq"], index=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_area = np.where(
            local_denom > 0,
            np.sqrt(
                np.sum(local_joint * area[np.newaxis, :], axis=1)
                / (math.pi * local_denom)
            ),
            np.nan,
        )
    xr.Dataset(
        {
            "era5_local_are_km": (
                ("lat", "lon"),
                local_area.reshape(len(lat), len(lon)).astype(np.float32),
            )
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "quantile": local_quantile,
            "definition": "Area-weighted averaged radius of exceedance using frozen ERA5 fit-period margins.",
            "units": "km",
            "test_years": args.test_years,
        },
    ).to_netcdf(outputs["local_are"])
    threshold_ds.close()
    return outputs


def _format_map(ax, lon: np.ndarray, lat: np.ndarray) -> None:
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    try:
        draw_borders(ax, lon, lat)
    except Exception:
        pass


_ARE_GEOGRAPHY_CACHE: dict[str, list] | None = None


def _format_are_map(ax, lon: np.ndarray, lat: np.ndarray) -> None:
    """Format the Figure 6 map and overlay Natural Earth geographic context."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{abs(value):g}°W"
                if value < 0
                else (f"{value:g}°E" if value > 0 else "0°")
            )
        )
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{abs(value):g}°S"
                if value < 0
                else (f"{value:g}°N" if value > 0 else "0°")
            )
        )
    )
    ax.grid(color="white", linestyle=":", linewidth=0.55, alpha=0.55, zorder=3)

    global _ARE_GEOGRAPHY_CACHE
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
    except Exception:
        # The ordinary border helper remains a useful fallback when Cartopy is
        # unavailable in a lightweight plotting environment.
        try:
            draw_borders(ax, lon, lat, style="black")
        except Exception:
            pass
        return

    if _ARE_GEOGRAPHY_CACHE is None:
        layers = {
            "coastline": ("50m", "physical", "coastline"),
            "countries": ("50m", "cultural", "admin_0_boundary_lines_land"),
            "states": ("50m", "cultural", "admin_1_states_provinces_lakes"),
            "lakes": ("50m", "physical", "lakes"),
        }
        _ARE_GEOGRAPHY_CACHE = {}
        for name, specification in layers.items():
            try:
                path = shpreader.natural_earth(*specification)
                _ARE_GEOGRAPHY_CACHE[name] = list(
                    shpreader.Reader(path).geometries()
                )
            except Exception:
                _ARE_GEOGRAPHY_CACHE[name] = []

    viewport = box(
        float(np.nanmin(lon)) - 1.0,
        float(np.nanmin(lat)) - 1.0,
        float(np.nanmax(lon)) + 1.0,
        float(np.nanmax(lat)) + 1.0,
    )

    def plot_geometry(geometry, *, color: str, linewidth: float, alpha: float, zorder: int) -> None:
        if geometry is None or geometry.is_empty:
            return
        if hasattr(geometry, "geoms"):
            for component in geometry.geoms:
                plot_geometry(
                    component,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
            return
        if hasattr(geometry, "exterior"):
            x_values, y_values = geometry.exterior.xy
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
            for interior in geometry.interiors:
                x_values, y_values = interior.xy
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
            return
        if hasattr(geometry, "xy"):
            x_values, y_values = geometry.xy
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

    layer_styles = {
        "lakes": {"color": "#deebf2", "linewidth": 0.55, "alpha": 0.95, "zorder": 5},
        "states": {"color": "#404040", "linewidth": 0.32, "alpha": 0.72, "zorder": 6},
        "countries": {"color": "#151515", "linewidth": 0.85, "alpha": 0.98, "zorder": 7},
        "coastline": {"color": "#151515", "linewidth": 0.85, "alpha": 0.98, "zorder": 8},
    }
    for layer in ("lakes", "states", "countries", "coastline"):
        for geometry in _ARE_GEOGRAPHY_CACHE.get(layer, []):
            try:
                clipped = geometry.intersection(viewport)
            except Exception:
                continue
            plot_geometry(clipped, **layer_styles[layer])


def _placeholder(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_path(paths: MarkedPaths, args, number: int, stem: str) -> Path:
    return paths.figures / f"figure{number:02d}_{stem}_{period_slug(args)}.png"


def plot_threshold_bias(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 1, "threshold_model_bias")
    with xr.open_dataset(threshold_artifact_path(paths, args)) as ds:
        quantiles = ds["quantile"].values.astype(float)
        index = int(np.argmin(np.abs(quantiles - float(args.percentile) / 100.0)))
        era = ds["era5_quantile_C"].isel(quantile=index).mean("calendar_day").values
        ace = ds["ace2_quantile_C"].isel(lag_group=0, quantile=index).mean("calendar_day").values
        difference = ace - era
        mean_bias = ds["ace2_minus_era5_mean_bias_C"].values
        lat = ds["lat"].values
        lon = ds["lon"].values
    fields = [
        (era, "ERA5 frozen threshold", "coolwarm"),
        (ace, "ACE2 relative-track threshold", "coolwarm"),
        (difference, "ACE2 − ERA5 threshold", "RdBu_r"),
        (mean_bias, "ACE2 − ERA5 calibration mean", "RdBu_r"),
    ]
    temp_min = float(np.nanpercentile(np.concatenate([era.ravel(), ace.ravel()]), 2))
    temp_max = float(np.nanpercentile(np.concatenate([era.ravel(), ace.ravel()]), 98))
    bias_limit = max(
        0.1,
        float(np.nanpercentile(np.abs(np.concatenate([difference.ravel(), mean_bias.ravel()])), 98)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (field, title, cmap) in zip(axes.ravel(), fields):
        if "−" in title:
            mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=-bias_limit, vmax=bias_limit)
        else:
            mesh = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=temp_min, vmax=temp_max)
        axis.set_title(title, fontweight="bold")
        _format_map(axis, lon, lat)
        fig.colorbar(mesh, ax=axis, shrink=0.82, label="°C")
    fig.suptitle("Frozen fit-only thresholds: relative-extreme versus absolute-hazard evaluation")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _event_day_mask(event: Event, ordinal: int) -> np.ndarray:
    mask = np.zeros((event.nlat, event.nlon), dtype=bool)
    selected = event.voxels[event.voxels[:, 0] == int(ordinal), 1]
    if selected.size:
        mask.reshape(-1)[selected.astype(int)] = True
    return mask


def _event_footprint_mask(event: Event) -> np.ndarray:
    mask = np.zeros((event.nlat, event.nlon), dtype=bool)
    if event.voxels.size:
        mask.reshape(-1)[np.unique(event.voxels[:, 1].astype(int))] = True
    return mask


def _exact_spacetime_iou(first: Event, second: Event) -> float:
    ncell = max(first.nlat * first.nlon, second.nlat * second.nlon)
    first_keys = np.unique(first.voxels[:, 0].astype(np.int64) * ncell + first.voxels[:, 1])
    second_keys = np.unique(second.voxels[:, 0].astype(np.int64) * ncell + second.voxels[:, 1])
    intersection = np.intersect1d(first_keys, second_keys, assume_unique=True).size
    union = first_keys.size + second_keys.size - intersection
    return float(intersection / union) if union else 1.0


def _representative_event(
    events: Sequence[Event],
    matches: pd.DataFrame,
    paths: MarkedPaths,
    args,
    *,
    track: str = "relative",
) -> tuple[Event, pd.DataFrame]:
    observed = {
        event.event_id: event
        for event in events
        if event.source == "era5" and event.track == track
    }
    primary = matches[
        (matches["row_type"] == "match")
        & np.isclose(matches["match_radius_km"], float(args.primary_match_radius_km))
        & (matches["match_tolerance_days"] == int(args.primary_match_tolerance_days))
        & (matches["track"] == track)
    ]
    if not observed or primary.empty:
        raise ValueError(f"No matched ERA5 {track} event is available.")
    support_file = event_support_path(paths, args)
    support_by_id: dict[str, float] = {}
    if support_file.exists():
        support = pd.read_csv(support_file)
        support = support[
            np.isclose(support["match_radius_km"], float(args.primary_match_radius_km))
            & (support["match_tolerance_days"] == int(args.primary_match_tolerance_days))
            & (support["track"] == track)
        ]
        support_by_id = dict(
            zip(
                support["observed_event_id"].astype(str),
                pd.to_numeric(support["member_support_probability"], errors="coerce").fillna(0.0),
            )
        )
    candidate_ids = [
        event_id
        for event_id in primary["observed_event_id"].dropna().astype(str).unique()
        if event_id in observed
    ]
    scored: list[tuple[float, str]] = []
    for event_id in candidate_ids:
        event = observed[event_id]
        cells = event.voxels[:, 1].astype(int)
        rows, cols = np.divmod(cells, event.nlon)
        touches_boundary = bool(
            rows.min() == 0
            or rows.max() == event.nlat - 1
            or cols.min() == 0
            or cols.max() == event.nlon - 1
        )
        support_probability = support_by_id.get(
            event_id,
            float(primary["observed_event_id"].astype(str).eq(event_id).mean()),
        )
        size_score = math.log1p(max(0.0, event.spacetime_volume_km2_days))
        duration_score = math.sqrt(max(1, event.duration_days))
        boundary_factor = 0.2 if touches_boundary else 1.0
        scored.append(
            (
                boundary_factor
                * max(0.04, support_probability)
                * size_score
                * duration_score,
                event_id,
            )
        )
    _, chosen_id = max(scored)
    return observed[chosen_id], primary[primary["observed_event_id"].astype(str) == chosen_id]


def _crop_slices(mask: np.ndarray, padding: int = 2) -> tuple[slice, slice]:
    rows, cols = np.where(mask)
    if not rows.size:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    return (
        slice(max(0, int(rows.min()) - padding), min(mask.shape[0], int(rows.max()) + padding + 1)),
        slice(max(0, int(cols.min()) - padding), min(mask.shape[1], int(cols.max()) + padding + 1)),
    )


def plot_example_event(
    events: Sequence[Event],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> Path:
    output = _figure_path(paths, args, 2, "example_event_probability_matches")
    match_file = match_catalog_path(paths, args)
    if not match_file.exists():
        return _placeholder(output, "Example matched event", "Matching artifact is unavailable.")
    matches = pd.read_csv(match_file)
    try:
        observed, matched_rows = _representative_event(events, matches, paths, args)
    except ValueError as error:
        return _placeholder(output, "Example matched event", str(error))
    lookup = {event.event_id: event for event in events}
    probability = np.zeros((observed.nlat, observed.nlon), dtype=float)
    valid_frame = pd.read_csv(valid_members_path(paths, args)).set_index("year")
    valid_member_ids = _parse_json_ints(valid_frame.loc[observed.year, "valid_members"])
    for member in valid_member_ids:
        member_mask = np.zeros_like(probability, dtype=bool)
        member_rows = matched_rows[
            pd.to_numeric(matched_rows["member"], errors="coerce") == int(member)
        ]
        for forecast_id in member_rows["forecast_event_id"].dropna().astype(str):
            forecast = lookup.get(forecast_id)
            if forecast is not None:
                member_mask |= _event_footprint_mask(forecast)
        probability += member_mask
    denominator = max(1, len(valid_member_ids))
    probability /= denominator
    obs_mask = _event_footprint_mask(observed)
    display_support = probability > 0
    row_slice, col_slice = _crop_slices(obs_mask | display_support, padding=3)
    display_lat = lat[row_slice]
    display_lon = lon[col_slice]
    display_observed = obs_mask[row_slice, col_slice]
    display_probability = probability[row_slice, col_slice]
    false_rows = matches[
        (matches["row_type"] == "false_alarm")
        & (matches["track"] == "relative")
        & (matches["year"] == observed.year)
        & np.isclose(matches["match_radius_km"], float(args.primary_match_radius_km))
        & (matches["match_tolerance_days"] == int(args.primary_match_tolerance_days))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].pcolormesh(
        display_lon,
        display_lat,
        display_observed.astype(float),
        shading="auto",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[0].scatter(observed.centroid_lon, observed.centroid_lat, marker="*", s=90, color="black")
    axes[0].set_title(
        f"Representative ERA5 event footprint\n{observed.start_date} to {observed.end_date}",
        fontweight="bold",
    )
    _format_map(axes[0], display_lon, display_lat)
    mesh = axes[1].pcolormesh(
        display_lon,
        display_lat,
        display_probability,
        shading="auto",
        cmap="magma_r",
        vmin=0,
        vmax=1,
    )
    contour_levels = sorted(
        {
            float(token)
            for raw in args.probability_thresholds
            for token in str(raw).split(",")
            if token.strip()
            and 0.0 < float(token) < 1.0
            and np.nanmin(display_probability) < float(token) < np.nanmax(display_probability)
        }
    )
    if contour_levels:
        contour = axes[1].contour(
            display_lon,
            display_lat,
            display_probability,
            levels=contour_levels,
            colors="black",
            linewidths=0.8,
        )
        axes[1].clabel(contour, fmt=lambda value: f"p={value:g}", fontsize=7)
    if np.any(display_observed):
        axes[1].contour(
            display_lon,
            display_lat,
            display_observed.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1.6,
        )
    for forecast_id in matched_rows["forecast_event_id"].drop_duplicates():
        forecast = lookup.get(str(forecast_id))
        if forecast is None:
            continue
        axes[1].scatter(
            forecast.centroid_lon,
            forecast.centroid_lat,
            s=12,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.75,
        )
        axes[1].plot(
            [observed.centroid_lon, forecast.centroid_lon],
            [observed.centroid_lat, forecast.centroid_lat],
            color="white",
            linewidth=0.35,
            alpha=0.4,
        )
    false_events = [
        lookup.get(str(forecast_id))
        for forecast_id in false_rows["forecast_event_id"]
    ]
    false_events = [
        event
        for event in false_events
        if event is not None
        and abs(event.peak_ordinal - observed.peak_ordinal)
        <= int(args.primary_match_tolerance_days)
        and float(display_lon.min()) <= event.centroid_lon <= float(display_lon.max())
        and float(display_lat.min()) <= event.centroid_lat <= float(display_lat.max())
    ][:20]
    for false_event in false_events:
        axes[1].scatter(
            false_event.centroid_lon,
            false_event.centroid_lat,
            marker="x",
            s=22,
            color="deepskyblue",
        )
    median_distance = pd.to_numeric(matched_rows["centroid_distance_km"], errors="coerce").median()
    median_timing = pd.to_numeric(matched_rows["timing_error_days"], errors="coerce").median()
    axes[1].set_title(
        f"Matched-component support: {len(matched_rows)}/{denominator} members\n"
        f"median displacement {median_distance:.0f} km; timing error {median_timing:.1f} d",
        fontweight="bold",
    )
    _format_map(axes[1], display_lon, display_lat)
    fig.colorbar(mesh, ax=axes[1], label="matched-member footprint probability")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_event_tubes(
    events: Sequence[Event],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> Path:
    output = _figure_path(paths, args, 3, "matched_event_tubes")
    match_file = match_catalog_path(paths, args)
    if not match_file.exists():
        return _placeholder(output, "Matched event tubes", "Matching artifact is unavailable.")
    matches = pd.read_csv(match_file)
    lookup = {event.event_id: event for event in events}
    try:
        observed, selected = _representative_event(events, matches, paths, args)
    except ValueError as error:
        return _placeholder(output, "Matched event tubes", str(error))
    median_iou = pd.to_numeric(selected["tolerant_iou"], errors="coerce").median()
    median_distance = pd.to_numeric(selected["centroid_distance_km"], errors="coerce").median()
    rank_score = (
        (pd.to_numeric(selected["tolerant_iou"], errors="coerce") - median_iou).abs()
        + (
            pd.to_numeric(selected["centroid_distance_km"], errors="coerce")
            - median_distance
        ).abs()
        / max(1.0, float(args.primary_match_radius_km))
    )
    row = selected.loc[rank_score.idxmin()]
    observed = lookup[str(row["observed_event_id"])]
    forecast = lookup[str(row["forecast_event_id"])]
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    for event, color, label_text in (
        (observed, "#2166ac", "ERA5"),
        (forecast, "#d73027", f"ACE2 member {forecast.member}"),
    ):
        voxels = event.voxels
        if len(voxels) > 12_000:
            step = int(math.ceil(len(voxels) / 12_000))
            voxels = voxels[::step]
        cells = voxels[:, 1].astype(int)
        rows, cols = np.divmod(cells, event.nlon)
        axis.scatter(
            lon[cols],
            lat[rows],
            voxels[:, 0] - min(observed.start_ordinal, forecast.start_ordinal),
            s=7,
            alpha=0.28,
            color=color,
            label=label_text,
        )
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.set_zlabel("day from event-pair origin")
    axis.set_title(
        f"Representative one-to-one matched event tubes\n"
        f"raw IoU={_exact_spacetime_iou(observed, forecast):.2f}; "
        f"tolerance-adjusted IoU={float(row['tolerant_iou']):.2f}; "
        f"distance={float(row['centroid_distance_km']):.0f} km, "
        f"timing={float(row['timing_error_days']):.0f} d",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_tolerance_surfaces(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 4, "radius_time_tolerance_surfaces")
    frame = pd.read_csv(tolerance_surface_path(paths, args))
    if frame.empty:
        return _placeholder(output, "Tolerance surfaces", "No matching scores are available.")
    metrics = [
        ("observed_event_f1", "observed"),
        ("shuffled_null_event_f1", "circular-shift null"),
        ("observed_minus_null_event_f1", "observed − null"),
    ]
    raw_values = pd.to_numeric(
        frame[["observed_event_f1", "shuffled_null_event_f1"]].stack(),
        errors="coerce",
    ).to_numpy()
    score_max = max(0.05, float(np.nanmax(raw_values))) if np.isfinite(raw_values).any() else 1.0
    gain_values = pd.to_numeric(
        frame.get("observed_minus_null_event_f1", pd.Series(dtype=float)),
        errors="coerce",
    ).to_numpy()
    gain_limit = (
        max(0.02, float(np.nanmax(np.abs(gain_values))))
        if np.isfinite(gain_values).any()
        else 0.05
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    score_image = None
    gain_image = None
    for row_index, track in enumerate(("relative", "absolute")):
        subset = frame[frame["track"] == track]
        for axis, (metric, title) in zip(axes[row_index], metrics):
            if metric not in subset:
                axis.axis("off")
                continue
            pivot = subset.pivot(
                index="match_radius_km",
                columns="match_tolerance_days",
                values=metric,
            )
            values = pivot.values
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r" if "minus" in metric else "viridis",
                vmin=-gain_limit if "minus" in metric else 0,
                vmax=gain_limit if "minus" in metric else score_max,
            )
            if "minus" in metric:
                gain_image = image
            else:
                score_image = image
            for y_index in range(values.shape[0]):
                for x_index in range(values.shape[1]):
                    value = values[y_index, x_index]
                    if np.isfinite(value):
                        axis.text(
                            x_index,
                            y_index,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=(
                                "white"
                                if (
                                    ("minus" not in metric and value > 0.55 * score_max)
                                    or ("minus" in metric and abs(value) > 0.55 * gain_limit)
                                )
                                else "black"
                            ),
                        )
            axis.set_xticks(np.arange(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
            axis.set_yticks(np.arange(len(pivot.index)), labels=[f"{value:g}" for value in pivot.index])
            axis.set_xlabel("temporal tolerance ± days")
            axis.set_ylabel("spatial gate (km)")
            axis.set_title(f"{track}: {title}", fontweight="bold")
            radius_values = np.asarray(pivot.index, dtype=float)
            tolerance_values = np.asarray(pivot.columns, dtype=float)
            radius_index = np.flatnonzero(
                np.isclose(radius_values, float(args.primary_match_radius_km))
            )
            tolerance_index = np.flatnonzero(
                np.isclose(tolerance_values, int(args.primary_match_tolerance_days))
            )
            if radius_index.size and tolerance_index.size:
                axis.scatter(
                    tolerance_index[0],
                    radius_index[0],
                    marker="s",
                    s=270,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.6,
                )
    if score_image is not None:
        fig.colorbar(
            score_image,
            ax=axes[:, :2],
            shrink=0.82,
            label="memberwise event F1",
        )
    if gain_image is not None:
        fig.colorbar(
            gain_image,
            ax=axes[:, 2],
            shrink=0.82,
            label="event F1 gain over circular-shift null",
        )
    fig.suptitle("Sensitivity to predeclared spatial and temporal matching tolerances")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_chi(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 5, "chi_distance_curves")
    file = paper_path(paths, args, "chi_distance")
    if not file.exists():
        return _placeholder(output, "Tail dependence χ", "Paper-metrics stage was not run.")
    frame = pd.read_csv(file)
    quantiles = sorted(pd.to_numeric(frame["quantile"], errors="coerce").dropna().unique())
    fig, axes = plt.subplots(
        1,
        len(quantiles),
        figsize=(4.7 * len(quantiles), 4.6),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    styles = {"era5": ("#d73027", "o", "ERA5"), "ace2": ("#2166ac", "s", "ACE2")}
    for axis, quantile in zip(axes, quantiles):
        for source in ("era5", "ace2"):
            color, marker, label = styles[source]
            group = frame[
                (frame["source"] == source)
                & np.isclose(pd.to_numeric(frame["quantile"], errors="coerce"), quantile)
            ].sort_values("mean_distance_km")
            if group.empty:
                continue
            axis.plot(
                group["mean_distance_km"],
                group["chi_mean"],
                marker=marker,
                color=color,
                label=label,
            )
            axis.fill_between(
                group["mean_distance_km"],
                group["chi_ci95_lower"],
                group["chi_ci95_upper"],
                color=color,
                alpha=0.18,
            )
        axis.set_title(f"frozen marginal u={quantile:g}", fontweight="bold")
        axis.set_xlabel("great-circle separation (km)")
        axis.grid(alpha=0.3)
        axis.legend(frameon=False)
    axes[0].set_ylabel("tail-dependence coefficient χ")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_are(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 6, "are_curves_local_map")
    curve_file = paper_path(paths, args, "are_threshold")
    map_file = paths.paper / f"local_era5_are_{period_slug(args)}.nc"
    if not curve_file.exists() or not map_file.exists():
        return _placeholder(output, "Averaged Radius of Exceedance", "Paper-metrics stage was not run.")
    frame = pd.read_csv(curve_file)
    with xr.open_dataset(map_file) as ds:
        local = ds["era5_local_are_km"].values
        lat = ds["lat"].values
        lon = ds["lon"].values
    plot_lon = lon_for_plot(np.asarray(lon, dtype=float))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for source, group in frame.groupby("source"):
        group = group.sort_values("quantile")
        axes[0].plot(group["quantile"], group["are_km_mean"], marker="o", label=source.upper())
        axes[0].fill_between(
            group["quantile"],
            group["are_km_ci95_lower"],
            group["are_km_ci95_upper"],
            alpha=0.18,
        )
    axes[0].set_xlabel("frozen marginal quantile u")
    axes[0].set_ylabel("conditional co-exceedance equivalent radius (km)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)
    axes[0].set_title("Domain-average dependence scale", fontweight="bold")
    mesh = axes[1].pcolormesh(
        plot_lon,
        lat,
        local,
        shading="auto",
        cmap="viridis",
        rasterized=True,
        zorder=1,
    )
    axes[1].set_title(
        f"Local ERA5 dependence scale at u={float(args.local_are_quantile):g}",
        fontweight="bold",
    )
    _format_are_map(axes[1], plot_lon, lat)
    fig.colorbar(mesh, ax=axes[1], label="km")
    fig.suptitle("ARE measures spatial tail dependence, not individual event size")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_counts_arrivals(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 7, "count_arrival_distributions")
    scores = pd.read_csv(process_scores_path(paths, args))
    curves = pd.read_csv(arrival_curves_path(paths, args))
    if scores.empty:
        return _placeholder(output, "Counts and arrival", "No fixed region-time units are available.")
    probability = pd.to_numeric(scores["ace2_occurrence_probability"], errors="coerce")
    candidates = scores[
        probability.between(0.05, 0.95)
        & (pd.to_numeric(scores["era5_occurrence"], errors="coerce") == 1)
    ].copy()
    if candidates.empty:
        candidates = scores[probability.between(0.0, 1.0, inclusive="neither")].copy()
    if candidates.empty:
        candidates = scores.copy()
    candidate_probability = pd.to_numeric(
        candidates["ace2_occurrence_probability"],
        errors="coerce",
    )
    candidate_variance = pd.to_numeric(
        candidates["ace2_count_variance"],
        errors="coerce",
    ).fillna(0.0)
    candidates["_selection_score"] = (
        (candidate_probability - 0.5).abs()
        - 0.02 * np.log1p(candidate_variance)
    )
    case = candidates.sort_values("_selection_score").iloc[0]
    counts = np.asarray(json.loads(case["member_event_counts"]), dtype=float)
    selected_curves = curves[
        (curves["year"] == case["year"])
        & (curves["track"] == case["track"])
        & (curves["region_id"] == case["region_id"])
        & (curves["window"] == case["window"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    bins = np.arange(-0.5, max(2, int(np.max(counts, initial=0)) + 1.5), 1)
    axes[0].hist(counts, bins=bins, color="#74add1", edgecolor="white")
    axes[0].axvline(case["era5_event_count"], color="#d73027", linewidth=2, label="ERA5")
    axes[0].set_xlabel("member event count")
    axes[0].set_ylabel("members")
    axes[0].legend(frameon=False)
    selected_curves = selected_curves.sort_values("day")
    curve_days = np.r_[-1.0, selected_curves["day"].to_numpy(dtype=float)]
    curve_survival = np.r_[
        1.0,
        selected_curves["survival_probability"].to_numpy(dtype=float),
    ]
    axes[1].step(
        curve_days,
        curve_survival,
        where="post",
        color="#2166ac",
        label="ACE2 survival",
    )
    if np.isfinite(case["era5_first_arrival_day"]):
        axes[1].axvline(case["era5_first_arrival_day"], color="#d73027", label="ERA5 arrival")
    axes[1].set_xlabel("day in fixed window")
    axes[1].set_ylabel("P(no arrival yet)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)
    region_label = (
        str(case["region_label"])
        if "region_label" in case.index
        else f"region {int(case['region_id'])}"
    )
    fig.suptitle(
        f"Representative fixed unit: {case['track']} track, {region_label}, "
        f"{case['window']} {int(case['year'])}\n"
        f"count CRPS={float(case['count_crps']):.2f}; "
        f"occurrence Brier={float(case['occurrence_brier_score']):.3f}",
        fontweight="bold",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_reliability(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 8, "reliability_brier")
    frame = pd.read_csv(reliability_path(paths, args))
    frame = frame[frame["track"].astype(str).str.lower() == "relative"].copy()
    if frame.empty:
        return _placeholder(
            output,
            "Relative-track reliability",
            "No relative-track fixed region-time units are available.",
        )
    scores = pd.read_csv(process_scores_path(paths, args))
    scores = scores[scores["track"].astype(str).str.lower() == "relative"].copy()
    if scores.empty:
        return _placeholder(
            output,
            "Relative-track reliability",
            "No relative-track process scores are available.",
        )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.5")
    probability_values = pd.to_numeric(
        scores["ace2_occurrence_probability"],
        errors="coerce",
    ).dropna()
    informative = (
        probability_values.nunique() >= 3
        and float(probability_values.max() - probability_values.min()) >= 0.05
    )
    group = frame.sort_values("mean_forecast_probability")
    axes[0].plot(
        group["mean_forecast_probability"],
        group["observed_frequency"],
        marker="o",
        color="#2166ac",
        label="relative track",
    )
    for row in group.itertuples():
        axes[0].annotate(
            f"n={int(row.n_region_time_units)}",
            (row.mean_forecast_probability, row.observed_frequency),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_xlabel("ACE2 forecast occurrence probability")
    axes[0].set_ylabel("ERA5 observed occurrence frequency")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)
    if not informative:
        axes[0].text(
            0.5,
            0.12,
            "Occurrence probabilities remain nearly constant.\n"
            "Use smaller process blocks or shorter windows.",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ace_brier = pd.to_numeric(
        scores["occurrence_brier_score"], errors="coerce"
    ).mean()
    climatology_brier = pd.to_numeric(
        scores["loyo_era5_occurrence_brier"], errors="coerce"
    ).mean()
    brier_values = [ace_brier, climatology_brier]
    bars = axes[1].bar(
        ["ACE2\nensemble", "LOYO ERA5\nclimatology"],
        brier_values,
        color=["#2166ac", "0.65"],
        width=0.62,
    )
    axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].set_ylabel("mean occurrence Brier score (lower is better)")
    axes[1].set_title("Relative-track aggregate probability error", fontweight="bold")
    brier_max = float(np.nanmax(brier_values))
    if np.isfinite(brier_max) and brier_max > 0:
        axes[1].set_ylim(0, brier_max * 1.4)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Relative-extreme occurrence reliability over fixed "
        f"{getattr(args, 'process_windows', 'weekly')} "
        f"{resolved_process_region_mode(args)} units"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_event_biases(events: Sequence[Event], paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 9, "event_bias_summaries")
    match_file = match_catalog_path(paths, args)
    if not match_file.exists():
        return _placeholder(output, "Event biases", "Matching artifact is unavailable.")
    matches = pd.read_csv(match_file)
    selected = matches[
        (matches["row_type"] == "match")
        & np.isclose(matches["match_radius_km"], float(args.primary_match_radius_km))
        & (matches["match_tolerance_days"] == int(args.primary_match_tolerance_days))
    ]
    if selected.empty:
        return _placeholder(output, "Event biases", "No valid event match was found.")
    lookup = {event.event_id: event for event in events}
    process = pd.read_csv(process_scores_path(paths, args))
    track_order = ("relative", "absolute")
    colors = {"relative": "#2166ac", "absolute": "#b2182b"}

    def matched_values(track: str, column: str) -> np.ndarray:
        return pd.to_numeric(
            selected.loc[selected["track"] == track, column],
            errors="coerce",
        ).to_numpy(dtype=float)

    def event_ratio_values(track: str, attribute: str) -> np.ndarray:
        values: list[float] = []
        for row in selected[selected["track"] == track].itertuples():
            forecast = lookup.get(str(row.forecast_event_id))
            observed = lookup.get(str(row.observed_event_id))
            if forecast is None or observed is None:
                continue
            forecast_value = float(getattr(forecast, attribute))
            observed_value = float(getattr(observed, attribute))
            if forecast_value > 0 and observed_value > 0:
                values.append(math.log(forecast_value / observed_value))
        return np.asarray(values, dtype=float)

    def intensity_values(track: str) -> np.ndarray:
        values: list[float] = []
        for row in selected[selected["track"] == track].itertuples():
            forecast = lookup.get(str(row.forecast_event_id))
            observed = lookup.get(str(row.observed_event_id))
            if forecast is not None and observed is not None:
                values.append(
                    forecast.standardized_quantile_intensity
                    - observed.standardized_quantile_intensity
                )
        return np.asarray(values, dtype=float)

    metrics = [
        (
            "count bias per fixed unit",
            lambda track: pd.to_numeric(
                process.loc[process["track"] == track, "count_bias"],
                errors="coerce",
            ).to_numpy(dtype=float),
            0.0,
        ),
        ("timing error (days)", lambda track: matched_values(track, "timing_error_days"), 0.0),
        (
            "centroid displacement (km)",
            lambda track: matched_values(track, "centroid_distance_km"),
            0.0,
        ),
        (
            "tolerance-adjusted IoU",
            lambda track: matched_values(track, "tolerant_iou"),
            1.0,
        ),
        (
            "duration log-ratio",
            lambda track: event_ratio_values(track, "duration_days"),
            0.0,
        ),
        (
            "max-area log-ratio",
            lambda track: event_ratio_values(track, "max_daily_area_km2"),
            0.0,
        ),
        ("intensity difference", intensity_values, 0.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    for axis, (name, value_function, ideal) in zip(axes.ravel(), metrics):
        for track in track_order:
            values = value_function(track)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            if np.nanmax(values) > np.nanmin(values):
                axis.hist(
                    values,
                    bins=24,
                    density=True,
                    histtype="step",
                    linewidth=1.7,
                    color=colors[track],
                    label=track,
                )
            else:
                axis.axvline(
                    values[0],
                    color=colors[track],
                    linewidth=1.7,
                    label=f"{track} (constant)",
                )
        axis.axvline(ideal, color="black", linewidth=1, linestyle="--")
        axis.set_title(name, fontweight="bold")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes.ravel()[len(metrics) :]:
        axis.axis("off")
    fig.suptitle(
        "Relative and absolute tracks kept separate; dashed line is the ideal value"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_track_conclusions(paths: MarkedPaths, args) -> Path:
    output = _figure_path(paths, args, 10, "relative_absolute_conclusions")
    member = pd.read_csv(member_scores_path(paths, args))
    process = pd.read_csv(process_scores_path(paths, args))
    selected = member[
        (member["score_type"] == "member")
        & np.isclose(member["match_radius_km"], float(args.primary_match_radius_km))
        & (member["match_tolerance_days"] == int(args.primary_match_tolerance_days))
    ]
    track_order = ["relative", "absolute"]
    colors = ["#2166ac", "#b2182b"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
    f1_values = [
        pd.to_numeric(
            selected.loc[selected["track"] == track, "event_f1"],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    f1_bars = axes[0, 0].bar(track_order, f1_values, color=colors)
    axes[0, 0].bar_label(f1_bars, fmt="%.3f", padding=2)
    axes[0, 0].set_ylabel("mean member event F1 (higher is better)")
    axes[0, 0].set_title("One-to-one event detection", fontweight="bold")

    surface = pd.read_csv(tolerance_surface_path(paths, args))
    primary_surface = surface[
        np.isclose(surface["match_radius_km"], float(args.primary_match_radius_km))
        & (
            pd.to_numeric(surface["match_tolerance_days"], errors="coerce")
            == int(args.primary_match_tolerance_days)
        )
    ]
    gain_values = [
        pd.to_numeric(
            primary_surface.loc[
                primary_surface["track"] == track,
                "observed_minus_null_event_f1",
            ],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    gain_bars = axes[0, 1].bar(track_order, gain_values, color=colors)
    axes[0, 1].bar_label(gain_bars, fmt="%+.3f", padding=2)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("event F1 minus circular-shift null")
    axes[0, 1].set_title("Skill beyond temporal coincidence", fontweight="bold")
    finite_gain = np.asarray(gain_values, dtype=float)
    if np.isfinite(finite_gain).any():
        lower = min(0.0, float(np.nanmin(finite_gain)) * 1.25)
        upper = max(0.01, float(np.nanmax(finite_gain)) * 1.18)
        axes[0, 1].set_ylim(lower, upper)

    x = np.arange(len(track_order), dtype=float)
    width = 0.36
    occurrence_model = [
        pd.to_numeric(
            process.loc[process["track"] == track, "occurrence_brier_score"],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    occurrence_reference = [
        pd.to_numeric(
            process.loc[process["track"] == track, "loyo_era5_occurrence_brier"],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    occurrence_first = axes[1, 0].bar(
        x - width / 2,
        occurrence_model,
        width,
        color="#2166ac",
        label="ACE2 ensemble",
    )
    occurrence_second = axes[1, 0].bar(
        x + width / 2,
        occurrence_reference,
        width,
        color="0.65",
        label="LOYO ERA5 climatology",
    )
    axes[1, 0].bar_label(occurrence_first, fmt="%.3f", padding=2, fontsize=8)
    axes[1, 0].bar_label(occurrence_second, fmt="%.3f", padding=2, fontsize=8)
    axes[1, 0].set_xticks(x, track_order)
    axes[1, 0].set_ylabel("occurrence Brier score (lower is better)")
    axes[1, 0].set_title("Fixed-unit occurrence probability", fontweight="bold")
    occurrence_max = float(np.nanmax(occurrence_model + occurrence_reference))
    if np.isfinite(occurrence_max) and occurrence_max > 0:
        axes[1, 0].set_ylim(0, occurrence_max * 1.4)
    axes[1, 0].legend(frameon=False)

    count_model = [
        pd.to_numeric(
            process.loc[process["track"] == track, "count_crps"],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    count_reference = [
        pd.to_numeric(
            process.loc[process["track"] == track, "loyo_era5_count_crps"],
            errors="coerce",
        ).mean()
        for track in track_order
    ]
    count_first = axes[1, 1].bar(
        x - width / 2,
        count_model,
        width,
        color="#2166ac",
        label="ACE2 ensemble",
    )
    count_second = axes[1, 1].bar(
        x + width / 2,
        count_reference,
        width,
        color="0.65",
        label="LOYO ERA5 climatology",
    )
    axes[1, 1].bar_label(count_first, fmt="%.2f", padding=2, fontsize=8)
    axes[1, 1].bar_label(count_second, fmt="%.2f", padding=2, fontsize=8)
    axes[1, 1].set_xticks(x, track_order)
    axes[1, 1].set_ylabel("count CRPS (lower is better)")
    axes[1, 1].set_title("Fixed-unit event counts", fontweight="bold")
    count_max = float(np.nanmax(count_model + count_reference))
    if np.isfinite(count_max) and count_max > 0:
        axes[1, 1].set_ylim(0, count_max * 1.4)
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Relative-extreme structure versus ERA5-defined absolute hazard\n"
        f"primary gate: {float(args.primary_match_radius_km):g} km, "
        f"±{int(args.primary_match_tolerance_days)} days"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def run_figures(
    events: Sequence[Event],
    lat: np.ndarray,
    lon: np.ndarray,
    paths: MarkedPaths,
    args,
) -> list[Path]:
    figures = [
        plot_threshold_bias(paths, args),
        plot_example_event(events, lat, lon, paths, args),
        plot_event_tubes(events, lat, lon, paths, args),
        plot_tolerance_surfaces(paths, args),
        plot_chi(paths, args),
        plot_are(paths, args),
        plot_counts_arrivals(paths, args),
        plot_reliability(paths, args),
        plot_event_biases(events, paths, args),
        plot_track_conclusions(paths, args),
    ]
    captions = [
        "1. **Frozen thresholds and model bias.** Fit-period ERA5 and pooled-member ACE2 calendar-window thresholds; the difference maps separate relative-extreme normalization from ERA5-defined absolute hazard.",
        "2. **Representative event and matched support.** A substantive interior ERA5 event is selected by a predeclared size/support rule. The probability field contains only the one-to-one ACE2 components matched to that event; cyan outlines ERA5, open white points are matched-component centroids, and blue crosses are nearby false alarms.",
        "3. **Representative matched event tubes.** The same ERA5 event is paired with a member near the median match quality. Raw and tolerance-adjusted IoU are reported separately so a tolerance-induced perfect score cannot be mistaken for exact overlap.",
        "4. **Space-time tolerance.** Observed memberwise event F1, circular-shift null F1, and their difference use shared color scales. Numbers are cell values and the white square marks the primary gate.",
        "5. **Tail dependence.** Each quantile panel overlays ERA5 and ACE2 χ(distance,u) using frozen source-specific fit-period margins and complete-year uncertainty.",
        "6. **Averaged Radius of Exceedance.** ARE is a conditional co-exceedance dependence scale, not the physical radius of an individual event; it uses spherical cell areas and complete-year uncertainty.",
        "7. **Counts and arrivals.** A representative, non-saturated fixed unit is selected near 50% ACE2 occurrence probability. Its member count distribution and right-censored first-arrival survival are compared with ERA5.",
        "8. **Relative-track reliability and Brier diagnostics.** Reliability uses predeclared weekly spatial units, retains ERA5 no-event units, labels bin sample sizes, and compares ACE2 Brier error with leave-one-year-out ERA5 climatology.",
        "9. **Track-separated biases.** Relative and absolute distributions are overlaid without pooling their different scientific targets. Dashed lines mark ideal values.",
        "10. **Track conclusions.** One-to-one F1 and gain over the circular-shift null are shown alongside occurrence and count scores versus leave-one-year-out ERA5 climatology.",
    ]
    captions_path(paths, args).write_text("# Workflow 5 figure captions\n\n" + "\n\n".join(captions) + "\n")
    return figures


def write_methods(paths: MarkedPaths, args, region_metadata: dict[str, object]) -> Path:
    output = methods_path(paths, args)
    text = f"""# Workflow 5: marked spatiotemporal extreme-event evaluation

## Scientific object

ERA5 supplies one marked event sequence and each ACE2 member supplies one
stochastic sequence. Members are retained through thresholding, component
labeling, matching, count scoring, and arrival scoring. No ensemble mean is
thresholded. The best-member result is written only as
`oracle_best_member_upper_bound` and is excluded from primary summaries.

## Period isolation

- Threshold fit: `{args.fit_years}`
- Event-filter validation: `{args.validation_years}`
- Reserved evaluation: `{args.test_years}`

The periods are checked for overlap before any field is opened. Frozen threshold
metadata records all three periods and states that only fit-period files were
read. Thresholds use a ±{int(args.calendar_window_days)}-day calendar window,
q={float(args.percentile) / 100.0:g}, and comparison `{args.comparison_operator}`.
The analyzed field is `{args.field_variant}`; the source Workflow 3 Gaussian
sigma is {float(args.gaussian_sigma):g} grid cells. The unsmoothed sensitivity
uses the cached unsmoothed daily-Tmax variable and refits all thresholds.

ERA5 thresholds are fit from ERA5 only. ACE2 relative-track thresholds pool
members within lag groups of size {int(args.lag_group_size)} (zero means one
all-member pool). The absolute-hazard track applies the frozen ERA5 physical
threshold to ERA5 and every ACE2 member. The threshold artifact also contains
ACE2-minus-ERA5 threshold, mean-bias, and upper-quantile maps.
Optional POT/GPD fitting is `{bool(args.fit_gpd_pot)}`. When enabled, positive
daily excesses are runs-declustered with a {int(args.gpd_run_length_days)}-day
run and require {int(args.gpd_min_peaks)} peaks; GEV is never fit to daily
exceedances.

For calendar day `d`, location `x`, source `s`, and fit-only window `W_d`,

```text
q_s(d,x) = empirical Quantile_q {{T_s(t,x): t in fit, calendar(t) in W_d}}.
```

The relative masks are `O=1{{T_ERA5>q_ERA5}}` and
`F_m=1{{T_ACE2,m>q_ACE2}}`; the absolute masks use `q_ERA5` for both.
Available JJA days define the calendar window, so windows at the cached seasonal
edges are truncated rather than borrowing unavailable May/September values.

## Event extraction

For each year and track, 3D connected components are labeled in
time-latitude-longitude using {int(args.connectivity)}-connectivity. Components
must last at least {int(args.min_duration_days)} day(s) and attain at least
{float(args.min_event_area_km2):g} km² daily area. These settings are
predeclared/validation settings and are never optimized on test scores.

For every component the catalog stores source, member, start/peak/end, spherical
area-weighted centroid, a compact ordinal/cell run-length mask, duration,
daily/maximum/footprint area, space-time volume, peak and mean temperature,
threshold excess, standardized tail intensity, year, and frozen-region
assignment. Grid-cell areas are spherical and expressed in km².

Regionalization metadata: `{json.dumps(region_metadata, sort_keys=True)}`.
When no validated frozen ERA5 region map is supplied, region 0 is the
predeclared full evaluation domain; an unfrozen diagnostic candidate is
rejected.

## One-to-one stochastic matching

Matching is performed independently for each member, year, and track using the
Hungarian assignment. Candidate pairs must pass hard gates in centroid distance
`r` and peak-time difference `tau`. For a candidate pair,

```text
C = w_t |Δt|/tau + w_r d/r + w_IoU (1-IoU*)
    + w_D |log(D_f/D_o)| + w_A |log(A_f/A_o)|
    + w_J |J_f-J_o|
```

where `IoU*` is the maximum symmetric area-weighted space-time IoU over allowed
temporal shifts. At each spatial tolerance, the two directional hit volumes
(ERA5 voxels inside the ACE2 dilation and ACE2 voxels inside the ERA5 dilation)
are calculated; their smaller value is the tolerant intersection, and the
denominator is the union of the original event volumes. `J` is standardized
tail intensity. Weights are timing={float(args.match_weight_timing):g},
distance={float(args.match_weight_distance):g},
IoU={float(args.match_weight_iou):g},
duration={float(args.match_weight_duration):g},
area={float(args.match_weight_area):g}, and
intensity={float(args.match_weight_intensity):g}. One forecast event cannot
match more than one ERA5 event. Unmatched ERA5 events are misses and unmatched
ACE2 events are false alarms.

For ERA5 event `i`, member support is

```text
P_i(r,tau) = sum_m 1{{member m has a valid one-to-one match to i}} / M_valid.
```

The denominator contains valid members only and is never divided by cells or
days. Pixel/neighborhood coverage and Fractions Skill Score are separate
diagnostics and are not called event probability.

## Counts and first arrivals

For every predeclared `{args.process_windows}` window and
`{resolved_process_region_mode(args)}` process region, the member count sample
is retained. Without a supplied frozen region map, the default process units are
regular {float(getattr(args, "process_grid_degrees", 10.0)):g}° latitude-
longitude blocks assigned by event centroid. These scoring units are declared
before evaluation and do not alter the event extraction or matching artifacts.
Weekly scoring uses complete seven-day windows; a final shorter remainder is
included only when `--include-partial-final-week` is requested.
Count CRPS is

```text
CRPS = mean_m |N_m-N_obs| - 0.5 mean_(m,m') |N_m-N_m'|.
```

The outputs also include occurrence Brier score, count/rate bias, variance to
mean ratio, and a method-of-moments negative-binomial dispersion diagnostic.
Poisson behavior is not assumed.

Members without an event are right-censored after the end of the window.
First-arrival CRPS, median timing error, probability of arrival by each day,
integrated Brier score, and survival curves retain those members. Reliability
uses every region-time unit, including ERA5 no-event units. ERA5 event
climatology is leave-one-year-out and uses the identical frozen event
definition.

## Baselines and uncertainty

The circular-shift null moves each complete member path within its season,
preserving component masks, marks, counts, and within-path serial structure.
Shifts closer than {int(args.minimum_null_shift_days)} days to zero modulo the
season are excluded. Real and shifted catalogs use identical matching and
scoring. Null results are a significance reference, not artificial calibration
negatives.

Complete years are resampled with replacement for {int(args.bootstrap_replicates)}
replicates. Every event and member belonging to a sampled year stays together.
The interval table reports 90% and 95% intervals where requested. Random seed:
{int(args.random_seed)}.

When requested, {int(args.independence_surrogate_replicates)} cellwise
independence replicates independently circular-shift each cell's binary event
path. This preserves each cell/member marginal event-day count while disrupting
cross-cell dependence; it is written as a separate optional null type.

## Tail diagnostics

Tail-weighted CRPS uses the upper-tail censoring transform
`v(z)=max(z,u)`. Dependence comparisons use the frozen ERA5 marginal quantiles
for ERA5 and frozen ACE2 marginal quantiles for ACE2. For locations `i,j`,

```text
chi_ij(u) = P(U_j > u | U_i > u).
```

Distance curves use great-circle kilometers and complete-year uncertainty.
Area-weighted ARE is

```text
ARE_i(u) = sqrt[ sum_j area_j P(U_j>u, U_i>u) /
                 (pi P(U_i>u)) ].
```

The paper-metrics directory also contains upper-tail Q-Q diagnostics and a
local ERA5 ARE map. GEV is not fit to daily exceedances; the repository's
Stage-B hierarchy reserves GEV for seasonal block maxima and GPD/POT for
declustered threshold excesses.

## Artifacts and interpretation

CSV catalogs use `ordinal_cell_rle_v1` masks. NetCDF thresholds include units,
coordinates, fit/test periods, comparison operator, smoothing choice, pooling,
and the leakage guard. The manifest records every artifact and configuration.
Relative-track results diagnose timing, location, dependence, and structure
after marginal normalization. Absolute-track results diagnose ERA5-defined
physical hazard and expose ACE2 marginal bias.

## Remaining limitations

- The workflow starts from Workflow 3 daily caches; it verifies their file
  layout but does not reread the raw six-hourly archive.
- If no validated frozen Stage-B map is supplied, only the predeclared full
  domain is scored regionally. Diagnostic/unfrozen candidate maps are rejected.
- χ uses at most {int(args.chi_max_pairs_per_bin)} sampled pairs per distance
  bin and ARE uses at most {int(args.are_max_reference_points)} reference cells;
  those Monte Carlo limits are recorded in the manifest and should receive a
  sensitivity run for a paper.
- The optional GPD output is diagnostic. Operational event thresholds remain
  empirical frozen quantiles; no short-record return level replaces them.
- The fit-period quantile pool is loaded into memory, while evaluation and
  scoring proceed one year at a time. This avoids loading the full multi-period
  archive eagerly but still requires memory proportional to the calibration
  pool.
"""
    output.write_text(text)
    return output


def write_manifest(
    paths: MarkedPaths,
    args,
    region_metadata: dict[str, object],
    artifacts: dict[str, object],
) -> Path:
    output = manifest_path(paths, args)
    with xr.open_dataset(threshold_artifact_path(paths, args)) as threshold_ds:
        lat = threshold_ds["lat"].values.astype(float)
        lon = threshold_ds["lon"].values.astype(float)
    manifest = {
        "workflow": "Workflow 5 full marked spatiotemporal extreme-event evaluation",
        "input_root": str(args.daily_first_root),
        "output_root": str(paths.root),
        "periods": {
            "fit_years": args.fit_years,
            "validation_years": args.validation_years,
            "test_years": args.test_years,
        },
        "thresholds": {
            "percentile": float(args.percentile),
            "calendar_window_half_width_days": int(args.calendar_window_days),
            "comparison_operator": args.comparison_operator,
            "field_variant": args.field_variant,
            "gaussian_sigma_grid_cells": float(args.gaussian_sigma),
            "lag_group_size": int(args.lag_group_size),
            "gpd_pot": {
                "enabled": bool(args.fit_gpd_pot),
                "run_length_days": int(args.gpd_run_length_days),
                "minimum_peaks": int(args.gpd_min_peaks),
                "gev_on_daily_exceedances": False,
            },
            "tracks": {
                "relative": "ERA5 frozen ERA5 threshold; ACE2 frozen pooled-member ACE2 threshold",
                "absolute": "ERA5 and every ACE2 member use the frozen ERA5 physical threshold",
            },
        },
        "grid": {
            "latitude_size": int(len(lat)),
            "longitude_size": int(len(lon)),
            "latitude_bounds_degrees_north": [float(np.min(lat)), float(np.max(lat))],
            "longitude_bounds_degrees_east": [float(np.min(lon)), float(np.max(lon))],
            "temperature_units": "degC",
            "distance_units": "km",
            "area_units": "km2",
            "spacetime_volume_units": "km2 days",
        },
        "event_extraction": {
            "connectivity": int(args.connectivity),
            "minimum_duration_days": int(args.min_duration_days),
            "minimum_max_daily_area_km2": float(args.min_event_area_km2),
            "mask_encoding": "ordinal_cell_rle_v1",
        },
        "matching": {
            "radii_km": args.match_radii_km_parsed,
            "temporal_tolerances_days": args.match_tolerances_days_parsed,
            "primary_radius_km": float(args.primary_match_radius_km),
            "primary_tolerance_days": int(args.primary_match_tolerance_days),
            "weights": {
                "timing": float(args.match_weight_timing),
                "distance": float(args.match_weight_distance),
                "iou": float(args.match_weight_iou),
                "duration": float(args.match_weight_duration),
                "area": float(args.match_weight_area),
                "intensity": float(args.match_weight_intensity),
            },
            "assignment": "global one-to-one Hungarian assignment within each member/year/track",
            "tolerant_iou": (
                "maximum over temporal shifts of symmetric bidirectional spatial-dilation "
                "IoU; intersection is the smaller directional area-weighted hit volume"
            ),
            "implementation": (
                "all configured radius-time tolerances share cached pair geometry; "
                "large components use equivalent dense boolean morphology"
            ),
            "primary_score_excludes_oracle": True,
        },
        "regionalization": region_metadata,
        "nulls": {
            "circular_shift_replicates": int(args.null_replicates),
            "marginal_preserving_independence_replicates": int(
                args.independence_surrogate_replicates
            ),
            "minimum_shift_days": int(args.minimum_null_shift_days),
        },
        "bootstrap": {
            "unit": "complete year with all members/events preserved",
            "replicates": int(args.bootstrap_replicates),
        },
        "paper_metrics": {
            "tail_quantiles": [float(value) for value in args.paper_quantiles_parsed],
            "twcrps_tail_quantile": float(args.twcrps_tail_quantile),
            "chi_distance_bin_edges_km": [
                float(value) for value in args.chi_distance_bins_km_parsed
            ],
            "chi_max_pairs_per_bin": int(args.chi_max_pairs_per_bin),
            "are_max_reference_points": int(args.are_max_reference_points),
            "local_are_quantile": float(args.local_are_quantile),
            "margins": "source-specific frozen fit-period empirical quantiles",
        },
        "random_seed": int(args.random_seed),
        "artifacts": {
            key: [str(item) for item in value]
            if isinstance(value, (list, tuple))
            else str(value)
            for key, value in artifacts.items()
        },
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def run_marked_event_workflow(args) -> dict[str, object]:
    from run_weekly_full_sweep import parse_years

    fit_years = parse_years(args.fit_years)
    validation_years = parse_years(args.validation_years)
    test_years = parse_years(args.test_years)
    from marked_event_core import ensure_disjoint_periods

    ensure_disjoint_periods(fit_years, validation_years, test_years)
    stages = set(args.marked_stages)
    if "all" in stages:
        stages = {
            "thresholds",
            "catalogs",
            "matching",
            "coverage",
            "process",
            "nulls",
            "paper-metrics",
            "figures",
        }
    paths = make_marked_paths(args.out_root)
    daily_paths = input_paths(args.daily_first_root)
    artifacts: dict[str, object] = {}
    threshold_file = threshold_artifact_path(paths, args)
    if "thresholds" in stages or not threshold_file.exists():
        _log(f"fitting frozen thresholds from {args.fit_years}")
        fit_cases = cases_for_years(fit_years, args, apply_debug_limit=True)
        validate_field_inputs(fit_cases, daily_paths, args)
        threshold_file = fit_frozen_thresholds(fit_cases, daily_paths, paths, args)
    artifacts["frozen_thresholds"] = threshold_file
    with xr.open_dataset(threshold_file) as ds:
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
    region_map, region_metadata = load_region_map(args.frozen_region_map, lat, lon)
    del region_map
    validation_cases = cases_for_years(
        validation_years,
        args,
        apply_debug_limit=True,
    )
    validate_field_inputs(validation_cases, daily_paths, args)
    _log(f"selecting event filters on validation years {args.validation_years}")
    artifacts["validation_event_filter_selection"] = select_event_filters_on_validation(
        validation_years,
        daily_paths,
        paths,
        args,
    )
    catalog_file = event_catalog_path(paths, args)
    reference_file = era5_reference_catalog_path(paths, args)
    valid_file = valid_members_path(paths, args)
    catalog_needed = bool(
        stages & {"catalogs", "matching", "coverage", "process", "nulls", "figures"}
    )
    if catalog_needed and (
        "catalogs" in stages
        or not all(path.exists() for path in (catalog_file, reference_file, valid_file))
    ):
        _log(f"extracting 3D marked-event catalogs; reserved test years {args.test_years}")
        reference_years = sorted(set(fit_years + validation_years + test_years))
        reference_cases = cases_for_years(reference_years, args, apply_debug_limit=True)
        validate_field_inputs(reference_cases, daily_paths, args)
        catalog_file, reference_file, valid_file, region_metadata = build_event_catalogs(
            reference_years,
            test_years,
            daily_paths,
            paths,
            args,
        )
    artifacts.update(
        {
            "event_catalog": catalog_file,
            "era5_loyo_reference_catalog": reference_file,
            "valid_members": valid_file,
        }
    )
    events: list[Event] = read_events(catalog_file) if catalog_file.exists() else []
    reference_events: list[Event] = read_events(reference_file) if reference_file.exists() else []
    valid_members = read_valid_members(valid_file) if valid_file.exists() else {}
    if "matching" in stages:
        _log("running memberwise global one-to-one event matching")
        match_outputs = run_event_matching(
            events,
            valid_members,
            lat,
            lon,
            paths,
            args,
        )
        artifacts["matching"] = list(match_outputs)
    if "coverage" in stages:
        _log("computing separately labeled spatiotemporal neighborhood coverage")
        artifacts["neighborhood_coverage"] = run_coverage_scores(
            events,
            valid_members,
            lat,
            lon,
            paths,
            args,
        )
    if "process" in stages:
        _log("scoring member event counts and censored first arrivals")
        process_outputs = run_count_arrival_process(
            events,
            reference_events,
            valid_members,
            region_metadata.get("region_ids", [0]),
            lat,
            lon,
            paths,
            args,
        )
        artifacts["count_arrival"] = list(process_outputs)
    if "nulls" in stages:
        _log("scoring complete-path circular-shift and optional independence nulls")
        artifacts["circular_shift_null"] = run_circular_shift_nulls(
            events,
            valid_members,
            lat,
            lon,
            paths,
            args,
        )
    if stages & {"matching", "process"}:
        if member_scores_path(paths, args).exists() and process_scores_path(paths, args).exists():
            _log("bootstrapping complete years")
            artifacts["bootstrap"] = run_bootstrap(paths, args)
        else:
            _log(
                "deferring year bootstrap until both matching and process scores exist"
            )
    if "paper-metrics" in stages:
        _log("computing tail-weighted CRPS, chi, ARE, and upper-tail Q-Q diagnostics")
        paper_cases = cases_for_years(test_years, args, apply_debug_limit=True)
        validate_field_inputs(paper_cases, daily_paths, args)
        artifacts["paper_metrics"] = list(
            run_paper_metrics(test_years, daily_paths, paths, args).values()
        )
    if "figures" in stages:
        _log("rendering the ten paper figures and Markdown captions")
        required = [
            match_catalog_path(paths, args),
            member_scores_path(paths, args),
            tolerance_surface_path(paths, args),
            process_scores_path(paths, args),
            reliability_path(paths, args),
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Figure prerequisites are missing; run matching/process/paper-metrics first:\n"
                + "\n".join(str(path) for path in missing)
            )
        artifacts["figures"] = run_figures(events, lat, lon, paths, args)
        artifacts["figure_captions"] = captions_path(paths, args)
    artifacts["methods"] = write_methods(paths, args, region_metadata)
    run_log = run_log_path(paths, args)
    run_log.write_text(
        json.dumps(
            {
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "stages": sorted(stages),
                "fit_years": args.fit_years,
                "validation_years": args.validation_years,
                "test_years": args.test_years,
                "selected_connectivity": int(args.connectivity),
                "selected_min_duration_days": int(args.min_duration_days),
                "selected_min_event_area_km2": float(args.min_event_area_km2),
                "status": "complete",
            },
            indent=2,
        )
        + "\n"
    )
    artifacts["run_log"] = run_log
    artifacts["all_existing_outputs"] = sorted(
        path
        for path in paths.root.rglob("*")
        if path.is_file() and path != manifest_path(paths, args)
    )
    artifacts["manifest"] = write_manifest(paths, args, region_metadata, artifacts)
    return artifacts
