#!/usr/bin/env python3
"""Paper-style bivariate KDE contours from cached Workflow 5 fields.

The contour figure in Ma, Zhang, and Wikle (2025) compares bivariate samples
at two spatial locations under two model conditions.  ACE2 does not provide a
counterfactual conditioning run, so the direct Workflow 5 analogue compares
the ERA5 and ACE2 joint distributions at the same two locations.

No event catalogs are rebuilt.  This script reads the cached daily Tmax
NetCDF files and the already fitted frozen-threshold artifact.  For each panel
date, samples are pooled across the requested test years and a centered
calendar-day window.  The default modes are:

``absolute``
    Both ERA5 and ACE2 are standardized by the frozen ERA5 q90 and q95 fields.
    This evaluates joint behavior relative to the same physical hazard.

``relative``
    ERA5 is standardized by its own frozen q90/q95 fields and every ACE2
    member by the frozen ACE2 q90/q95 fields for its lag group.  This removes
    most marginal threshold bias and emphasizes joint dependence structure.

``physical`` is also available and plots daily Tmax directly in degrees C.
Contours are highest-density regions (HDRs): a line labelled 90% encloses
approximately 90% of the fitted bivariate KDE probability mass.

Example (defaults match the production Workflow 5 configuration)::

    python event_segmentation/stochastic/plot_bivariate_kde_contours.py \
      --daily-first-root \
        /home/vt55/ace2/outputs/lag_may/event_segmentation/daily_first_shape_sweep \
      --site-a 35 260 --site-b 35 266

The default output directory is
``<daily-first-root>/workflow5_temporal_support/figures/bivariate_kde``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/ace2_era5_mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAILY_FIRST_ROOT = (
    PROJECT_ROOT / "outputs/lag_may/event_segmentation/daily_first_shape_sweep"
)
DEFAULT_SITE_A = (35.0, 260.0)
DEFAULT_SITE_B = (35.0, 266.0)
SOURCE_COLORS = {"era5": "#d73027", "ace2": "#2166ac"}
SOURCE_LABELS = {"era5": "ERA5", "ace2": "ACE2 ensemble"}


@dataclass(frozen=True)
class Site:
    requested_lat: float
    requested_lon: float
    lat_index: int
    lon_index: int
    lat: float
    lon: float


@dataclass(frozen=True)
class DayPair:
    day: date
    era5: np.ndarray
    ace2: np.ndarray
    members: np.ndarray


@dataclass(frozen=True)
class PanelSamples:
    center_month_day: str
    days: tuple[date, ...]
    era5: np.ndarray
    ace2: np.ndarray
    era5_weights: np.ndarray
    ace2_weights: np.ndarray


def _slug_number(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_years(specification: str) -> list[int]:
    years: list[int] = []
    for raw_token in str(specification).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending year range is not allowed: {token!r}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    output = sorted(set(years))
    if not output:
        raise ValueError("At least one test year is required.")
    return output


def _parse_month_day(value: str) -> tuple[int, int]:
    try:
        month_text, day_text = str(value).strip().split("-", 1)
        month, day = int(month_text), int(day_text)
        date(2001, month, day)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"Expected a valid MM-DD calendar date, received {value!r}."
        ) from error
    return month, day


def _parse_bandwidth(value: str) -> str | float:
    normalized = str(value).strip().lower()
    if normalized in {"scott", "silverman"}:
        return normalized
    try:
        numeric = float(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "KDE bandwidth must be 'scott', 'silverman', or a positive number."
        ) from error
    if numeric <= 0:
        raise argparse.ArgumentTypeError("KDE bandwidth must be positive.")
    return numeric


def _normalize_scale_names(values: Iterable[str]) -> tuple[str, ...]:
    valid = {"physical", "absolute", "relative"}
    output: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            name = token.strip().lower()
            if not name:
                continue
            if name not in valid:
                raise ValueError(
                    f"Unknown scale {name!r}; choose from physical, absolute, relative."
                )
            if name not in output:
                output.append(name)
    if not output:
        raise ValueError("At least one scale is required.")
    return tuple(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paper-style bivariate KDE contours comparing cached ERA5 and "
            "ACE2 daily Tmax samples at two locations."
        )
    )
    parser.add_argument(
        "--daily-first-root",
        type=Path,
        default=DEFAULT_DAILY_FIRST_ROOT,
        help="Root containing daily_fields/era5 and daily_fields/ace2.",
    )
    parser.add_argument(
        "--workflow5-root",
        type=Path,
        default=None,
        help=(
            "Root containing the fitted Workflow 5 thresholds. Defaults to "
            "<daily-first-root>/workflow5_temporal_support."
        ),
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=None,
        help="Explicit frozen_dual_track_thresholds_*.nc artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-years", default="2011:2022")
    parser.add_argument("--fit-years", default="1981:2000")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--upper-percentile", type=float, default=95.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument(
        "--field-variant",
        choices=("smoothed", "unsmoothed"),
        default="smoothed",
    )
    parser.add_argument(
        "--calendar-window-days",
        type=int,
        default=7,
        help="Half-width of each pooled calendar-day panel.",
    )
    parser.add_argument(
        "--panel-dates",
        nargs="+",
        default=("06-15", "07-15", "08-15"),
        help="Panel center dates as MM-DD values.",
    )
    parser.add_argument(
        "--site-a",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=DEFAULT_SITE_A,
        help="First site; longitude may use either -180..180 or 0..360.",
    )
    parser.add_argument(
        "--site-b",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        default=DEFAULT_SITE_B,
        help="Second site; defaults to roughly 550 km east of site A.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=("absolute", "relative"),
        help="Any of: physical, absolute, relative.",
    )
    parser.add_argument(
        "--contour-masses",
        nargs="+",
        type=float,
        default=(0.50, 0.75, 0.90, 0.95),
        help="Probability masses enclosed by the highest-density contours.",
    )
    parser.add_argument("--kde-bandwidth", type=_parse_bandwidth, default="scott")
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay a light subsample of raw bivariate points.",
    )
    args = parser.parse_args(argv)
    args.panel_dates_parsed = tuple(_parse_month_day(value) for value in args.panel_dates)
    args.test_years_parsed = tuple(_parse_years(args.test_years))
    args.scales_parsed = _normalize_scale_names(args.scales)
    masses = np.asarray(args.contour_masses, dtype=float)
    if np.any(~np.isfinite(masses)) or np.any((masses <= 0) | (masses >= 1)):
        parser.error("Every --contour-masses value must lie strictly between 0 and 1.")
    args.contour_masses_parsed = tuple(sorted(set(float(value) for value in masses)))
    if args.calendar_window_days < 0:
        parser.error("--calendar-window-days cannot be negative.")
    if args.grid_size < 50:
        parser.error("--grid-size must be at least 50.")
    return args


def _daily_variable(source: str, field_variant: str) -> str:
    if source == "era5":
        return (
            "era5_smoothed_daily_tmax_C"
            if field_variant == "smoothed"
            else "era5_daily_tmax_C"
        )
    if source == "ace2":
        return (
            "ace2_smoothed_daily_tmax_C"
            if field_variant == "smoothed"
            else "ace2_daily_tmax_C"
        )
    raise ValueError(source)


def _daily_path(
    root: Path,
    source: str,
    target_day: date,
    gaussian_sigma: float,
) -> Path | None:
    directory = root / "daily_fields" / source
    pattern = (
        f"{source}_y{target_day.year}_w*_d*_md{target_day.month:02d}{target_day.day:02d}_"
        f"daily_tmax_sigma{_slug_number(gaussian_sigma)}.nc"
    )
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one cached {source.upper()} field for {target_day}, found:\n"
            + "\n".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def _angular_lon_distance(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((np.asarray(values, dtype=float) - float(target) + 180.0) % 360.0 - 180.0)


def _nearest_site(
    lat: np.ndarray,
    lon: np.ndarray,
    requested: Sequence[float],
) -> Site:
    requested_lat, requested_lon = (float(requested[0]), float(requested[1]))
    lat_index = int(np.argmin(np.abs(np.asarray(lat, dtype=float) - requested_lat)))
    lon_index = int(np.argmin(_angular_lon_distance(lon, requested_lon)))
    return Site(
        requested_lat=requested_lat,
        requested_lon=requested_lon,
        lat_index=lat_index,
        lon_index=lon_index,
        lat=float(lat[lat_index]),
        lon=float(lon[lon_index]),
    )


def _great_circle_km(site_a: Site, site_b: Site) -> float:
    radius = 6371.0088
    lat_a, lat_b = np.deg2rad([site_a.lat, site_b.lat])
    delta_lat = lat_b - lat_a
    delta_lon = np.deg2rad(
        (site_b.lon - site_a.lon + 180.0) % 360.0 - 180.0
    )
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * radius * math.asin(min(1.0, math.sqrt(haversine))))


def _candidate_days(
    years: Sequence[int],
    center: tuple[int, int],
    window_days: int,
) -> list[date]:
    output: list[date] = []
    month, day_of_month = center
    for year in years:
        center64 = np.datetime64(date(int(year), month, day_of_month), "D")
        for offset in range(-int(window_days), int(window_days) + 1):
            value = (center64 + np.timedelta64(offset, "D")).astype(object)
            if isinstance(value, date):
                output.append(value)
    return output


def _discover_grid_and_sites(
    daily_root: Path,
    requested_days: Sequence[date],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Site, Site]:
    for target_day in requested_days:
        path = _daily_path(daily_root, "era5", target_day, args.gaussian_sigma)
        if path is None:
            continue
        with xr.open_dataset(path) as dataset:
            lat = dataset["lat"].values.astype(float)
            lon = dataset["lon"].values.astype(float)
        return (
            lat,
            lon,
            _nearest_site(lat, lon, args.site_a),
            _nearest_site(lat, lon, args.site_b),
        )
    raise FileNotFoundError(
        f"No cached ERA5 daily fields were found beneath {daily_root / 'daily_fields/era5'}."
    )


def _read_day_pair(
    daily_root: Path,
    target_day: date,
    site_a: Site,
    site_b: Site,
    args: argparse.Namespace,
) -> DayPair | None:
    era_path = _daily_path(daily_root, "era5", target_day, args.gaussian_sigma)
    ace_path = _daily_path(daily_root, "ace2", target_day, args.gaussian_sigma)
    if era_path is None or ace_path is None:
        return None
    era_variable = _daily_variable("era5", args.field_variant)
    ace_variable = _daily_variable("ace2", args.field_variant)
    with xr.open_dataset(era_path) as dataset:
        if era_variable not in dataset:
            raise KeyError(f"{era_path} has no variable {era_variable!r}.")
        field = dataset[era_variable]
        era_values = np.asarray(
            [
                field.isel(lat=site_a.lat_index, lon=site_a.lon_index).values,
                field.isel(lat=site_b.lat_index, lon=site_b.lon_index).values,
            ],
            dtype=float,
        )
    with xr.open_dataset(ace_path) as dataset:
        if ace_variable not in dataset:
            raise KeyError(f"{ace_path} has no variable {ace_variable!r}.")
        field = dataset[ace_variable]
        ace_values = np.column_stack(
            [
                field.isel(lat=site_a.lat_index, lon=site_a.lon_index).values,
                field.isel(lat=site_b.lat_index, lon=site_b.lon_index).values,
            ]
        ).astype(float)
        members = (
            dataset["member"].values
            if "member" in dataset.coords
            else np.arange(ace_values.shape[0], dtype=int)
        )
    return DayPair(
        day=target_day,
        era5=era_values,
        ace2=ace_values,
        members=np.asarray(members),
    )


def _load_panels(
    daily_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[DayPair]], Site, Site]:
    all_requested_days = [
        target_day
        for center in args.panel_dates_parsed
        for target_day in _candidate_days(
            args.test_years_parsed,
            center,
            args.calendar_window_days,
        )
    ]
    _, _, site_a, site_b = _discover_grid_and_sites(
        daily_root,
        all_requested_days,
        args,
    )
    panels: dict[str, list[DayPair]] = {}
    missing: list[date] = []
    for center in args.panel_dates_parsed:
        center_label = f"{center[0]:02d}-{center[1]:02d}"
        records: list[DayPair] = []
        for target_day in _candidate_days(
            args.test_years_parsed,
            center,
            args.calendar_window_days,
        ):
            record = _read_day_pair(daily_root, target_day, site_a, site_b, args)
            if record is None:
                missing.append(target_day)
            else:
                records.append(record)
        if not records:
            raise FileNotFoundError(
                f"No paired ERA5/ACE2 daily fields were found for panel {center_label}."
            )
        panels[center_label] = records
    if missing:
        unique = sorted(set(missing))
        preview = ", ".join(str(value) for value in unique[:8])
        suffix = f", and {len(unique) - 8} more" if len(unique) > 8 else ""
        warnings.warn(
            f"Skipped {len(unique)} dates missing one or both cached fields: {preview}{suffix}",
            RuntimeWarning,
        )
    return panels, site_a, site_b


def _threshold_matches(dataset: xr.Dataset, args: argparse.Namespace) -> bool:
    attrs = dataset.attrs
    checks = (
        str(attrs.get("fit_years", "")) == str(args.fit_years),
        str(attrs.get("test_years_reserved", "")) == str(args.test_years),
        str(attrs.get("field_variant", "")) == str(args.field_variant),
        np.isclose(float(attrs.get("percentile", np.nan)), float(args.percentile)),
        np.isclose(
            float(attrs.get("gaussian_sigma_grid_cells", np.nan)),
            float(args.gaussian_sigma),
        ),
        int(attrs.get("calendar_window_half_width_days", -1))
        == int(args.calendar_window_days),
    )
    return bool(all(checks))


def _find_threshold_file(workflow5_root: Path, args: argparse.Namespace) -> Path:
    if args.threshold_file is not None:
        path = args.threshold_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Frozen threshold artifact does not exist: {path}")
        return path
    candidates = sorted(
        (workflow5_root / "thresholds").glob("frozen_dual_track_thresholds_*.nc")
    )
    matches: list[Path] = []
    for path in candidates:
        try:
            with xr.open_dataset(path) as dataset:
                if _threshold_matches(dataset, args):
                    matches.append(path)
        except (OSError, ValueError, TypeError):
            continue
    if not matches:
        detail = "\n".join(str(path) for path in candidates[:10])
        raise FileNotFoundError(
            "Could not identify the frozen Workflow 5 threshold artifact matching "
            f"fit={args.fit_years}, test={args.test_years}, q={args.percentile:g}, "
            f"sigma={args.gaussian_sigma:g}, field={args.field_variant}. "
            "Pass it explicitly with --threshold-file."
            + (f"\nCandidates inspected:\n{detail}" if detail else "")
        )
    if len(matches) > 1:
        non_gpd = [path for path in matches if "_gpd" not in path.stem]
        if len(non_gpd) == 1:
            return non_gpd[0]
        raise RuntimeError(
            "Multiple matching frozen-threshold artifacts were found. Pass one with "
            "--threshold-file:\n" + "\n".join(str(path) for path in matches)
        )
    return matches[0]


def _quantile_value(dataset: xr.Dataset, percentile: float) -> float:
    available = dataset["quantile"].values.astype(float)
    target = float(percentile) / 100.0
    index = int(np.argmin(np.abs(available - target)))
    if not np.isclose(available[index], target, atol=1.0e-8):
        raise ValueError(
            f"Frozen threshold artifact lacks q={target:g}; available values are "
            f"{available.tolist()}."
        )
    return float(available[index])


def _calendar_index(dataset: xr.Dataset, target_day: date) -> int:
    month_day = f"{target_day.month:02d}-{target_day.day:02d}"
    values = dataset["calendar_day"].values.astype(str)
    indices = np.flatnonzero(values == month_day)
    if indices.size != 1:
        raise KeyError(f"Frozen thresholds have no unique calendar day {month_day}.")
    return int(indices[0])


def _member_groups(dataset: xr.Dataset, members: np.ndarray) -> np.ndarray:
    if "member_lag_group" not in dataset:
        if dataset.sizes.get("lag_group", 1) == 1:
            return np.zeros(len(members), dtype=int)
        raise KeyError("Threshold artifact has no member_lag_group mapping.")
    mapping = dataset["member_lag_group"]
    if "member" in mapping.coords:
        labels = mapping["member"].values
        lookup = {str(label): int(group) for label, group in zip(labels, mapping.values)}
        missing = [str(member) for member in members if str(member) not in lookup]
        if missing:
            raise KeyError(
                "Daily ACE2 members are absent from the frozen lag-group map: "
                + ", ".join(missing[:8])
            )
        return np.asarray([lookup[str(member)] for member in members], dtype=int)
    values = mapping.values.astype(int)
    if len(values) != len(members):
        raise ValueError("Frozen member lag-group map does not match daily member count.")
    return values


def _site_thresholds(
    dataset: xr.Dataset,
    variable: str,
    quantile: float,
    calendar_index: int,
    site_a: Site,
    site_b: Site,
) -> np.ndarray:
    field = dataset[variable].sel(quantile=quantile).isel(calendar_day=calendar_index)
    if "lag_group" in field.dims:
        return np.column_stack(
            [
                field.isel(lat=site_a.lat_index, lon=site_a.lon_index).values,
                field.isel(lat=site_b.lat_index, lon=site_b.lon_index).values,
            ]
        ).astype(float)
    return np.asarray(
        [
            field.isel(lat=site_a.lat_index, lon=site_a.lon_index).values,
            field.isel(lat=site_b.lat_index, lon=site_b.lon_index).values,
        ],
        dtype=float,
    )


def _safe_standardize(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    scale = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, np.nan)
    return (np.asarray(values, dtype=float) - np.asarray(lower, dtype=float)) / scale


def _transform_record(
    record: DayPair,
    scale_name: str,
    threshold_dataset: xr.Dataset | None,
    lower_quantile: float | None,
    upper_quantile: float | None,
    site_a: Site,
    site_b: Site,
) -> tuple[np.ndarray, np.ndarray]:
    if scale_name == "physical":
        return record.era5.copy(), record.ace2.copy()
    if threshold_dataset is None or lower_quantile is None or upper_quantile is None:
        raise RuntimeError(f"Scale {scale_name!r} requires the frozen thresholds.")
    calendar_index = _calendar_index(threshold_dataset, record.day)
    era_lower = _site_thresholds(
        threshold_dataset,
        "era5_quantile_C",
        lower_quantile,
        calendar_index,
        site_a,
        site_b,
    )
    era_upper = _site_thresholds(
        threshold_dataset,
        "era5_quantile_C",
        upper_quantile,
        calendar_index,
        site_a,
        site_b,
    )
    era5 = _safe_standardize(record.era5, era_lower, era_upper)
    if scale_name == "absolute":
        ace2 = _safe_standardize(record.ace2, era_lower, era_upper)
    elif scale_name == "relative":
        ace_lower_group = _site_thresholds(
            threshold_dataset,
            "ace2_quantile_C",
            lower_quantile,
            calendar_index,
            site_a,
            site_b,
        )
        ace_upper_group = _site_thresholds(
            threshold_dataset,
            "ace2_quantile_C",
            upper_quantile,
            calendar_index,
            site_a,
            site_b,
        )
        groups = _member_groups(threshold_dataset, record.members)
        ace2 = _safe_standardize(
            record.ace2,
            ace_lower_group[groups],
            ace_upper_group[groups],
        )
    else:
        raise ValueError(scale_name)
    return era5, ace2


def _panel_samples(
    center_month_day: str,
    records: Sequence[DayPair],
    scale_name: str,
    threshold_dataset: xr.Dataset | None,
    lower_quantile: float | None,
    upper_quantile: float | None,
    site_a: Site,
    site_b: Site,
) -> PanelSamples:
    era_rows: list[np.ndarray] = []
    ace_rows: list[np.ndarray] = []
    era_weights: list[float] = []
    ace_weights: list[np.ndarray] = []
    used_days: list[date] = []
    for record in records:
        era5, ace2 = _transform_record(
            record,
            scale_name,
            threshold_dataset,
            lower_quantile,
            upper_quantile,
            site_a,
            site_b,
        )
        era_valid = np.all(np.isfinite(era5))
        ace_valid = np.all(np.isfinite(ace2), axis=1)
        if era_valid:
            era_rows.append(era5)
            era_weights.append(1.0)
        if np.any(ace_valid):
            valid_values = ace2[ace_valid]
            ace_rows.append(valid_values)
            # Every date receives equal total mass even if a member is missing.
            ace_weights.append(np.full(len(valid_values), 1.0 / len(valid_values)))
        if era_valid or np.any(ace_valid):
            used_days.append(record.day)
    era_array = np.asarray(era_rows, dtype=float).reshape(-1, 2)
    ace_array = np.concatenate(ace_rows, axis=0) if ace_rows else np.empty((0, 2))
    if len(era_array) < 5 or len(ace_array) < 5:
        raise ValueError(
            f"Panel {center_month_day} has too few valid bivariate samples for KDE: "
            f"ERA5={len(era_array)}, ACE2={len(ace_array)}."
        )
    return PanelSamples(
        center_month_day=center_month_day,
        days=tuple(used_days),
        era5=era_array,
        ace2=ace_array,
        era5_weights=np.asarray(era_weights, dtype=float),
        ace2_weights=np.concatenate(ace_weights),
    )


def _axis_limits(samples: PanelSamples) -> tuple[float, float, float, float]:
    combined = np.vstack([samples.era5, samples.ace2])
    lower = np.nanquantile(combined, 0.002, axis=0)
    upper = np.nanquantile(combined, 0.998, axis=0)
    width = np.maximum(upper - lower, 1.0e-3)
    padding = 0.10 * width
    return (
        float(lower[0] - padding[0]),
        float(upper[0] + padding[0]),
        float(lower[1] - padding[1]),
        float(upper[1] + padding[1]),
    )


def _kde_grid(
    values: np.ndarray,
    weights: np.ndarray,
    limits: tuple[float, float, float, float],
    grid_size: int,
    bandwidth: str | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = limits
    x = np.linspace(x_min, x_max, int(grid_size))
    y = np.linspace(y_min, y_max, int(grid_size))
    xx, yy = np.meshgrid(x, y)
    try:
        kde = gaussian_kde(
            np.asarray(values, dtype=float).T,
            bw_method=bandwidth,
            weights=np.asarray(weights, dtype=float),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The bivariate sample covariance is singular; choose different sites or "
            "a wider calendar window."
        ) from error
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, density


def _hdr_density_threshold(density: np.ndarray, mass: float) -> float:
    values = np.asarray(density, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise ValueError("KDE returned no positive finite density values.")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, mass, side="left")), len(ordered) - 1)
    return float(ordered[index])


def _draw_source_contours(
    axis: plt.Axes,
    values: np.ndarray,
    weights: np.ndarray,
    source: str,
    limits: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> None:
    xx, yy, density = _kde_grid(
        values,
        weights,
        limits,
        args.grid_size,
        args.kde_bandwidth,
    )
    color = SOURCE_COLORS[source]
    masses = args.contour_masses_parsed
    for index, mass in enumerate(masses):
        threshold = _hdr_density_threshold(density, mass)
        contour = axis.contour(
            xx,
            yy,
            density,
            levels=[threshold],
            colors=[color],
            linewidths=[1.8 if mass <= 0.75 else 1.25],
            alpha=0.98 if mass <= 0.75 else 0.82,
            zorder=4,
        )
        if index in {0, len(masses) - 1}:
            axis.clabel(
                contour,
                fmt={threshold: f"{100 * mass:.0f}%"},
                inline=True,
                fontsize=7,
                colors=[color],
            )
    mean = np.average(values, axis=0, weights=weights)
    axis.scatter(
        mean[0],
        mean[1],
        marker="x",
        s=34,
        linewidth=1.5,
        color=color,
        zorder=6,
    )


def _site_label(site: Site) -> str:
    lon = ((site.lon + 180.0) % 360.0) - 180.0
    lat_suffix = "N" if site.lat >= 0 else "S"
    lon_suffix = "E" if lon >= 0 else "W"
    return f"{abs(site.lat):.2f} deg {lat_suffix}, {abs(lon):.2f} deg {lon_suffix}"


def _axis_text(scale_name: str, site: Site, lower: float, upper: float) -> str:
    if scale_name == "physical":
        return f"daily Tmax at {_site_label(site)} (deg C)"
    if scale_name == "absolute":
        qualifier = "ERA5-frozen"
    else:
        qualifier = "source-relative frozen"
    return (
        f"{qualifier} standardized Tmax at {_site_label(site)}\n"
        rf"$(T-q_{{{lower:g}}})/(q_{{{upper:g}}}-q_{{{lower:g}}})$"
    )


def _scale_title(scale_name: str) -> str:
    return {
        "physical": "Physical daily Tmax",
        "absolute": "ERA5-defined absolute hazard",
        "relative": "Source-relative extremes",
    }[scale_name]


def _plot_scale(
    panels: Sequence[PanelSamples],
    scale_name: str,
    site_a: Site,
    site_b: Site,
    distance_km: float,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.5 * len(panels), 5.1),
        squeeze=False,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.70,
        wspace=0.24,
    )
    axes_row = axes[0]
    random = np.random.default_rng(90210)
    for index, (axis, samples) in enumerate(zip(axes_row, panels)):
        limits = _axis_limits(samples)
        if scale_name != "physical":
            axis.axvline(0.0, color="0.75", linestyle="--", linewidth=0.8, zorder=0)
            axis.axhline(0.0, color="0.75", linestyle="--", linewidth=0.8, zorder=0)
        if args.show_points:
            for source, values in (("era5", samples.era5), ("ace2", samples.ace2)):
                count = min(400, len(values))
                indices = random.choice(len(values), size=count, replace=False)
                axis.scatter(
                    values[indices, 0],
                    values[indices, 1],
                    s=5,
                    alpha=0.08,
                    color=SOURCE_COLORS[source],
                    linewidth=0,
                    zorder=1,
                )
        _draw_source_contours(
            axis,
            samples.era5,
            samples.era5_weights,
            "era5",
            limits,
            args,
        )
        _draw_source_contours(
            axis,
            samples.ace2,
            samples.ace2_weights,
            "ace2",
            limits,
            args,
        )
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        center = date(2001, *_parse_month_day(samples.center_month_day)).strftime("%b %d")
        axis.set_title(
            f"{center} +/- {args.calendar_window_days} days\n"
            f"{len(samples.era5):,} ERA5 days; {len(samples.ace2):,} ACE2 member-days",
            fontsize=11,
            fontweight="bold",
        )
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.set_xlabel(
            _axis_text(
                scale_name,
                site_a,
                args.percentile,
                args.upper_percentile,
            ),
            fontsize=9,
        )
        if index == 0:
            axis.set_ylabel(
                _axis_text(
                    scale_name,
                    site_b,
                    args.percentile,
                    args.upper_percentile,
                ),
                fontsize=9,
            )
    legend = [
        Line2D([0], [0], color=SOURCE_COLORS[source], lw=2.0, label=SOURCE_LABELS[source])
        for source in ("era5", "ace2")
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            marker="x",
            color="0.2",
            linestyle="none",
            label="weighted mean",
        )
    )
    figure.legend(
        handles=legend,
        loc="center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.79),
    )
    figure.suptitle(
        f"Bivariate daily-temperature distributions: {_scale_title(scale_name)}\n"
        f"two sites separated by {distance_km:.0f} km; test years {args.test_years}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _summary_rows(
    panels: Sequence[PanelSamples],
    scale_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for samples in panels:
        for source, values, weights in (
            ("era5", samples.era5, samples.era5_weights),
            ("ace2", samples.ace2, samples.ace2_weights),
        ):
            normalized_weights = weights / weights.sum()
            mean = np.sum(values * normalized_weights[:, None], axis=0)
            covariance = np.cov(values.T, aweights=weights)
            correlation = covariance[0, 1] / math.sqrt(
                covariance[0, 0] * covariance[1, 1]
            )
            rows.append(
                {
                    "scale": scale_name,
                    "center_month_day": samples.center_month_day,
                    "source": source,
                    "n_samples": int(len(values)),
                    "n_dates": int(len(set(samples.days))),
                    "weighted_mean_site_a": float(mean[0]),
                    "weighted_mean_site_b": float(mean[1]),
                    "weighted_correlation": float(correlation),
                    "weighted_probability_both_above_q90": (
                        float(np.sum(normalized_weights[(values[:, 0] > 0) & (values[:, 1] > 0)]))
                        if scale_name != "physical"
                        else np.nan
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.10g}")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    daily_root = args.daily_first_root.expanduser().resolve()
    workflow5_root = (
        args.workflow5_root.expanduser().resolve()
        if args.workflow5_root is not None
        else daily_root / "workflow5_temporal_support"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow5_root / "figures" / "bivariate_kde"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_panels, site_a, site_b = _load_panels(daily_root, args)
    distance_km = _great_circle_km(site_a, site_b)
    print(
        f"[kde-contours] site A: {_site_label(site_a)}; "
        f"site B: {_site_label(site_b)}; separation={distance_km:.1f} km",
        flush=True,
    )

    threshold_dataset: xr.Dataset | None = None
    threshold_file: Path | None = None
    lower_quantile = upper_quantile = None
    if any(scale != "physical" for scale in args.scales_parsed):
        threshold_file = _find_threshold_file(workflow5_root, args)
        threshold_dataset = xr.load_dataset(threshold_file)
        lower_quantile = _quantile_value(threshold_dataset, args.percentile)
        upper_quantile = _quantile_value(threshold_dataset, args.upper_percentile)
        if upper_quantile <= lower_quantile:
            raise ValueError("--upper-percentile must exceed --percentile.")
        print(f"[kde-contours] thresholds: {threshold_file}", flush=True)

    summary: list[dict[str, object]] = []
    figures: list[Path] = []
    try:
        for scale_name in args.scales_parsed:
            panels = [
                _panel_samples(
                    center,
                    records,
                    scale_name,
                    threshold_dataset,
                    lower_quantile,
                    upper_quantile,
                    site_a,
                    site_b,
                )
                for center, records in raw_panels.items()
            ]
            stem = (
                "bivariate_kde_contours_"
                f"{scale_name}_test{args.test_years.replace(':', '-')}_"
                f"sites{site_a.lat:.2f}_{site_a.lon:.2f}-"
                f"{site_b.lat:.2f}_{site_b.lon:.2f}"
            ).replace(".", "p")
            output_path = output_dir / f"{stem}.png"
            _plot_scale(
                panels,
                scale_name,
                site_a,
                site_b,
                distance_km,
                output_path,
                args,
            )
            summary.extend(_summary_rows(panels, scale_name))
            figures.append(output_path)
            print(f"[kde-contours] wrote {output_path}", flush=True)
    finally:
        if threshold_dataset is not None:
            threshold_dataset.close()

    summary_path = output_dir / "bivariate_kde_contour_summary.csv"
    _write_csv(summary_path, summary)
    metadata = {
        "definition": (
            "Bivariate highest-density-region KDE contours at two fixed grid cells, "
            "pooling cached test-period daily fields within centered calendar windows."
        ),
        "paper_analogue": (
            "ERA5 versus ACE2 replaces the paper's original-versus-counterfactual "
            "condition comparison; no counterfactual ACE2 experiment was available."
        ),
        "daily_first_root": str(daily_root),
        "workflow5_root": str(workflow5_root),
        "threshold_file": str(threshold_file) if threshold_file else None,
        "test_years": list(args.test_years_parsed),
        "panel_dates": [f"{month:02d}-{day_value:02d}" for month, day_value in args.panel_dates_parsed],
        "calendar_window_half_width_days": int(args.calendar_window_days),
        "field_variant": args.field_variant,
        "gaussian_sigma_grid_cells": float(args.gaussian_sigma),
        "site_a": {"lat": site_a.lat, "lon": site_a.lon},
        "site_b": {"lat": site_b.lat, "lon": site_b.lon},
        "site_separation_km": distance_km,
        "scales": list(args.scales_parsed),
        "contour_probability_masses": list(args.contour_masses_parsed),
        "kde_bandwidth": args.kde_bandwidth,
        "figures": [str(path) for path in figures],
        "summary": str(summary_path),
    }
    metadata_path = output_dir / "bivariate_kde_contour_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[kde-contours] wrote {summary_path}", flush=True)
    print(f"[kde-contours] wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
