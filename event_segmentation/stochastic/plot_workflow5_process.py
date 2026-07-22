#!/usr/bin/env python3
"""Plot the Workflow 5 transformation and matching process from cached outputs.

The script creates two explanatory figures without refitting thresholds,
extracting events, or repeating the expensive memberwise matching:

1. ``field_to_event`` follows one representative ERA5 event from raw daily
   Tmax through Gaussian smoothing, its frozen threshold, the exceedance mask,
   and the final connected component.
2. ``memberwise_matching`` follows that same ERA5 event through one ACE2
   member match and then through aggregation of matched footprints over all
   valid members.

By default the representative event and ACE2 member are exactly the automatic
choices used by Workflow 5 Figures 2 and 3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/ace2_era5_mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
EVENT_SEGMENTATION_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVENT_SEGMENTATION_DIR))

from marked_event_core import Event
from marked_event_workflow import (
    _event_day_mask,
    _event_footprint_mask,
    _exact_spacetime_iou,
    _format_are_map,
    read_events,
)
from plot_bivariate_kde_contours import _daily_path, _find_threshold_file
from run_daily_first_shape_sweep import lon_for_plot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAILY_FIRST_ROOT = (
    PROJECT_ROOT / "outputs/lag_may/event_segmentation/daily_first_shape_sweep"
)


def _comma_floats(values: Sequence[str]) -> tuple[float, ...]:
    output: list[float] = []
    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                output.append(float(token))
    return tuple(sorted(set(output)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the cached Workflow 5 field-to-event and matching process."
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
            "Root containing Workflow 5 thresholds, catalogs, and matching. "
            "Defaults to <daily-first-root>/workflow5_temporal_support."
        ),
    )
    parser.add_argument("--threshold-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--event-id", default=None)
    parser.add_argument(
        "--member",
        type=int,
        default=None,
        help="ACE2 member to display; default chooses the near-median valid match.",
    )
    parser.add_argument("--track", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--fit-years", default="1981:2000")
    parser.add_argument("--test-years", default="2011:2022")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument(
        "--field-variant",
        choices=("smoothed", "unsmoothed"),
        default="smoothed",
    )
    parser.add_argument("--calendar-window-days", type=int, default=7)
    parser.add_argument("--primary-match-radius-km", type=float, default=500.0)
    parser.add_argument("--primary-match-tolerance-days", type=int, default=3)
    parser.add_argument(
        "--probability-contours",
        nargs="+",
        default=("0.25", "0.50", "0.75"),
    )
    parser.add_argument("--padding-cells", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    args.daily_first_root = args.daily_first_root.expanduser().resolve()
    args.workflow5_root = (
        args.workflow5_root.expanduser().resolve()
        if args.workflow5_root is not None
        else args.daily_first_root / "workflow5_temporal_support"
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.workflow5_root / "figures" / "process"
    )
    args.probability_contours_parsed = _comma_floats(args.probability_contours)
    if any(value <= 0 or value >= 1 for value in args.probability_contours_parsed):
        parser.error("--probability-contours values must lie strictly between 0 and 1.")
    if args.padding_cells < 0:
        parser.error("--padding-cells cannot be negative.")
    return args


def _workflow_slug(threshold_file: Path) -> str:
    prefix = "frozen_dual_track_thresholds_"
    if not threshold_file.stem.startswith(prefix):
        raise ValueError(
            "Expected a frozen_dual_track_thresholds_*.nc file, received "
            f"{threshold_file}."
        )
    return threshold_file.stem[len(prefix) :]


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _artifact_paths(root: Path, slug: str) -> dict[str, Path]:
    return {
        "events": _required_file(
            root / "catalogs" / f"marked_event_catalog_{slug}.csv",
            "marked-event catalog",
        ),
        "matches": _required_file(
            root / "matching" / f"one_to_one_event_matches_{slug}.csv",
            "one-to-one match catalog",
        ),
        "support": _required_file(
            root / "matching" / f"observed_event_member_support_{slug}.csv",
            "event-support table",
        ),
        "valid_members": _required_file(
            root / "catalogs" / f"valid_members_by_year_{slug}.csv",
            "valid-member table",
        ),
    }


def _primary_rows(matches: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    return matches[
        (matches["row_type"] == "match")
        & (matches["track"].astype(str) == args.track)
        & np.isclose(
            pd.to_numeric(matches["match_radius_km"], errors="coerce"),
            float(args.primary_match_radius_km),
        )
        & (
            pd.to_numeric(matches["match_tolerance_days"], errors="coerce")
            == int(args.primary_match_tolerance_days)
        )
    ].copy()


def _valid_members_for_year(path: Path, year: int) -> list[int]:
    frame = pd.read_csv(path)
    selected = frame[pd.to_numeric(frame["year"], errors="coerce") == int(year)]
    if selected.empty:
        raise KeyError(f"No valid-member record exists for {year} in {path}.")
    return [int(value) for value in json.loads(str(selected.iloc[0]["valid_members"]))]


def _select_observed_event(
    events: Sequence[Event],
    primary: pd.DataFrame,
    support_path: Path,
    valid_members_path: Path,
    args: argparse.Namespace,
) -> tuple[Event, pd.DataFrame]:
    observed = {
        event.event_id: event
        for event in events
        if event.source == "era5" and event.track == args.track
    }
    if args.event_id is not None:
        if args.event_id not in observed:
            raise KeyError(
                f"{args.event_id!r} is not an ERA5 {args.track} event in the catalog."
            )
        chosen = observed[args.event_id]
        rows = primary[primary["observed_event_id"].astype(str) == chosen.event_id]
        if rows.empty:
            raise ValueError(
                f"ERA5 event {chosen.event_id} has no primary-gate member matches."
            )
        return chosen, rows

    support = pd.read_csv(support_path)
    support = support[
        (support["track"].astype(str) == args.track)
        & np.isclose(
            pd.to_numeric(support["match_radius_km"], errors="coerce"),
            float(args.primary_match_radius_km),
        )
        & (
            pd.to_numeric(support["match_tolerance_days"], errors="coerce")
            == int(args.primary_match_tolerance_days)
        )
    ]
    support_lookup = dict(
        zip(
            support["observed_event_id"].astype(str),
            pd.to_numeric(
                support["member_support_probability"], errors="coerce"
            ).fillna(0.0),
        )
    )
    candidates = [
        event_id
        for event_id in primary["observed_event_id"].dropna().astype(str).unique()
        if event_id in observed
    ]
    if not candidates:
        raise ValueError(f"No matched ERA5 {args.track} events are available.")
    scored: list[tuple[float, str]] = []
    for event_id in candidates:
        event = observed[event_id]
        cells = event.voxels[:, 1].astype(int)
        rows, cols = np.divmod(cells, event.nlon)
        touches_boundary = bool(
            rows.min() == 0
            or rows.max() == event.nlat - 1
            or cols.min() == 0
            or cols.max() == event.nlon - 1
        )
        probability = support_lookup.get(event_id)
        if probability is None:
            n_valid = len(_valid_members_for_year(valid_members_path, event.year))
            n_matching = primary[
                primary["observed_event_id"].astype(str) == event_id
            ]["member"].nunique()
            probability = n_matching / max(1, n_valid)
        boundary_factor = 0.2 if touches_boundary else 1.0
        score = (
            boundary_factor
            * max(0.04, float(probability))
            * math.log1p(max(0.0, event.spacetime_volume_km2_days))
            * math.sqrt(max(1, event.duration_days))
        )
        scored.append((score, event_id))
    chosen_id = max(scored)[1]
    chosen = observed[chosen_id]
    return chosen, primary[primary["observed_event_id"].astype(str) == chosen_id]


def _select_forecast_event(
    events: Sequence[Event],
    matched_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Event, pd.Series]:
    lookup = {event.event_id: event for event in events}
    selected = matched_rows.copy()
    if args.member is not None:
        selected = selected[
            pd.to_numeric(selected["member"], errors="coerce") == int(args.member)
        ]
        if selected.empty:
            available = sorted(
                pd.to_numeric(matched_rows["member"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            raise ValueError(
                f"Member {args.member} does not match this event. Available members: "
                + ", ".join(str(value) for value in available)
            )
        row = selected.iloc[0]
    else:
        iou = pd.to_numeric(selected["tolerant_iou"], errors="coerce")
        distance = pd.to_numeric(selected["centroid_distance_km"], errors="coerce")
        score = (
            (iou - iou.median()).abs()
            + (distance - distance.median()).abs()
            / max(1.0, float(args.primary_match_radius_km))
        )
        row = selected.loc[score.idxmin()]
    forecast_id = str(row["forecast_event_id"])
    if forecast_id not in lookup:
        raise KeyError(f"Matched forecast event {forecast_id} is absent from the catalog.")
    return lookup[forecast_id], row


def _crop(mask: np.ndarray, padding: int) -> tuple[slice, slice]:
    rows, cols = np.where(mask)
    if not rows.size:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    return (
        slice(max(0, int(rows.min()) - padding), min(mask.shape[0], int(rows.max()) + padding + 1)),
        slice(max(0, int(cols.min()) - padding), min(mask.shape[1], int(cols.max()) + padding + 1)),
    )


def _plot_longitude(value: float) -> float:
    return float(value - 360.0 if value > 180.0 else value)


def _format_axes(axes: Sequence[plt.Axes], lon: np.ndarray, lat: np.ndarray) -> None:
    for axis in axes:
        _format_are_map(axis, lon, lat)
        axis.tick_params(labelsize=7)


def _threshold_for_day(
    dataset: xr.Dataset,
    target_day: date,
    percentile: float,
) -> np.ndarray:
    quantiles = dataset["quantile"].values.astype(float)
    target_quantile = float(percentile) / 100.0
    quantile_index = int(np.argmin(np.abs(quantiles - target_quantile)))
    if not np.isclose(quantiles[quantile_index], target_quantile):
        raise ValueError(
            f"Threshold artifact lacks u={target_quantile:g}; available={quantiles.tolist()}."
        )
    month_day = target_day.strftime("%m-%d")
    calendar = dataset["calendar_day"].values.astype(str)
    indices = np.flatnonzero(calendar == month_day)
    if indices.size != 1:
        raise KeyError(f"Threshold artifact has no unique calendar day {month_day}.")
    return dataset["era5_quantile_C"].isel(
        quantile=quantile_index,
        calendar_day=int(indices[0]),
    ).values.astype(float)


def _event_day_counts(event: Event) -> np.ndarray:
    counts = np.zeros((event.nlat, event.nlon), dtype=int)
    if event.voxels.size:
        np.add.at(counts.reshape(-1), event.voxels[:, 1].astype(int), 1)
    return counts


def plot_field_to_event(
    observed: Event,
    threshold_file: Path,
    daily_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> None:
    peak_day = date.fromordinal(observed.peak_ordinal)
    daily_file = _daily_path(daily_root, "era5", peak_day, args.gaussian_sigma)
    if daily_file is None:
        raise FileNotFoundError(f"No cached ERA5 daily field exists for {peak_day}.")
    with xr.open_dataset(daily_file) as dataset:
        raw = dataset["era5_daily_tmax_C"].values.astype(float)
        smoothed = dataset["era5_smoothed_daily_tmax_C"].values.astype(float)
        lat = dataset["lat"].values.astype(float)
        lon = dataset["lon"].values.astype(float)
    with xr.open_dataset(threshold_file) as dataset:
        threshold = _threshold_for_day(dataset, peak_day, args.percentile)
    comparison_field = smoothed if args.field_variant == "smoothed" else raw
    comparison_label = "smoothed Tmax" if args.field_variant == "smoothed" else "raw Tmax"
    exceedance = comparison_field > threshold
    selected_day = _event_day_mask(observed, observed.peak_ordinal)
    footprint = _event_footprint_mask(observed)
    day_counts = _event_day_counts(observed)
    row_slice, col_slice = _crop(footprint, int(args.padding_cells))
    display_lat = lat[row_slice]
    display_lon = lon_for_plot(lon[col_slice])
    arrays = [raw, smoothed, threshold, comparison_field - threshold]
    raw_crop, smooth_crop, threshold_crop, excess_crop = [
        field[row_slice, col_slice] for field in arrays
    ]
    exceedance_crop = exceedance[row_slice, col_slice]
    selected_crop = selected_day[row_slice, col_slice]
    counts_crop = day_counts[row_slice, col_slice]

    temperature_values = np.concatenate(
        [raw_crop.ravel(), smooth_crop.ravel(), threshold_crop.ravel()]
    )
    temperature_values = temperature_values[np.isfinite(temperature_values)]
    temp_min, temp_max = np.quantile(temperature_values, [0.02, 0.98])
    finite_excess = np.abs(excess_crop[np.isfinite(excess_crop)])
    excess_limit = max(0.5, float(np.quantile(finite_excess, 0.98)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.3), constrained_layout=True)
    axes_flat = list(axes.ravel())
    temperature_panels = (
        (raw_crop, "1. Daily Tmax\nmax of 00/06/12/18 UTC"),
        (smooth_crop, f"2. Gaussian smoothing\n$\\sigma={args.gaussian_sigma:g}$ grid cell"),
        (threshold_crop, f"3. Frozen threshold\nERA5 $q_{{{args.percentile / 100:g}}}$, fit {args.fit_years}"),
    )
    temp_mesh = None
    for axis, (field, title) in zip(axes_flat[:3], temperature_panels):
        temp_mesh = axis.pcolormesh(
            display_lon,
            display_lat,
            field,
            shading="auto",
            cmap="coolwarm",
            vmin=float(temp_min),
            vmax=float(temp_max),
            rasterized=True,
        )
        axis.set_title(title, fontweight="bold")
    if temp_mesh is not None:
        fig.colorbar(temp_mesh, ax=axes_flat[:3], shrink=0.78, label="°C")

    excess_mesh = axes_flat[3].pcolormesh(
        display_lon,
        display_lat,
        excess_crop,
        shading="auto",
        cmap="RdBu_r",
        vmin=-excess_limit,
        vmax=excess_limit,
        rasterized=True,
    )
    axes_flat[3].contour(
        display_lon,
        display_lat,
        excess_crop,
        levels=[0.0],
        colors="black",
        linewidths=0.7,
    )
    axes_flat[3].set_title(
        f"4. Threshold excess\n{comparison_label} $-q_{{{args.percentile / 100:g}}}$",
        fontweight="bold",
    )
    fig.colorbar(excess_mesh, ax=axes_flat[3], shrink=0.78, label="°C")

    axes_flat[4].pcolormesh(
        display_lon,
        display_lat,
        exceedance_crop.astype(float),
        shading="auto",
        cmap=ListedColormap(["white", "#bdbdbd"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes_flat[4].pcolormesh(
        display_lon,
        display_lat,
        np.ma.masked_where(~selected_crop, selected_crop.astype(float)),
        shading="auto",
        cmap=ListedColormap(["#d73027"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes_flat[4].legend(
        handles=[
            Patch(facecolor="#bdbdbd", label="other daily exceedances"),
            Patch(facecolor="#d73027", label="selected component"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=7,
    )
    axes_flat[4].set_title(
        f"5. Peak-day exceedances\nselected connected component: {peak_day}",
        fontweight="bold",
    )

    count_mesh = axes_flat[5].pcolormesh(
        display_lon,
        display_lat,
        np.ma.masked_where(counts_crop == 0, counts_crop),
        shading="auto",
        cmap="viridis",
        vmin=1,
        vmax=max(1, observed.duration_days),
        rasterized=True,
    )
    axes_flat[5].set_title(
        "6. Segmented event\nnumber of days each cell belongs to event",
        fontweight="bold",
    )
    fig.colorbar(count_mesh, ax=axes_flat[5], shrink=0.78, label="occupied days")
    _format_axes(axes_flat, display_lon, display_lat)
    fig.suptitle(
        f"From daily temperature to one ERA5 {args.track}-track event\n"
        f"event {observed.event_id}: {observed.start_date} to {observed.end_date}; "
        f"peak {observed.peak_date}",
        fontweight="bold",
    )
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def _matched_support_probability(
    events: Sequence[Event],
    matched_rows: pd.DataFrame,
    valid_members: Sequence[int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[Event]]:
    lookup = {event.event_id: event for event in events}
    probability = np.zeros(shape, dtype=float)
    matched_events: list[Event] = []
    for member in valid_members:
        member_mask = np.zeros(shape, dtype=bool)
        rows = matched_rows[
            pd.to_numeric(matched_rows["member"], errors="coerce") == int(member)
        ]
        for event_id in rows["forecast_event_id"].dropna().astype(str):
            event = lookup.get(event_id)
            if event is None:
                continue
            member_mask |= _event_footprint_mask(event)
            matched_events.append(event)
        probability += member_mask
    probability /= max(1, len(valid_members))
    return probability, matched_events


def plot_memberwise_matching(
    observed: Event,
    forecast: Event,
    match_row: pd.Series,
    events: Sequence[Event],
    matched_rows: pd.DataFrame,
    valid_members: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    args: argparse.Namespace,
) -> None:
    obs_mask = _event_footprint_mask(observed)
    forecast_mask = _event_footprint_mask(forecast)
    probability, matched_events = _matched_support_probability(
        events,
        matched_rows,
        valid_members,
        obs_mask.shape,
    )
    combined = obs_mask | forecast_mask | (probability > 0)
    row_slice, col_slice = _crop(combined, int(args.padding_cells))
    display_lat = lat[row_slice]
    display_lon = lon_for_plot(lon[col_slice])
    obs_crop = obs_mask[row_slice, col_slice]
    forecast_crop = forecast_mask[row_slice, col_slice]
    probability_crop = probability[row_slice, col_slice]
    overlap = np.zeros(obs_crop.shape, dtype=int)
    overlap[obs_crop & ~forecast_crop] = 1
    overlap[~obs_crop & forecast_crop] = 2
    overlap[obs_crop & forecast_crop] = 3

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), constrained_layout=True)
    axes[0].pcolormesh(
        display_lon,
        display_lat,
        obs_crop.astype(float),
        shading="auto",
        cmap=ListedColormap(["white", "#2166ac"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[0].scatter(
        _plot_longitude(observed.centroid_lon),
        observed.centroid_lat,
        marker="*",
        s=75,
        color="black",
        zorder=10,
    )
    axes[0].set_title(
        f"1. ERA5 event\n{observed.start_date} to {observed.end_date}",
        fontweight="bold",
    )

    axes[1].pcolormesh(
        display_lon,
        display_lat,
        forecast_crop.astype(float),
        shading="auto",
        cmap=ListedColormap(["white", "#d73027"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[1].scatter(
        _plot_longitude(forecast.centroid_lon),
        forecast.centroid_lat,
        marker="s",
        s=35,
        color="black",
        zorder=10,
    )
    axes[1].set_title(
        f"2. ACE2 member {forecast.member} event\n"
        f"{forecast.start_date} to {forecast.end_date}",
        fontweight="bold",
    )

    overlap_cmap = ListedColormap(["white", "#2166ac", "#d73027", "#762a83"])
    axes[2].pcolormesh(
        display_lon,
        display_lat,
        overlap,
        shading="auto",
        cmap=overlap_cmap,
        vmin=0,
        vmax=3,
        rasterized=True,
    )
    axes[2].plot(
        [_plot_longitude(observed.centroid_lon), _plot_longitude(forecast.centroid_lon)],
        [observed.centroid_lat, forecast.centroid_lat],
        color="black",
        linewidth=1.0,
        zorder=10,
    )
    axes[2].scatter(
        [_plot_longitude(observed.centroid_lon), _plot_longitude(forecast.centroid_lon)],
        [observed.centroid_lat, forecast.centroid_lat],
        marker="o",
        s=24,
        color=["#2166ac", "#d73027"],
        edgecolors="black",
        linewidths=0.5,
        zorder=11,
    )
    axes[2].legend(
        handles=[
            Patch(facecolor="#2166ac", label="ERA5 only"),
            Patch(facecolor="#d73027", label="ACE2 only"),
            Patch(facecolor="#762a83", label="both footprints"),
        ],
        loc="upper right",
        fontsize=7,
        frameon=True,
    )
    axes[2].set_title(
        "3. One-to-one assignment\n"
        f"distance {float(match_row['centroid_distance_km']):.0f} km; "
        f"peak error {float(match_row['timing_error_days']):.0f} d",
        fontweight="bold",
    )

    support_mesh = axes[3].pcolormesh(
        display_lon,
        display_lat,
        probability_crop,
        shading="auto",
        cmap="magma_r",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    if np.any(obs_crop):
        axes[3].contour(
            display_lon,
            display_lat,
            obs_crop.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1.5,
        )
    contour_levels = [
        value
        for value in args.probability_contours_parsed
        if float(np.nanmin(probability_crop)) < value < float(np.nanmax(probability_crop))
    ]
    if contour_levels:
        contours = axes[3].contour(
            display_lon,
            display_lat,
            probability_crop,
            levels=contour_levels,
            colors="black",
            linewidths=0.8,
        )
        axes[3].clabel(contours, fmt=lambda value: f"p={value:g}", fontsize=7)
    for event in matched_events:
        axes[3].scatter(
            _plot_longitude(event.centroid_lon),
            event.centroid_lat,
            s=10,
            facecolors="none",
            edgecolors="white",
            linewidths=0.55,
            alpha=0.8,
            zorder=9,
        )
    axes[3].set_title(
        "4. Repeat for every member\n"
        f"matched support: {len(matched_rows)}/{len(valid_members)} members",
        fontweight="bold",
    )
    fig.colorbar(
        support_mesh,
        ax=axes[3],
        shrink=0.78,
        label="matched-member footprint probability",
    )
    _format_axes(list(axes), display_lon, display_lat)
    fig.suptitle(
        "Memberwise one-to-one event matching\n"
        f"eligible pairs: centroid distance ≤ {args.primary_match_radius_km:g} km and "
        f"peak-time error ≤ {args.primary_match_tolerance_days:d} d; "
        f"raw IoU={_exact_spacetime_iou(observed, forecast):.2f}, "
        f"tolerance-adjusted IoU={float(match_row['tolerant_iou']):.2f}",
        fontweight="bold",
    )
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_file = _find_threshold_file(args.workflow5_root, args)
    slug = _workflow_slug(threshold_file)
    artifacts = _artifact_paths(args.workflow5_root, slug)
    events = read_events(artifacts["events"])
    matches = pd.read_csv(artifacts["matches"])
    primary = _primary_rows(matches, args)
    if primary.empty:
        raise ValueError(
            f"No {args.track} matches exist at {args.primary_match_radius_km:g} km, "
            f"±{args.primary_match_tolerance_days:d} days."
        )
    observed, matched_rows = _select_observed_event(
        events,
        primary,
        artifacts["support"],
        artifacts["valid_members"],
        args,
    )
    forecast, match_row = _select_forecast_event(events, matched_rows, args)
    valid_members = _valid_members_for_year(artifacts["valid_members"], observed.year)
    with xr.open_dataset(threshold_file) as dataset:
        lat = dataset["lat"].values.astype(float)
        lon = dataset["lon"].values.astype(float)
    first_output = args.output_dir / f"workflow5_process01_field_to_event_{slug}.png"
    second_output = args.output_dir / f"workflow5_process02_memberwise_matching_{slug}.png"
    plot_field_to_event(
        observed,
        threshold_file,
        args.daily_first_root,
        first_output,
        args,
    )
    plot_memberwise_matching(
        observed,
        forecast,
        match_row,
        events,
        matched_rows,
        valid_members,
        lat,
        lon,
        second_output,
        args,
    )
    manifest = {
        "threshold_file": str(threshold_file),
        "workflow_slug": slug,
        "track": args.track,
        "observed_event_id": observed.event_id,
        "observed_start": observed.start_date,
        "observed_peak": observed.peak_date,
        "observed_end": observed.end_date,
        "forecast_event_id": forecast.event_id,
        "forecast_member": int(forecast.member),
        "forecast_start": forecast.start_date,
        "forecast_peak": forecast.peak_date,
        "forecast_end": forecast.end_date,
        "n_matching_members": int(len(matched_rows)),
        "n_valid_members": int(len(valid_members)),
        "centroid_distance_km": float(match_row["centroid_distance_km"]),
        "peak_timing_error_days": float(match_row["timing_error_days"]),
        "raw_spacetime_iou": _exact_spacetime_iou(observed, forecast),
        "tolerance_adjusted_iou": float(match_row["tolerant_iou"]),
        "outputs": [str(first_output), str(second_output)],
    }
    manifest_path = args.output_dir / f"workflow5_process_figures_{slug}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(first_output)
    print(second_output)
    print(manifest_path)


if __name__ == "__main__":
    main()
