#!/usr/bin/env python3
"""Compare the ACE2 hindcast high/low-HHE composite with SST experiments.

This is a post-processing diagnostic; it does not run ACE2 inference.  The
high/low years are read from the SST-composite artifact so the land response
uses exactly the years that defined the prescribed SST anomalies.  JJA mean
temperature is calculated from existing original-hindcast prediction files
and cached as a compact NetCDF for subsequent plotting.

Outputs
-------
hindcast_high_low_land_composite.png
    The requested two-panel hindcast JJA-temperature and HHE composite.
hindcast_vs_monthly_sst_response_maps.png
    Hindcast composite beside persistent March, April, May, and June responses.
hindcast_monthly_sst_response_summary.png
    Regional HHE amplitudes and spatial pattern correlations.
hindcast_monthly_sst_attribution.nc/json
    Compact numerical results and provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_ROOT = Path("/home/vt55/ace2")
MPL_CACHE = Path(tempfile.gettempdir()) / "ace2_matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr

from plot_raw_sst_composites import (
    coordinate_edges,
    draw_world_borders,
    world_geometries,
)


HI_TAGS = (
    ("persistent_mar_global", "Persistent March"),
    ("persistent_apr_global", "Persistent April"),
    ("persistent_may_global", "Persistent May"),
    ("persistent_jun_global", "Persistent June"),
)
MAP_EXTENT = (-125.0, -66.0, 22.0, 50.5)
NAVY = "#172936"
MUTED = "#66727D"
GRID = "#D8DDE1"
RED = "#BE3E4A"
BLUE = "#3479B9"
GREEN = "#008A5B"


@dataclass
class Experiment:
    key: str
    label: str
    temperature: np.ndarray
    hhe: np.ndarray
    regional_hhe: float
    ci_lower: float
    ci_upper: float
    temperature_pattern_r: float = math.nan
    hhe_pattern_r: float = math.nan
    amplitude_ratio_percent: float = math.nan


def coordinate_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"None of {candidates} found; coordinates={list(obj.coords)}")


def parse_recorded_years(value: object, label: str) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} metadata is not a list: {value!r}")
    result = sorted({int(year) for year in value})
    if not result:
        raise ValueError(f"No {label} are recorded")
    return result


def selected_years(composite: Path) -> tuple[list[int], list[int]]:
    with xr.open_dataset(composite) as ds:
        high = parse_recorded_years(ds.attrs.get("high_years"), "high_years")
        low = parse_recorded_years(ds.attrs.get("low_years"), "low_years")
    overlap = sorted(set(high).intersection(low))
    if overlap:
        raise ValueError(f"High and low groups overlap: {overlap}")
    return high, low


def lag_times(year: int, init_month: int = 5, init_day: int = 1) -> list[datetime]:
    center = datetime(year, init_month, init_day)
    return [center + timedelta(hours=6 * (member - 12)) for member in range(25)]


def prediction_path(runs_root: Path, year: int, member: int) -> Path:
    candidates = (
        runs_root / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc",
        runs_root / str(year) / f"member_{member:02d}" / "prediction.nc",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return candidates[0]


def member_jja_mean_temperature(path: Path, initialization: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as ds:
        if "TMP2m" not in ds:
            raise KeyError(f"TMP2m is absent from {path}")
        field = ds["TMP2m"]
        if "sample" in field.dims:
            field = field.isel(sample=0)
        time_name = coordinate_name(field, ("time",))
        lat_name = coordinate_name(field, ("lat", "latitude"))
        lon_name = coordinate_name(field, ("lon", "longitude"))
        raw_time = np.asarray(ds[time_name]).astype(np.int64)
        times = pd.Timestamp(initialization) + pd.to_timedelta(raw_time, unit="us")
        field = field.assign_coords({time_name: times})
        jja = field[time_name].dt.month.isin((6, 7, 8))
        field = field.sel({time_name: jja})
        if field.sizes.get(time_name, 0) == 0:
            raise ValueError(f"No JJA steps found in {path}")
        values = np.asarray(field.mean(time_name, skipna=True), dtype=np.float64)
        units = str(field.attrs.get("units", "")).lower()
        if "kelvin" in units or units.strip() in {"k", "degk"} or np.nanmedian(values) > 150.0:
            values -= 273.15
        lat = np.asarray(field[lat_name], dtype=float)
        lon = np.asarray(field[lon_name], dtype=float) % 360.0
    return values, lat, lon


def build_selected_temperature_cache(
    runs_root: Path,
    years: list[int],
    cache: Path,
    min_members: int,
) -> xr.Dataset:
    maps: list[np.ndarray] = []
    counts: list[int] = []
    lat = lon = None
    for year in years:
        member_maps: list[np.ndarray] = []
        for member, initialization in enumerate(lag_times(year)):
            path = prediction_path(runs_root, year, member)
            if not path.is_file():
                continue
            values, member_lat, member_lon = member_jja_mean_temperature(path, initialization)
            if lat is None:
                lat, lon = member_lat, member_lon
            elif not np.allclose(lat, member_lat) or not np.allclose(lon, member_lon):
                raise ValueError(f"Grid changed in {path}")
            member_maps.append(values)
        if len(member_maps) < min_members:
            raise RuntimeError(
                f"{year} has {len(member_maps)} original hindcast members; need {min_members}. "
                "No inference is required, but the existing prediction files must be available."
            )
        maps.append(np.nanmean(np.stack(member_maps), axis=0))
        counts.append(len(member_maps))
        print(f"hindcast temperature {year}: {len(member_maps)} members", flush=True)
    if lat is None or lon is None:
        raise RuntimeError("No original hindcast temperature fields were found")
    output = xr.Dataset(
        {
            "ace2_jja_mean_temperature": (
                ("year", "lat", "lon"), np.asarray(maps, dtype=np.float32)
            ),
            "n_members": (("year",), np.asarray(counts, dtype=np.int16)),
        },
        coords={"year": years, "lat": lat, "lon": lon},
        attrs={
            "description": "ACE2 original-hindcast ensemble-mean JJA 6-hourly TMP2m",
            "units": "degC",
            "inference_performed": "false",
            "source_runs": str(runs_root.resolve()),
        },
    )
    output["ace2_jja_mean_temperature"].attrs["units"] = "degC"
    cache.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(cache)
    print(f"wrote temperature cache {cache}", flush=True)
    return output


def load_hindcast_temperature(
    path: Path | None,
    runs_root: Path,
    cache: Path,
    years: list[int],
    min_members: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = path if path is not None else cache
    if not source.is_file():
        dataset = build_selected_temperature_cache(runs_root, years, cache, min_members)
    else:
        dataset = xr.open_dataset(source)
    try:
        candidates = (
            "ace2_jja_mean_temperature",
            "ace2_jja_mean_temperature_by_year",
        )
        variable = next((name for name in candidates if name in dataset), None)
        if variable is None:
            raise KeyError(f"No supported by-year temperature variable in {source}")
        da = dataset[variable]
        year_name = coordinate_name(da, ("year", "years"))
        lat_name = coordinate_name(da, ("lat", "latitude"))
        lon_name = coordinate_name(da, ("lon", "longitude"))
        available = {int(year) for year in da[year_name].values}
        missing = sorted(set(years).difference(available))
        if missing:
            raise ValueError(f"Temperature source {source} is missing selected years: {missing}")
        selected = da.sel({year_name: years}).transpose(year_name, lat_name, lon_name)
        values = np.asarray(selected, dtype=np.float64)
        units = str(da.attrs.get("units", dataset.attrs.get("units", ""))).lower()
        if "kelvin" in units or units.strip() in {"k", "degk"} or np.nanmedian(values) > 150.0:
            values -= 273.15
        return (
            np.asarray(years, dtype=int),
            values,
            np.asarray(selected[lat_name], dtype=float),
            np.asarray(selected[lon_name], dtype=float) % 360.0,
        )
    finally:
        dataset.close()


def load_hindcast_hhe(
    path: Path, variable: str, years: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} is absent from {path}; variables={list(ds.data_vars)}")
        da = ds[variable]
        year_name = coordinate_name(da, ("year", "years"))
        lat_name = coordinate_name(da, ("lat", "latitude"))
        lon_name = coordinate_name(da, ("lon", "longitude"))
        available = {int(year) for year in da[year_name].values}
        missing = sorted(set(years).difference(available))
        if missing:
            raise ValueError(f"HHE source {path} is missing selected years: {missing}")
        selected = da.sel({year_name: years}).transpose(year_name, lat_name, lon_name).load()
    return (
        np.asarray(years, dtype=int),
        np.asarray(selected, dtype=np.float64),
        np.asarray(selected[lat_name], dtype=float),
        np.asarray(selected[lon_name], dtype=float) % 360.0,
    )


def load_land(path: Path, expected_lat: np.ndarray, expected_lon: np.ndarray) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        if "land_fraction" not in ds:
            raise KeyError(f"land_fraction is absent from {path}")
        da = ds["land_fraction"]
        if "time" in da.dims:
            da = da.isel(time=0)
        lat_name = coordinate_name(ds, ("latitude", "lat"))
        lon_name = coordinate_name(ds, ("longitude", "lon"))
        lat = np.asarray(ds[lat_name], dtype=float)
        lon = np.asarray(ds[lon_name], dtype=float) % 360.0
        values = np.asarray(da, dtype=float).squeeze()
    if values.shape == (expected_lat.size, expected_lon.size) and np.allclose(
        lat, expected_lat
    ) and np.allclose(lon, expected_lon):
        return values > 0.5
    lat_index = np.asarray([int(np.argmin(np.abs(lat - value))) for value in expected_lat])
    lon_distance = np.abs(((lon[None, :] - expected_lon[:, None] + 180.0) % 360.0) - 180.0)
    lon_index = np.argmin(lon_distance, axis=1)
    return values[np.ix_(lat_index, lon_index)] > 0.5


def box_mask(lat: np.ndarray, lon: np.ndarray, box: list[float]) -> np.ndarray:
    south, north, west, east = box
    lon360 = lon % 360.0
    if west % 360.0 <= east % 360.0:
        longitude = (lon360 >= west % 360.0) & (lon360 <= east % 360.0)
    else:
        longitude = (lon360 >= west % 360.0) | (lon360 <= east % 360.0)
    return (
        (lat[:, None] >= south)
        & (lat[:, None] <= north)
        & longitude[None, :]
    )


def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(field)
    weights = np.cos(np.deg2rad(lat))[:, None] * valid
    denominator = weights.sum()
    return float(np.nansum(field * weights) / denominator) if denominator > 0 else math.nan


def pattern_correlation(
    first: np.ndarray, second: np.ndarray, lat: np.ndarray, mask: np.ndarray
) -> float:
    valid = mask & np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return math.nan
    weights = np.cos(np.deg2rad(lat))[:, None] * valid
    weights /= weights.sum()
    first_mean = np.sum(np.where(valid, first, 0.0) * weights)
    second_mean = np.sum(np.where(valid, second, 0.0) * weights)
    first_anomaly = np.where(valid, first - first_mean, 0.0)
    second_anomaly = np.where(valid, second - second_mean, 0.0)
    covariance = np.sum(weights * first_anomaly * second_anomaly)
    variance = np.sum(weights * first_anomaly**2) * np.sum(weights * second_anomaly**2)
    return float(covariance / np.sqrt(variance)) if variance > 0 else math.nan


def load_experiment(
    key: str,
    label: str,
    hhe_dir: Path,
    temperature_dir: Path,
    period: str,
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
) -> Experiment:
    hhe_path = hhe_dir / f"{key}_{period}.nc"
    temperature_path = temperature_dir / f"{key}_{period}_raw_tmax.nc"
    if not hhe_path.is_file():
        raise FileNotFoundError(hhe_path)
    if not temperature_path.is_file():
        raise FileNotFoundError(temperature_path)
    with xr.open_dataset(hhe_path) as ds:
        hhe = 100.0 * np.asarray(ds["high_minus_low_hhe_frequency"], dtype=np.float64)
        lat = np.asarray(ds["lat"], dtype=float)
        lon = np.asarray(ds["lon"], dtype=float) % 360.0
        if "regional_effect_mean" in ds:
            regional = 100.0 * float(ds["regional_effect_mean"])
        else:
            regional = 100.0 * float(np.mean(ds["regional_effect_by_year"]))
        ci_lower = (
            100.0 * float(ds["regional_effect_ci_95_lower"])
            if "regional_effect_ci_95_lower" in ds else math.nan
        )
        ci_upper = (
            100.0 * float(ds["regional_effect_ci_95_upper"])
            if "regional_effect_ci_95_upper" in ds else math.nan
        )
    with xr.open_dataset(temperature_path) as ds:
        temperature = np.asarray(
            ds["high_minus_low_jja_6hourly_temperature"], dtype=np.float64
        )
    if not np.allclose(lat, expected_lat) or not np.allclose(lon, expected_lon):
        raise ValueError(f"Grid mismatch in {hhe_path}")
    return Experiment(key, label, temperature, hhe, regional, ci_lower, ci_upper)


def robust_limit(fields: list[np.ndarray], mask: np.ndarray, floor: float) -> float:
    values = np.concatenate(
        [np.abs(field[mask & np.isfinite(field)]) for field in fields]
    )
    return max(float(np.percentile(values, 99.5)), floor) if values.size else floor


def map_panel(
    ax: plt.Axes,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land: np.ndarray,
    vmax: float,
    title: str,
    borders: list[tuple[object, str, float]],
    index_box: list[float],
) -> object:
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon180)
    plotted = np.where(land, field, np.nan)[:, order]
    mesh = ax.pcolormesh(
        coordinate_edges(lon180[order]),
        coordinate_edges(lat),
        plotted,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    draw_world_borders(ax, borders)
    ax.set_xlim(MAP_EXTENT[:2])
    ax.set_ylim(MAP_EXTENT[2:])
    ax.set_xticks([-120, -110, -100, -90, -80, -70])
    ax.set_yticks([25, 30, 35, 40, 45, 50])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=NAVY)
    south, north, west, east = index_box
    west180 = west - 360.0 if west > 180.0 else west
    east180 = east - 360.0 if east > 180.0 else east
    ax.add_patch(
        Rectangle(
            (west180, south), east180 - west180, north - south,
            fill=False, edgecolor=GREEN, linewidth=1.5, zorder=6,
        )
    )
    ax.set_facecolor("#EEF1F2")
    return mesh


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_hindcast_pair(
    temperature: np.ndarray,
    hhe: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land: np.ndarray,
    high_years: list[int],
    low_years: list[int],
    regional_hhe: float,
    index_box: list[float],
    output: Path,
    dpi: int,
) -> None:
    borders = world_geometries()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    temp_limit = robust_limit([temperature], land, 0.1)
    hhe_limit = robust_limit([hhe], land, 0.5)
    first = map_panel(
        axes[0], temperature, lat, lon, land, temp_limit,
        "JJA mean temperature: high minus low HHE years", borders, index_box,
    )
    second = map_panel(
        axes[1], hhe, lat, lon, land, hhe_limit,
        f"HHE frequency: high minus low  |  SEUS {regional_hhe:+.2f} pp",
        borders, index_box,
    )
    cbar = fig.colorbar(first, ax=axes[0], orientation="horizontal", pad=0.10, fraction=0.055)
    cbar.set_label("High-HHE minus low-HHE years (°C)")
    cbar = fig.colorbar(second, ax=axes[1], orientation="horizontal", pad=0.10, fraction=0.055)
    cbar.set_label("High-HHE minus low-HHE years (percentage points)")
    fig.suptitle(
        f"ACE2 hindcast composite  |  {len(high_years)} high and {len(low_years)} low years",
        fontsize=15, fontweight="bold", color=NAVY,
    )
    save(fig, output, dpi)


def plot_comparison_maps(
    hindcast_temperature: np.ndarray,
    hindcast_hhe: np.ndarray,
    experiments: list[Experiment],
    lat: np.ndarray,
    lon: np.ndarray,
    land: np.ndarray,
    index_box: list[float],
    output: Path,
    dpi: int,
) -> None:
    borders = world_geometries()
    temperature_fields = [hindcast_temperature, *(experiment.temperature for experiment in experiments)]
    hhe_fields = [hindcast_hhe, *(experiment.hhe for experiment in experiments)]
    temperature_limit = robust_limit(temperature_fields, land, 0.1)
    hhe_limit = robust_limit(hhe_fields, land, 0.5)
    fig, axes = plt.subplots(2, 5, figsize=(20.0, 7.7), constrained_layout=True)
    labels = ["Hindcast high−low", *(experiment.label for experiment in experiments)]
    top_mesh = bottom_mesh = None
    for column, (label, temperature, hhe) in enumerate(zip(labels, temperature_fields, hhe_fields)):
        top_title = label
        bottom_title = label
        if column > 0:
            experiment = experiments[column - 1]
            if np.isfinite(experiment.temperature_pattern_r):
                top_title += f"  |  r={experiment.temperature_pattern_r:+.2f}"
            if np.isfinite(experiment.hhe_pattern_r):
                bottom_title += f"  |  r={experiment.hhe_pattern_r:+.2f}"
        top_mesh = map_panel(
            axes[0, column], temperature, lat, lon, land, temperature_limit,
            top_title, borders, index_box,
        )
        bottom_mesh = map_panel(
            axes[1, column], hhe, lat, lon, land, hhe_limit,
            bottom_title, borders, index_box,
        )
        if column > 0:
            axes[0, column].set_ylabel("")
            axes[1, column].set_ylabel("")
    assert top_mesh is not None and bottom_mesh is not None
    cbar = fig.colorbar(top_mesh, ax=axes[0, :], orientation="horizontal", fraction=0.035, pad=0.07)
    cbar.set_label("JJA seasonal-mean temperature contrast (°C)")
    cbar = fig.colorbar(bottom_mesh, ax=axes[1, :], orientation="horizontal", fraction=0.035, pad=0.07)
    cbar.set_label("HHE-frequency contrast (percentage points)")
    fig.suptitle(
        "Hindcast high–low composite versus prescribed monthly SST responses",
        fontsize=15, fontweight="bold", color=NAVY,
    )
    save(fig, output, dpi)


def plot_summary(
    hindcast_regional_hhe: float,
    experiments: list[Experiment],
    output: Path,
    dpi: int,
) -> None:
    labels = [experiment.label.replace("Persistent ", "") for experiment in experiments]
    x = np.arange(len(experiments))
    means = np.asarray([experiment.regional_hhe for experiment in experiments])
    lower = np.asarray([experiment.ci_lower for experiment in experiments])
    upper = np.asarray([experiment.ci_upper for experiment in experiments])
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 4.9), constrained_layout=True)

    bars = axes[0].bar(x, means, color=RED, width=0.66, zorder=3)
    valid_ci = np.isfinite(lower) & np.isfinite(upper)
    axes[0].errorbar(
        x[valid_ci], means[valid_ci],
        yerr=np.vstack((means[valid_ci] - lower[valid_ci], upper[valid_ci] - means[valid_ci])),
        fmt="none", ecolor=NAVY, elinewidth=1.4, capsize=4, zorder=4,
    )
    axes[0].axhline(0.0, color=NAVY, linewidth=0.9)
    # The high-minus-low hindcast benchmark is much larger than the prescribed-
    # SST responses.  Keeping it as a horizontal line compresses the bars into
    # the bottom quarter of the panel, so show it as an explicit callout instead.
    axes[0].text(
        1.0,
        1.015,
        f"Hindcast high−low benchmark: {hindcast_regional_hhe:+.2f} pp",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=GREEN,
        fontweight="semibold",
    )
    finite_bounds = np.r_[
        means[np.isfinite(means)],
        lower[np.isfinite(lower)],
        upper[np.isfinite(upper)],
        0.0,
    ]
    amplitude_span = max(float(np.max(finite_bounds) - np.min(finite_bounds)), 0.25)
    label_pad = 0.045 * amplitude_span
    label_positions: list[float] = []
    for index, (bar, experiment) in enumerate(zip(bars, experiments)):
        ratio = experiment.amplitude_ratio_percent
        if experiment.regional_hhe >= 0.0:
            interval_edge = upper[index] if np.isfinite(upper[index]) else experiment.regional_hhe
            label_y = max(experiment.regional_hhe, interval_edge) + label_pad
            vertical_alignment = "bottom"
        else:
            interval_edge = lower[index] if np.isfinite(lower[index]) else experiment.regional_hhe
            label_y = min(experiment.regional_hhe, interval_edge) - label_pad
            vertical_alignment = "top"
        label_positions.append(label_y)
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{experiment.regional_hhe:+.2f} pp\n({ratio:.0f}% of high−low)",
            ha="center", va=vertical_alignment, fontsize=9, color=NAVY,
        )
    plot_min = min(float(np.min(finite_bounds)), min(label_positions), 0.0)
    plot_max = max(float(np.max(finite_bounds)), max(label_positions), 0.0)
    plot_span = max(plot_max - plot_min, 0.25)
    axes[0].set_ylim(plot_min - 0.08 * plot_span, plot_max + 0.15 * plot_span)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Regional HHE response (percentage points)")
    axes[0].set_title("Response amplitude", loc="left", fontweight="bold", color=NAVY)
    axes[0].grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    temperature_r = np.asarray(
        [experiment.temperature_pattern_r for experiment in experiments], dtype=float
    )
    hhe_r = np.asarray(
        [experiment.hhe_pattern_r for experiment in experiments], dtype=float
    )
    axes[1].plot(x, temperature_r, color=RED, marker="o", linewidth=1.7, label="JJA temperature")
    axes[1].plot(x, hhe_r, color=BLUE, marker="s", linewidth=1.7, label="HHE frequency")
    axes[1].axhline(0.0, color=NAVY, linewidth=0.9)
    finite_correlations = np.r_[
        temperature_r[np.isfinite(temperature_r)],
        hhe_r[np.isfinite(hhe_r)],
    ]
    correlation_bottom = 0.0 if (
        finite_correlations.size and np.min(finite_correlations) >= 0.0
    ) else -1.0
    axes[1].set_ylim(correlation_bottom, 1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Land pattern correlation with hindcast composite")
    axes[1].set_title("Spatial-pattern agreement", loc="left", fontweight="bold", color=NAVY)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, loc="best")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Persistent monthly SST responses versus the hindcast high−low composite",
        fontsize=13.5, fontweight="bold", color=NAVY,
    )
    save(fig, output, dpi)


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_CLUSTER_ROOT if DEFAULT_CLUSTER_ROOT.exists() else PROJECT_ROOT
    exp = root / "outputs/lag_may/sst_causal_v2"
    hhe_root = root / "outputs/lag_may/heat_index_era5"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite", type=Path, default=exp / "ace2_hhe_sst_composites_mar_aug.nc")
    parser.add_argument("--hindcast-hhe-nc", type=Path, default=hhe_root / "jja_hi_freq_ace2.nc")
    parser.add_argument("--hindcast-hhe-var", default="ace2_hi_freq")
    parser.add_argument("--hindcast-runs", type=Path, default=root / "outputs/lag_may/runs")
    parser.add_argument("--hindcast-temperature-nc", type=Path)
    parser.add_argument(
        "--hindcast-temperature-cache", type=Path,
        default=exp / "hindcast_attribution/jja_mean_temperature_selected_years.nc",
    )
    parser.add_argument("--hhe-evaluation-dir", type=Path, default=exp / "evaluation_1995_2005")
    parser.add_argument(
        "--temperature-evaluation-dir", type=Path,
        default=exp / "evaluation_raw_tmax_1995_2005",
    )
    parser.add_argument("--period", default="1995_2005")
    parser.add_argument(
        "--land-mask-forcing", type=Path,
        default=root / "data/lag_data/forcing_data_ace2era5/forcing_2000.nc",
    )
    parser.add_argument("--index-box", nargs=4, type=float, default=[23, 38, 260, 283])
    parser.add_argument("--min-members", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--out-dir", type=Path, default=exp / "hindcast_attribution")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    high_years, low_years = selected_years(args.composite)
    all_selected = sorted(set(high_years + low_years))

    _, hhe_by_year, lat, lon = load_hindcast_hhe(
        args.hindcast_hhe_nc, args.hindcast_hhe_var, all_selected
    )
    _, temperature_by_year, temperature_lat, temperature_lon = load_hindcast_temperature(
        args.hindcast_temperature_nc,
        args.hindcast_runs,
        args.hindcast_temperature_cache,
        all_selected,
        args.min_members,
    )
    if not np.allclose(lat, temperature_lat) or not np.allclose(lon, temperature_lon):
        raise ValueError("Hindcast HHE and temperature grids differ")
    year_index = {year: index for index, year in enumerate(all_selected)}
    high_index = [year_index[year] for year in high_years]
    low_index = [year_index[year] for year in low_years]
    hindcast_temperature = (
        np.nanmean(temperature_by_year[high_index], axis=0)
        - np.nanmean(temperature_by_year[low_index], axis=0)
    )
    hindcast_hhe = 100.0 * (
        np.nanmean(hhe_by_year[high_index], axis=0)
        - np.nanmean(hhe_by_year[low_index], axis=0)
    )

    land = load_land(args.land_mask_forcing, lat, lon)
    region = land & box_mask(lat, lon, args.index_box)
    comparison_domain = land & box_mask(
        lat, lon, [MAP_EXTENT[2], MAP_EXTENT[3], MAP_EXTENT[0] % 360.0, MAP_EXTENT[1] % 360.0]
    )
    regional_hindcast_hhe = weighted_mean(hindcast_hhe, lat, region)

    experiments = [
        load_experiment(
            key, label, args.hhe_evaluation_dir, args.temperature_evaluation_dir,
            args.period, lat, lon,
        )
        for key, label in HI_TAGS
    ]
    for experiment in experiments:
        experiment.temperature_pattern_r = pattern_correlation(
            hindcast_temperature, experiment.temperature, lat, comparison_domain
        )
        experiment.hhe_pattern_r = pattern_correlation(
            hindcast_hhe, experiment.hhe, lat, comparison_domain
        )
        experiment.amplitude_ratio_percent = (
            100.0 * experiment.regional_hhe / regional_hindcast_hhe
            if regional_hindcast_hhe != 0.0 else math.nan
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_hindcast_pair(
        hindcast_temperature, hindcast_hhe, lat, lon, land,
        high_years, low_years, regional_hindcast_hhe, args.index_box,
        args.out_dir / "hindcast_high_low_land_composite.png", args.dpi,
    )
    plot_comparison_maps(
        hindcast_temperature, hindcast_hhe, experiments, lat, lon, land,
        args.index_box, args.out_dir / "hindcast_vs_monthly_sst_response_maps.png", args.dpi,
    )
    plot_summary(
        regional_hindcast_hhe, experiments,
        args.out_dir / "hindcast_monthly_sst_response_summary.png", args.dpi,
    )

    output = xr.Dataset(
        {
            "hindcast_high_minus_low_jja_temperature": (
                ("lat", "lon"), hindcast_temperature.astype(np.float32)
            ),
            "hindcast_high_minus_low_hhe_frequency": (
                ("lat", "lon"), hindcast_hhe.astype(np.float32)
            ),
            "experiment_jja_temperature_response": (
                ("experiment", "lat", "lon"),
                np.stack([experiment.temperature for experiment in experiments]).astype(np.float32),
            ),
            "experiment_hhe_frequency_response": (
                ("experiment", "lat", "lon"),
                np.stack([experiment.hhe for experiment in experiments]).astype(np.float32),
            ),
            "regional_hhe_response": (
                ("experiment",),
                np.asarray([experiment.regional_hhe for experiment in experiments], dtype=np.float32),
            ),
            "regional_hhe_response_ci_95_lower": (
                ("experiment",),
                np.asarray([experiment.ci_lower for experiment in experiments], dtype=np.float32),
            ),
            "regional_hhe_response_ci_95_upper": (
                ("experiment",),
                np.asarray([experiment.ci_upper for experiment in experiments], dtype=np.float32),
            ),
            "temperature_pattern_correlation": (
                ("experiment",),
                np.asarray([experiment.temperature_pattern_r for experiment in experiments], dtype=np.float32),
            ),
            "hhe_pattern_correlation": (
                ("experiment",),
                np.asarray([experiment.hhe_pattern_r for experiment in experiments], dtype=np.float32),
            ),
            "regional_amplitude_ratio_to_hindcast_percent": (
                ("experiment",),
                np.asarray([experiment.amplitude_ratio_percent for experiment in experiments], dtype=np.float32),
            ),
        },
        coords={
            "experiment": [experiment.key for experiment in experiments],
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "description": "Hindcast high/low-HHE land composite and monthly SST-response comparison",
            "high_years": json.dumps(high_years),
            "low_years": json.dumps(low_years),
            "selection_source": str(args.composite.resolve()),
            "hindcast_composite_interpretation": "associational high-minus-low composite",
            "sst_experiment_interpretation": "causal prescribed-boundary-condition contrasts",
            "regional_hindcast_high_minus_low_hhe_percentage_points": regional_hindcast_hhe,
            "amplitude_ratio_warning": (
                "Ratios compare amplitudes; they are not additive variance fractions or formal mediation estimates."
            ),
        },
    )
    output["hindcast_high_minus_low_jja_temperature"].attrs["units"] = "degC"
    output["hindcast_high_minus_low_hhe_frequency"].attrs["units"] = "percentage points"
    output_path = args.out_dir / "hindcast_monthly_sst_attribution.nc"
    output.to_netcdf(output_path)
    summary = {
        "high_years": high_years,
        "low_years": low_years,
        "regional_hindcast_high_minus_low_hhe_percentage_points": regional_hindcast_hhe,
        "experiments": {
            experiment.key: {
                "regional_hhe_response_percentage_points": experiment.regional_hhe,
                "bootstrap_95pct_ci_percentage_points": [experiment.ci_lower, experiment.ci_upper],
                "amplitude_ratio_to_hindcast_percent": experiment.amplitude_ratio_percent,
                "temperature_pattern_correlation": experiment.temperature_pattern_r,
                "hhe_pattern_correlation": experiment.hhe_pattern_r,
            }
            for experiment in experiments
        },
        "interpretation_warning": (
            "Monthly amplitude ratios do not sum to 100%; the persistent experiments are separate, "
            "non-additive interventions rather than a formal decomposition."
        ),
    }
    (args.out_dir / "hindcast_monthly_sst_attribution.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
