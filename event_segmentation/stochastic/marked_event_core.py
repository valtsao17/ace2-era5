"""Core algorithms for marked spatiotemporal extreme-event evaluation.

The functions in this module are deliberately independent of the ACE2 archive
layout.  ``run_daily_first_temporal_support.py`` handles archive I/O and calls
these routines after fit/validation/test periods have been separated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label, maximum_filter, uniform_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Event:
    """One marked event with a compact sparse space-time mask.

    ``voxels`` is an ``(n, 2)`` integer array.  Column 0 contains Gregorian day
    ordinals and column 1 contains flattened latitude-longitude cell indices.
    """

    event_id: str
    source: str
    track: str
    member: int | None
    year: int
    start_ordinal: int
    peak_ordinal: int
    end_ordinal: int
    centroid_lat: float
    centroid_lon: float
    duration_days: int
    footprint_area_km2: float
    max_daily_area_km2: float
    spacetime_volume_km2_days: float
    peak_temperature_C: float
    mean_temperature_C: float
    peak_threshold_excess_C: float
    mean_threshold_excess_C: float
    standardized_quantile_intensity: float
    region_id: int
    daily_area_km2: tuple[float, ...]
    voxels: np.ndarray
    nlat: int
    nlon: int

    @property
    def start_date(self) -> str:
        return date.fromordinal(self.start_ordinal).isoformat()

    @property
    def peak_date(self) -> str:
        return date.fromordinal(self.peak_ordinal).isoformat()

    @property
    def end_date(self) -> str:
        return date.fromordinal(self.end_ordinal).isoformat()


@dataclass(frozen=True)
class MatchConfig:
    max_distance_km: float = 500.0
    max_timing_days: int = 3
    tolerant_iou_radius_pixels: int = 1
    timing_weight: float = 1.0
    distance_weight: float = 1.0
    iou_weight: float = 2.0
    duration_weight: float = 0.5
    area_weight: float = 0.5
    intensity_weight: float = 0.5
    unmatched_cost: float = 1.0e6


@dataclass(frozen=True)
class Match:
    observed_event_id: str
    forecast_event_id: str
    member: int
    track: str
    cost: float
    timing_error_days: float
    centroid_distance_km: float
    tolerant_iou: float
    best_shift_days: int
    duration_ratio: float
    area_ratio: float
    intensity_error: float


def ensure_disjoint_periods(
    fit_years: Sequence[int],
    validation_years: Sequence[int],
    test_years: Sequence[int],
) -> None:
    """Raise when any fit/validation/test period overlaps."""
    periods = {
        "fit": set(int(year) for year in fit_years),
        "validation": set(int(year) for year in validation_years),
        "test": set(int(year) for year in test_years),
    }
    for left, right in (("fit", "validation"), ("fit", "test"), ("validation", "test")):
        overlap = sorted(periods[left] & periods[right])
        if overlap:
            raise ValueError(f"{left} and {right} years overlap: {overlap}")
    if not all(periods.values()):
        empty = [name for name, years in periods.items() if not years]
        raise ValueError(f"Empty evaluation period(s): {', '.join(empty)}")


def percentile_thresholds(
    era5_fit: np.ndarray,
    ace2_fit: np.ndarray,
    quantiles: Sequence[float],
    lag_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ERA5 and pooled-member ACE2 quantiles from calibration arrays only.

    Parameters
    ----------
    era5_fit
        ``(time, lat, lon)`` calibration values.
    ace2_fit
        ``(time, member, lat, lon)`` calibration values.
    quantiles
        Quantiles expressed on ``[0, 1]``.
    lag_groups
        Optional member-index groups.  The default pools every member.
    """
    era5 = np.asarray(era5_fit, dtype=float)
    ace2 = np.asarray(ace2_fit, dtype=float)
    if era5.ndim != 3 or ace2.ndim != 4:
        raise ValueError("Expected ERA5 (time,lat,lon) and ACE2 (time,member,lat,lon).")
    if era5.shape[0] != ace2.shape[0] or era5.shape[1:] != ace2.shape[2:]:
        raise ValueError("ERA5 and ACE2 calibration grids/times do not match.")
    q = np.asarray(quantiles, dtype=float)
    if q.size == 0 or np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("All quantiles must be strictly between zero and one.")
    groups = list(lag_groups or [tuple(range(ace2.shape[1]))])
    if not groups or any(len(group) == 0 for group in groups):
        raise ValueError("Lag groups must contain at least one member.")
    with np.errstate(invalid="ignore"):
        era5_q = np.nanquantile(era5, q, axis=0)
        ace2_q = np.stack(
            [
                np.nanquantile(
                    ace2[:, np.asarray(group, dtype=int)].reshape(-1, *ace2.shape[2:]),
                    q,
                    axis=0,
                )
                for group in groups
            ],
            axis=0,
        )
    return era5_q.astype(np.float32), ace2_q.astype(np.float32)


def dual_track_masks(
    era5_values: np.ndarray,
    ace2_values: np.ndarray,
    era5_threshold: np.ndarray,
    ace2_threshold_by_member: np.ndarray,
    *,
    comparison: str = ">",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct relative-extreme and absolute-hazard masks without averaging members."""
    obs = np.asarray(era5_values, dtype=float)
    ens = np.asarray(ace2_values, dtype=float)
    era_th = np.asarray(era5_threshold, dtype=float)
    ace_th = np.asarray(ace2_threshold_by_member, dtype=float)
    if obs.ndim != 3 or ens.ndim != 4:
        raise ValueError("Expected ERA5 (time,lat,lon) and ACE2 (time,member,lat,lon).")
    if era_th.shape != obs.shape:
        raise ValueError("ERA5 threshold must match ERA5 time-space shape.")
    if ace_th.shape != ens.shape:
        raise ValueError("ACE2 relative threshold must match ACE2 member-time-space shape.")
    if comparison not in {">", ">="}:
        raise ValueError("comparison must be '>' or '>='.")
    op = np.greater if comparison == ">" else np.greater_equal
    obs_mask = op(obs, era_th) & np.isfinite(obs) & np.isfinite(era_th)
    relative = op(ens, ace_th) & np.isfinite(ens) & np.isfinite(ace_th)
    absolute_threshold = np.broadcast_to(era_th[:, np.newaxis, :, :], ens.shape)
    absolute = op(ens, absolute_threshold) & np.isfinite(ens) & np.isfinite(absolute_threshold)
    return {
        "relative": (obs_mask, relative),
        "absolute": (obs_mask.copy(), absolute),
    }


def _weighted_longitude(longitudes: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(longitudes)
    x = float(np.sum(weights * np.cos(radians)))
    y = float(np.sum(weights * np.sin(radians)))
    if x == 0.0 and y == 0.0:
        return float(np.average(longitudes, weights=weights))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def _connectivity_structure(connectivity: int) -> np.ndarray:
    mapping = {6: 1, 18: 2, 26: 3}
    if connectivity not in mapping:
        raise ValueError("3D connectivity must be one of 6, 18, or 26.")
    return generate_binary_structure(3, mapping[connectivity])


def encode_voxels_rle(voxels: np.ndarray) -> str:
    """Encode ordinal/cell pairs as JSON ``[ordinal, start_cell, length]`` runs."""
    arr = np.asarray(voxels, dtype=np.int64)
    if arr.size == 0:
        return "[]"
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    runs: list[list[int]] = []
    current_day = int(arr[0, 0])
    start = previous = int(arr[0, 1])
    for day_raw, cell_raw in arr[1:]:
        day = int(day_raw)
        cell = int(cell_raw)
        if day == current_day and cell == previous + 1:
            previous = cell
            continue
        runs.append([current_day, start, previous - start + 1])
        current_day = day
        start = previous = cell
    runs.append([current_day, start, previous - start + 1])
    return json.dumps(runs, separators=(",", ":"))


def decode_voxels_rle(payload: str) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for ordinal, start, length in json.loads(payload):
        rows.extend((int(ordinal), int(cell)) for cell in range(int(start), int(start) + int(length)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def extract_3d_events(
    mask: np.ndarray,
    temperature_C: np.ndarray,
    threshold_C: np.ndarray,
    standardized_intensity: np.ndarray,
    dates: Sequence[pd.Timestamp],
    lat: np.ndarray,
    lon: np.ndarray,
    cell_area_km2: np.ndarray,
    *,
    source: str,
    track: str,
    member: int | None,
    connectivity: int = 6,
    min_duration_days: int = 1,
    min_max_area_km2: float = 0.0,
    region_map: np.ndarray | None = None,
) -> list[Event]:
    """Label and mark 3D connected components in time-latitude-longitude."""
    event_mask = np.asarray(mask, dtype=bool)
    temp = np.asarray(temperature_C, dtype=float)
    threshold = np.asarray(threshold_C, dtype=float)
    standardized = np.asarray(standardized_intensity, dtype=float)
    if not (event_mask.shape == temp.shape == threshold.shape == standardized.shape):
        raise ValueError("Mask, temperature, threshold, and intensity arrays must have identical shapes.")
    if event_mask.ndim != 3 or event_mask.shape[0] != len(dates):
        raise ValueError("Event arrays must have shape (time, lat, lon).")
    ntime, nlat, nlon = event_mask.shape
    if cell_area_km2.shape != (nlat, nlon):
        raise ValueError("Cell-area grid does not match event grid.")
    ordinals = np.asarray([pd.Timestamp(day).date().toordinal() for day in dates], dtype=np.int64)
    labels, nlabels = label(event_mask, structure=_connectivity_structure(connectivity))
    lat2d, lon2d = np.meshgrid(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float), indexing="ij")
    area_flat = np.asarray(cell_area_km2, dtype=float).reshape(-1)
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    region_flat = None if region_map is None else np.asarray(region_map, dtype=int).reshape(-1)
    events: list[Event] = []
    member_label = "obs" if member is None else f"m{int(member):02d}"
    year = int(pd.Timestamp(dates[0]).year)
    for component in range(1, nlabels + 1):
        time_index, lat_index, lon_index = np.nonzero(labels == component)
        unique_times = np.unique(time_index)
        duration = int(unique_times.size)
        if duration < int(min_duration_days):
            continue
        cells = lat_index * nlon + lon_index
        voxel_area = area_flat[cells]
        daily_area = np.asarray(
            [float(voxel_area[time_index == idx].sum()) for idx in unique_times],
            dtype=float,
        )
        max_daily_area = float(daily_area.max(initial=0.0))
        if max_daily_area < float(min_max_area_km2):
            continue
        values = temp[time_index, lat_index, lon_index]
        excess = values - threshold[time_index, lat_index, lon_index]
        std_values = standardized[time_index, lat_index, lon_index]
        weights = np.where(np.isfinite(voxel_area) & (voxel_area > 0), voxel_area, 1.0)
        peak_local = int(np.nanargmax(excess)) if np.any(np.isfinite(excess)) else 0
        daily_peak_excess = [
            float(np.nanmax(excess[time_index == idx])) if np.any(np.isfinite(excess[time_index == idx])) else -np.inf
            for idx in unique_times
        ]
        peak_time_index = int(unique_times[int(np.argmax(daily_peak_excess))])
        union_cells = np.unique(cells)
        region_id = 0
        if region_flat is not None:
            valid_regions = region_flat[cells]
            candidates = np.unique(valid_regions[valid_regions >= 0])
            if candidates.size == 0:
                region_id = -1
            else:
                region_area = {
                    int(candidate): float(weights[valid_regions == candidate].sum())
                    for candidate in candidates
                }
                region_id = max(region_area, key=region_area.get)
        voxels = np.column_stack((ordinals[time_index], cells)).astype(np.int64)
        sequence = len(events) + 1
        event_id = f"{track}_{source}_{member_label}_{year}_{sequence:05d}"
        events.append(
            Event(
                event_id=event_id,
                source=source,
                track=track,
                member=None if member is None else int(member),
                year=year,
                start_ordinal=int(ordinals[unique_times.min()]),
                peak_ordinal=int(ordinals[peak_time_index]),
                end_ordinal=int(ordinals[unique_times.max()]),
                centroid_lat=float(np.average(lat_flat[cells], weights=weights)),
                centroid_lon=_weighted_longitude(lon_flat[cells], weights),
                duration_days=duration,
                footprint_area_km2=float(area_flat[union_cells].sum()),
                max_daily_area_km2=max_daily_area,
                spacetime_volume_km2_days=float(weights.sum()),
                peak_temperature_C=float(values[peak_local]),
                mean_temperature_C=float(np.average(values, weights=weights)),
                peak_threshold_excess_C=float(excess[peak_local]),
                mean_threshold_excess_C=float(np.average(excess, weights=weights)),
                standardized_quantile_intensity=(
                    float(np.nanmax(std_values))
                    if np.any(np.isfinite(std_values))
                    else float("nan")
                ),
                region_id=int(region_id),
                daily_area_km2=tuple(float(value) for value in daily_area),
                voxels=voxels,
                nlat=nlat,
                nlon=nlon,
            )
        )
    return events


def event_to_record(event: Event) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "source": event.source,
        "track": event.track,
        "member": np.nan if event.member is None else int(event.member),
        "year": int(event.year),
        "start": event.start_date,
        "peak": event.peak_date,
        "end": event.end_date,
        "centroid_lat": event.centroid_lat,
        "centroid_lon": event.centroid_lon,
        "duration_days": event.duration_days,
        "daily_area_km2": json.dumps(event.daily_area_km2, separators=(",", ":")),
        "footprint_area_km2": event.footprint_area_km2,
        "max_daily_area_km2": event.max_daily_area_km2,
        "spacetime_volume_km2_days": event.spacetime_volume_km2_days,
        "peak_temperature_C": event.peak_temperature_C,
        "mean_temperature_C": event.mean_temperature_C,
        "peak_threshold_excess_C": event.peak_threshold_excess_C,
        "mean_threshold_excess_C": event.mean_threshold_excess_C,
        "standardized_quantile_intensity": event.standardized_quantile_intensity,
        "region_id": event.region_id,
        "mask_encoding": "ordinal_cell_rle_v1",
        "mask_rle": encode_voxels_rle(event.voxels),
        "nlat": event.nlat,
        "nlon": event.nlon,
    }


def event_from_record(record: Mapping[str, object]) -> Event:
    member_value = record.get("member")
    member = None if member_value is None or pd.isna(member_value) else int(float(member_value))
    daily_area = tuple(float(value) for value in json.loads(str(record["daily_area_km2"])))
    return Event(
        event_id=str(record["event_id"]),
        source=str(record["source"]),
        track=str(record["track"]),
        member=member,
        year=int(record["year"]),
        start_ordinal=pd.Timestamp(record["start"]).date().toordinal(),
        peak_ordinal=pd.Timestamp(record["peak"]).date().toordinal(),
        end_ordinal=pd.Timestamp(record["end"]).date().toordinal(),
        centroid_lat=float(record["centroid_lat"]),
        centroid_lon=float(record["centroid_lon"]),
        duration_days=int(record["duration_days"]),
        footprint_area_km2=float(record["footprint_area_km2"]),
        max_daily_area_km2=float(record["max_daily_area_km2"]),
        spacetime_volume_km2_days=float(record["spacetime_volume_km2_days"]),
        peak_temperature_C=float(record["peak_temperature_C"]),
        mean_temperature_C=float(record["mean_temperature_C"]),
        peak_threshold_excess_C=float(record["peak_threshold_excess_C"]),
        mean_threshold_excess_C=float(record["mean_threshold_excess_C"]),
        standardized_quantile_intensity=float(record["standardized_quantile_intensity"]),
        region_id=int(record["region_id"]),
        daily_area_km2=daily_area,
        voxels=decode_voxels_rle(str(record["mask_rle"])),
        nlat=int(record["nlat"]),
        nlon=int(record["nlon"]),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    value = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(EARTH_RADIUS_KM * 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, value)))))


@dataclass
class _EventGeometry:
    """Sparse space-time coordinates and a reusable nearest-neighbor index."""

    coordinates: np.ndarray
    weights: np.ndarray
    total_weight: float
    tree: cKDTree
    time_scale: int


def _prepare_event_geometry(
    event: Event,
    cell_area_km2: np.ndarray,
    *,
    time_scale: int,
    cache: dict[tuple[int, int], _EventGeometry] | None,
) -> _EventGeometry:
    key = (id(event), int(time_scale))
    if cache is not None and key in cache:
        return cache[key]
    voxels = np.asarray(event.voxels, dtype=np.int64)
    cells = voxels[:, 1]
    rows, cols = np.divmod(cells, int(event.nlon))
    coordinates = np.column_stack(
        (
            voxels[:, 0].astype(np.float64) * float(time_scale),
            rows.astype(np.float64),
            cols.astype(np.float64),
        )
    )
    area = np.asarray(cell_area_km2, dtype=float).reshape(-1)
    weights = area[cells]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    geometry = _EventGeometry(
        coordinates=coordinates,
        weights=weights,
        total_weight=float(weights.sum()),
        tree=cKDTree(coordinates),
        time_scale=int(time_scale),
    )
    if cache is not None:
        cache[key] = geometry
    return geometry


def _nearest_hit_weights(
    source: _EventGeometry,
    target: _EventGeometry,
    *,
    source_time_adjustment_days: int,
    radii_pixels: Sequence[int],
    query_batch_size: int = 250_000,
) -> dict[int, float]:
    """Area of source voxels lying inside each spatial dilation of target."""
    radii = tuple(sorted(set(int(value) for value in radii_pixels)))
    hits = {radius: 0.0 for radius in radii}
    if not radii or source.coordinates.size == 0 or target.coordinates.size == 0:
        return hits
    maximum_radius = max(radii)
    time_adjustment = float(source_time_adjustment_days * source.time_scale)
    for start in range(0, len(source.coordinates), int(query_batch_size)):
        stop = min(len(source.coordinates), start + int(query_batch_size))
        query = source.coordinates[start:stop].copy()
        query[:, 0] += time_adjustment
        distances, _ = target.tree.query(
            query,
            k=1,
            p=np.inf,
            distance_upper_bound=float(maximum_radius) + 1.0e-9,
            workers=1,
        )
        batch_weights = source.weights[start:stop]
        for radius in radii:
            hits[radius] += float(
                batch_weights[distances <= float(radius) + 1.0e-9].sum()
            )
    return hits


def _dense_tolerant_event_iou_surface(
    observed: Event,
    forecast: Event,
    cell_area_km2: np.ndarray,
    *,
    tolerances: Sequence[int],
    radii: Sequence[int],
) -> dict[tuple[int, int], tuple[float, int]]:
    """Dense-array path for very large components, avoiding millions of tree queries."""
    maximum_shift = max(tolerances)
    minimum_ordinal = min(
        int(np.min(observed.voxels[:, 0])),
        int(np.min(forecast.voxels[:, 0])),
    ) - maximum_shift
    maximum_ordinal = max(
        int(np.max(observed.voxels[:, 0])),
        int(np.max(forecast.voxels[:, 0])),
    ) + maximum_shift
    ntime = maximum_ordinal - minimum_ordinal + 1
    shape = (ntime, int(observed.nlat), int(observed.nlon))
    observed_mask = np.zeros(shape, dtype=bool)
    forecast_mask = np.zeros(shape, dtype=bool)
    for event, mask in ((observed, observed_mask), (forecast, forecast_mask)):
        voxels = np.asarray(event.voxels, dtype=np.int64)
        rows, cols = np.divmod(voxels[:, 1], int(event.nlon))
        mask[voxels[:, 0] - minimum_ordinal, rows, cols] = True
    area = np.asarray(cell_area_km2, dtype=float)
    area = np.where(np.isfinite(area) & (area > 0), area, 0.0)
    observed_total = float(np.sum(observed_mask * area[None, :, :]))
    forecast_total = float(np.sum(forecast_mask * area[None, :, :]))
    by_radius_and_shift: dict[tuple[int, int], float] = {}
    for radius in radii:
        if radius == 0:
            observed_dilated = observed_mask
            forecast_dilated = forecast_mask
        else:
            filter_size = (1, 2 * int(radius) + 1, 2 * int(radius) + 1)
            observed_dilated = maximum_filter(
                observed_mask,
                size=filter_size,
                mode="constant",
                cval=0,
            )
            forecast_dilated = maximum_filter(
                forecast_mask,
                size=filter_size,
                mode="constant",
                cval=0,
            )
        for shift in range(-maximum_shift, maximum_shift + 1):
            if shift >= 0:
                observed_slice = slice(shift, ntime)
                forecast_slice = slice(0, ntime - shift)
            else:
                observed_slice = slice(0, ntime + shift)
                forecast_slice = slice(-shift, ntime)
            observed_hit_weight = float(
                np.sum(
                    (
                        observed_mask[observed_slice]
                        & forecast_dilated[forecast_slice]
                    )
                    * area[None, :, :]
                )
            )
            forecast_hit_weight = float(
                np.sum(
                    (
                        forecast_mask[forecast_slice]
                        & observed_dilated[observed_slice]
                    )
                    * area[None, :, :]
                )
            )
            intersection = min(observed_hit_weight, forecast_hit_weight)
            denominator = observed_total + forecast_total - intersection
            by_radius_and_shift[(int(radius), shift)] = (
                intersection / denominator if denominator > 0 else 0.0
            )
    results: dict[tuple[int, int], tuple[float, int]] = {}
    for radius in radii:
        for tolerance in tolerances:
            best_iou = -1.0
            best_shift = 0
            for shift in range(-int(tolerance), int(tolerance) + 1):
                iou = by_radius_and_shift[(int(radius), shift)]
                if iou > best_iou:
                    best_iou = iou
                    best_shift = shift
            results[(int(radius), int(tolerance))] = (
                max(0.0, float(best_iou)),
                int(best_shift),
            )
    return results


def tolerant_event_iou_surface(
    observed: Event,
    forecast: Event,
    cell_area_km2: np.ndarray,
    *,
    temporal_tolerances_days: Sequence[int],
    radii_pixels: Sequence[int],
    geometry_cache: dict[tuple[int, int], _EventGeometry] | None = None,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Evaluate symmetric tolerant IoU for an entire radius-time surface.

    A voxel is a spatial hit when it lies within Chebyshev grid distance
    ``radius_pixels`` of the other event on the same shifted day.  The tolerant
    intersection is the smaller of the two directional, area-weighted hit
    volumes, which keeps the intersection bounded by both original event
    volumes.  Spatial indices and per-shift nearest distances are reused across
    every requested tolerance instead of explicitly materializing dilated
    voxel sets.
    """
    tolerances = tuple(sorted(set(int(value) for value in temporal_tolerances_days)))
    radii = tuple(sorted(set(max(0, int(value)) for value in radii_pixels)))
    if not tolerances or not radii:
        return {}
    maximum_shift = max(tolerances)
    maximum_radius = max(radii)
    minimum_ordinal = min(
        int(np.min(observed.voxels[:, 0])),
        int(np.min(forecast.voxels[:, 0])),
    ) - maximum_shift
    maximum_ordinal = max(
        int(np.max(observed.voxels[:, 0])),
        int(np.max(forecast.voxels[:, 0])),
    ) + maximum_shift
    dense_cells = (
        (maximum_ordinal - minimum_ordinal + 1)
        * int(observed.nlat)
        * int(observed.nlon)
    )
    total_sparse_voxels = len(observed.voxels) + len(forecast.voxels)
    if dense_cells <= 25_000_000 and total_sparse_voxels >= 50_000:
        return _dense_tolerant_event_iou_surface(
            observed,
            forecast,
            cell_area_km2,
            tolerances=tolerances,
            radii=radii,
        )
    time_scale = max(
        int(observed.nlat),
        int(observed.nlon),
        int(forecast.nlat),
        int(forecast.nlon),
        maximum_radius,
    ) + 1
    observed_geometry = _prepare_event_geometry(
        observed,
        cell_area_km2,
        time_scale=time_scale,
        cache=geometry_cache,
    )
    forecast_geometry = _prepare_event_geometry(
        forecast,
        cell_area_km2,
        time_scale=time_scale,
        cache=geometry_cache,
    )
    by_radius_and_shift: dict[tuple[int, int], float] = {}
    for shift in range(-maximum_shift, maximum_shift + 1):
        observed_hits = _nearest_hit_weights(
            observed_geometry,
            forecast_geometry,
            source_time_adjustment_days=-shift,
            radii_pixels=radii,
        )
        forecast_hits = _nearest_hit_weights(
            forecast_geometry,
            observed_geometry,
            source_time_adjustment_days=shift,
            radii_pixels=radii,
        )
        for radius in radii:
            intersection = min(observed_hits[radius], forecast_hits[radius])
            denominator = (
                observed_geometry.total_weight
                + forecast_geometry.total_weight
                - intersection
            )
            iou = intersection / denominator if denominator > 0 else 0.0
            by_radius_and_shift[(radius, shift)] = float(iou)
    results: dict[tuple[int, int], tuple[float, int]] = {}
    for radius in radii:
        for tolerance in tolerances:
            best_iou = -1.0
            best_shift = 0
            for shift in range(-tolerance, tolerance + 1):
                iou = by_radius_and_shift[(radius, shift)]
                if iou > best_iou:
                    best_iou = iou
                    best_shift = shift
            results[(radius, tolerance)] = (max(0.0, best_iou), best_shift)
    return results


def tolerant_event_iou(
    observed: Event,
    forecast: Event,
    cell_area_km2: np.ndarray,
    *,
    max_shift_days: int,
    radius_pixels: int,
) -> tuple[float, int]:
    """Maximum symmetric area-weighted IoU across allowed space-time tolerance."""
    surface = tolerant_event_iou_surface(
        observed,
        forecast,
        cell_area_km2,
        temporal_tolerances_days=[int(max_shift_days)],
        radii_pixels=[int(radius_pixels)],
    )
    return surface[(max(0, int(radius_pixels)), int(max_shift_days))]


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        return float("inf")
    return abs(math.log(numerator / denominator))


def event_pair_cost(
    observed: Event,
    forecast: Event,
    cell_area_km2: np.ndarray,
    config: MatchConfig,
) -> tuple[float, dict[str, float | int]] | None:
    timing = float(abs(forecast.peak_ordinal - observed.peak_ordinal))
    distance = haversine_km(
        observed.centroid_lat,
        observed.centroid_lon,
        forecast.centroid_lat,
        forecast.centroid_lon,
    )
    if timing > config.max_timing_days or distance > config.max_distance_km:
        return None
    iou, best_shift = tolerant_event_iou(
        observed,
        forecast,
        cell_area_km2,
        max_shift_days=config.max_timing_days,
        radius_pixels=config.tolerant_iou_radius_pixels,
    )
    duration_ratio = forecast.duration_days / observed.duration_days
    area_ratio = forecast.max_daily_area_km2 / observed.max_daily_area_km2
    intensity_error = (
        abs(forecast.standardized_quantile_intensity - observed.standardized_quantile_intensity)
        if np.isfinite(forecast.standardized_quantile_intensity)
        and np.isfinite(observed.standardized_quantile_intensity)
        else 0.0
    )
    cost = (
        config.timing_weight * timing / max(1.0, float(config.max_timing_days))
        + config.distance_weight * distance / max(1.0, config.max_distance_km)
        + config.iou_weight * (1.0 - iou)
        + config.duration_weight * _safe_log_ratio(forecast.duration_days, observed.duration_days)
        + config.area_weight * _safe_log_ratio(forecast.max_daily_area_km2, observed.max_daily_area_km2)
        + config.intensity_weight * intensity_error
    )
    return float(cost), {
        "timing_error_days": timing,
        "centroid_distance_km": distance,
        "tolerant_iou": iou,
        "best_shift_days": best_shift,
        "duration_ratio": duration_ratio,
        "area_ratio": area_ratio,
        "intensity_error": intensity_error,
    }


def match_events_one_to_one(
    observed_events: Sequence[Event],
    forecast_events: Sequence[Event],
    cell_area_km2: np.ndarray,
    config: MatchConfig,
    *,
    member: int,
    track: str,
    observed_geometry_cache: dict[tuple[int, int], _EventGeometry] | None = None,
) -> tuple[list[Match], list[Event], list[Event]]:
    """Globally match one member's events for a single tolerance configuration."""
    return match_events_over_configs(
        observed_events,
        forecast_events,
        cell_area_km2,
        [config],
        member=member,
        track=track,
        observed_geometry_cache=observed_geometry_cache,
    )[0]


def match_events_over_configs(
    observed_events: Sequence[Event],
    forecast_events: Sequence[Event],
    cell_area_km2: np.ndarray,
    configs: Sequence[MatchConfig],
    *,
    member: int,
    track: str,
    observed_geometry_cache: dict[tuple[int, int], _EventGeometry] | None = None,
) -> list[tuple[list[Match], list[Event], list[Event]]]:
    """Match once across many tolerances, reusing pair geometry and IoU queries."""
    observed = list(observed_events)
    forecast = list(forecast_events)
    configurations = list(configs)
    if not configurations:
        return []
    if not observed:
        return [([], [], forecast.copy()) for _ in configurations]
    if not forecast:
        return [([], observed.copy(), []) for _ in configurations]
    costs = [
        np.full(
            (len(observed), len(forecast)),
            config.unmatched_cost,
            dtype=float,
        )
        for config in configurations
    ]
    details: list[dict[tuple[int, int], dict[str, float | int]]] = [
        {} for _ in configurations
    ]
    geometry_cache: dict[tuple[int, int], _EventGeometry] = (
        observed_geometry_cache
        if observed_geometry_cache is not None
        else {}
    )
    for obs_index, obs_event in enumerate(observed):
        for pred_index, pred_event in enumerate(forecast):
            timing = float(abs(pred_event.peak_ordinal - obs_event.peak_ordinal))
            distance = haversine_km(
                obs_event.centroid_lat,
                obs_event.centroid_lon,
                pred_event.centroid_lat,
                pred_event.centroid_lon,
            )
            eligible = [
                index
                for index, config in enumerate(configurations)
                if timing <= config.max_timing_days
                and distance <= config.max_distance_km
            ]
            if not eligible:
                continue
            radii = [
                configurations[index].tolerant_iou_radius_pixels
                for index in eligible
            ]
            tolerances = [
                configurations[index].max_timing_days for index in eligible
            ]
            iou_surface = tolerant_event_iou_surface(
                obs_event,
                pred_event,
                cell_area_km2,
                temporal_tolerances_days=tolerances,
                radii_pixels=radii,
                geometry_cache=geometry_cache,
            )
            duration_ratio = pred_event.duration_days / obs_event.duration_days
            area_ratio = (
                pred_event.max_daily_area_km2 / obs_event.max_daily_area_km2
            )
            intensity_error = (
                abs(
                    pred_event.standardized_quantile_intensity
                    - obs_event.standardized_quantile_intensity
                )
                if np.isfinite(pred_event.standardized_quantile_intensity)
                and np.isfinite(obs_event.standardized_quantile_intensity)
                else 0.0
            )
            for config_index in eligible:
                config = configurations[config_index]
                iou, best_shift = iou_surface[
                    (
                        int(config.tolerant_iou_radius_pixels),
                        int(config.max_timing_days),
                    )
                ]
                cost = (
                    config.timing_weight
                    * timing
                    / max(1.0, float(config.max_timing_days))
                    + config.distance_weight
                    * distance
                    / max(1.0, config.max_distance_km)
                    + config.iou_weight * (1.0 - iou)
                    + config.duration_weight
                    * _safe_log_ratio(
                        pred_event.duration_days,
                        obs_event.duration_days,
                    )
                    + config.area_weight
                    * _safe_log_ratio(
                        pred_event.max_daily_area_km2,
                        obs_event.max_daily_area_km2,
                    )
                    + config.intensity_weight * intensity_error
                )
                if not np.isfinite(cost) or cost >= config.unmatched_cost:
                    continue
                costs[config_index][obs_index, pred_index] = float(cost)
                details[config_index][(obs_index, pred_index)] = {
                    "timing_error_days": timing,
                    "centroid_distance_km": distance,
                    "tolerant_iou": iou,
                    "best_shift_days": best_shift,
                    "duration_ratio": duration_ratio,
                    "area_ratio": area_ratio,
                    "intensity_error": intensity_error,
                }
    results: list[tuple[list[Match], list[Event], list[Event]]] = []
    for config_index, config in enumerate(configurations):
        row_index, col_index = linear_sum_assignment(costs[config_index])
        matches: list[Match] = []
        matched_obs: set[int] = set()
        matched_pred: set[int] = set()
        for obs_index_raw, pred_index_raw in zip(row_index, col_index):
            obs_index = int(obs_index_raw)
            pred_index = int(pred_index_raw)
            if costs[config_index][obs_index, pred_index] >= config.unmatched_cost:
                continue
            detail = details[config_index][(obs_index, pred_index)]
            matches.append(
                Match(
                    observed_event_id=observed[obs_index].event_id,
                    forecast_event_id=forecast[pred_index].event_id,
                    member=int(member),
                    track=track,
                    cost=float(costs[config_index][obs_index, pred_index]),
                    timing_error_days=float(detail["timing_error_days"]),
                    centroid_distance_km=float(detail["centroid_distance_km"]),
                    tolerant_iou=float(detail["tolerant_iou"]),
                    best_shift_days=int(detail["best_shift_days"]),
                    duration_ratio=float(detail["duration_ratio"]),
                    area_ratio=float(detail["area_ratio"]),
                    intensity_error=float(detail["intensity_error"]),
                )
            )
            matched_obs.add(obs_index)
            matched_pred.add(pred_index)
        misses = [
            event for index, event in enumerate(observed) if index not in matched_obs
        ]
        false_alarms = [
            event for index, event in enumerate(forecast) if index not in matched_pred
        ]
        results.append((matches, misses, false_alarms))
    if observed_geometry_cache is not None:
        observed_ids = {id(event) for event in observed}
        for key in list(observed_geometry_cache):
            if key[0] not in observed_ids:
                del observed_geometry_cache[key]
    return results


def event_support(
    observed_events: Sequence[Event],
    matches_by_member: Mapping[int, Sequence[Match]],
    valid_members: Iterable[int],
) -> pd.DataFrame:
    """Compute matched-event support with valid members as the sole denominator."""
    members = sorted(set(int(member) for member in valid_members))
    denominator = len(members)
    if denominator == 0:
        raise ValueError("Event support requires at least one valid member.")
    match_sets = {
        member: {match.observed_event_id for match in matches_by_member.get(member, ())}
        for member in members
    }
    rows = []
    for event in observed_events:
        supporting = [member for member in members if event.event_id in match_sets[member]]
        rows.append(
            {
                "observed_event_id": event.event_id,
                "track": event.track,
                "year": event.year,
                "region_id": event.region_id,
                "n_supporting_members": len(supporting),
                "n_valid_members": denominator,
                "member_support_probability": len(supporting) / denominator,
                "supporting_members": json.dumps(supporting, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def count_crps(samples: Sequence[float], observation: float) -> float:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(observation):
        return float("nan")
    return float(
        np.mean(np.abs(values - observation))
        - 0.5 * np.mean(np.abs(values[:, np.newaxis] - values[np.newaxis, :]))
    )


def arrival_scores(
    arrival_days: Sequence[float],
    observed_arrival_day: float | None,
    horizon_days: int,
) -> dict[str, float]:
    """Score first arrival while retaining no-event members as right-censored."""
    horizon = int(horizon_days)
    censor_value = float(horizon + 1)
    samples = np.asarray(arrival_days, dtype=float)
    samples = np.where(np.isfinite(samples), samples, censor_value)
    observed = censor_value if observed_arrival_day is None or not np.isfinite(observed_arrival_day) else float(observed_arrival_day)
    days = np.arange(0, horizon + 1, dtype=float)
    forecast_cdf = np.asarray([np.mean(samples <= day) for day in days])
    observed_cdf = (days >= observed).astype(float)
    return {
        "first_arrival_crps_days": count_crps(samples, observed),
        "median_absolute_timing_error_days": (
            float(abs(np.median(samples) - observed)) if observed <= horizon else float("nan")
        ),
        "arrival_probability_by_window_end": float(np.mean(samples <= horizon)),
        "integrated_arrival_brier_score": float(np.mean((forecast_cdf - observed_cdf) ** 2)),
        "censored_member_fraction": float(np.mean(samples > horizon)),
    }


def fractions_skill_score(
    observed_mask: np.ndarray,
    forecast_mask: np.ndarray,
    radius_pixels: int,
    temporal_radius_days: int = 0,
) -> float:
    """Spatiotemporal coverage score, explicitly not an event probability."""
    obs = np.asarray(observed_mask, dtype=float)
    pred = np.asarray(forecast_mask, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError("Observed and forecast masks must have identical shapes.")
    if obs.ndim != 3:
        raise ValueError("FSS masks must be time-latitude-longitude arrays.")
    radius = int(radius_pixels)
    tau = int(temporal_radius_days)
    if radius < 0 or tau < 0:
        raise ValueError("FSS neighborhood radii must be nonnegative.")
    size = (2 * tau + 1, 2 * radius + 1, 2 * radius + 1)
    obs_fraction = uniform_filter(obs, size=size, mode="constant", cval=0.0)
    pred_fraction = uniform_filter(pred, size=size, mode="constant", cval=0.0)
    numerator = float(np.mean((pred_fraction - obs_fraction) ** 2))
    denominator = float(np.mean(pred_fraction**2 + obs_fraction**2))
    return 1.0 if denominator == 0 else float(1.0 - numerator / denominator)


def circular_shift_event_path(
    events: Sequence[Event],
    season_ordinals: Sequence[int],
    shift_days: int,
) -> list[Event]:
    """Circularly shift complete event masks while preserving their marked shapes."""
    season = [int(value) for value in season_ordinals]
    if not season:
        raise ValueError("season_ordinals cannot be empty.")
    index = {ordinal: idx for idx, ordinal in enumerate(season)}
    shift = int(shift_days) % len(season)
    shifted_events: list[Event] = []
    for event in events:
        shifted_voxels = event.voxels.copy()
        for row in range(shifted_voxels.shape[0]):
            original = int(shifted_voxels[row, 0])
            if original not in index:
                raise ValueError(f"Event day {original} is outside the supplied season.")
            shifted_voxels[row, 0] = season[(index[original] + shift) % len(season)]
        unique_days = np.unique(shifted_voxels[:, 0])
        original_peak_index = index[event.peak_ordinal]
        shifted_peak = season[(original_peak_index + shift) % len(season)]
        shifted_events.append(
            replace(
                event,
                event_id=f"{event.event_id}_shift{shift:+d}",
                start_ordinal=int(unique_days.min()),
                peak_ordinal=int(shifted_peak),
                end_ordinal=int(unique_days.max()),
                voxels=shifted_voxels,
            )
        )
    return shifted_events


def allowed_circular_shifts(season_length: int, minimum_shift_days: int) -> np.ndarray:
    length = int(season_length)
    minimum = int(minimum_shift_days)
    if length <= 1 or minimum < 1:
        raise ValueError("Season length must exceed one and minimum shift must be positive.")
    shifts = np.arange(1, length, dtype=int)
    circular_distance = np.minimum(shifts, length - shifts)
    allowed = shifts[circular_distance >= minimum]
    if allowed.size == 0:
        raise ValueError("No nontrivial circular shifts satisfy the minimum shift.")
    return allowed


def marginal_preserving_independence_surrogate(
    mask: np.ndarray,
    *,
    minimum_shift_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently circular-shift each cell's complete time series.

    Every cell keeps exactly the same number of event days (its binary marginal)
    while cross-cell temporal dependence is disrupted.
    """
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Independence surrogate expects (time,lat,lon).")
    allowed = allowed_circular_shifts(values.shape[0], minimum_shift_days)
    output = np.empty_like(values)
    for lat_index in range(values.shape[1]):
        for lon_index in range(values.shape[2]):
            shift = int(rng.choice(allowed))
            output[:, lat_index, lon_index] = np.roll(
                values[:, lat_index, lon_index],
                shift,
            )
    return output


def bootstrap_year_interval(
    rows: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    n_replicates: int,
    random_seed: int,
    confidence_levels: Sequence[float] = (0.90, 0.95),
) -> pd.DataFrame:
    """Bootstrap complete years; all rows and members within a sampled year stay together."""
    if "year" not in rows:
        raise ValueError("Year-block bootstrap requires a year column.")
    years = np.asarray(sorted(pd.to_numeric(rows["year"], errors="coerce").dropna().astype(int).unique()))
    if years.size == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(int(random_seed))
    sampled_indices = rng.integers(
        0,
        len(years),
        size=(int(n_replicates), len(years)),
    )
    numeric_year = pd.to_numeric(rows["year"], errors="coerce")
    estimates: dict[str, np.ndarray] = {}
    for metric in metric_columns:
        values = pd.to_numeric(rows.get(metric), errors="coerce")
        yearly_sums = np.asarray(
            [
                values[numeric_year == year].sum(skipna=True)
                for year in years
            ],
            dtype=float,
        )
        yearly_counts = np.asarray(
            [
                values[numeric_year == year].count()
                for year in years
            ],
            dtype=float,
        )
        sampled_sums = yearly_sums[sampled_indices].sum(axis=1)
        sampled_counts = yearly_counts[sampled_indices].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            estimates[metric] = np.where(
                sampled_counts > 0,
                sampled_sums / sampled_counts,
                np.nan,
            )
    output = []
    for metric, values in estimates.items():
        array = np.asarray(values, dtype=float)
        record: dict[str, object] = {
            "metric": metric,
            "estimate": float(pd.to_numeric(rows.get(metric), errors="coerce").mean()),
            "bootstrap_replicates": int(n_replicates),
        }
        for level in confidence_levels:
            alpha = 1.0 - float(level)
            suffix = int(round(100 * float(level)))
            finite = array[np.isfinite(array)]
            record[f"ci{suffix}_lower"] = (
                float(np.quantile(finite, alpha / 2.0)) if finite.size else np.nan
            )
            record[f"ci{suffix}_upper"] = (
                float(np.quantile(finite, 1.0 - alpha / 2.0)) if finite.size else np.nan
            )
        output.append(record)
    return pd.DataFrame(output)


def empirical_crps(samples: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Vectorized ensemble CRPS with member axis first and O(M) memory.

    The pairwise ensemble term is evaluated from order statistics rather than
    materializing an ``M x M`` member array.
    """
    ensemble = np.asarray(samples, dtype=float)
    obs = np.asarray(observation, dtype=float)
    finite = np.isfinite(ensemble)
    count = finite.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        first = np.sum(
            np.where(finite, np.abs(ensemble - obs[np.newaxis, ...]), 0.0),
            axis=0,
        ) / count
    ordered = np.sort(np.where(finite, ensemble, np.inf), axis=0)
    ordered_finite = np.isfinite(ordered)
    rank_shape = (ensemble.shape[0],) + (1,) * obs.ndim
    ranks = np.arange(1, ensemble.shape[0] + 1, dtype=float).reshape(rank_shape)
    coefficients = 2.0 * ranks - count[np.newaxis, ...] - 1.0
    safe_ordered = np.where(ordered_finite, ordered, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pairwise_half = np.sum(
            safe_ordered * coefficients,
            axis=0,
        ) / (count**2)
    result = first - pairwise_half
    return np.where((count > 0) & np.isfinite(obs), result, np.nan)


def tail_weighted_crps(
    samples: np.ndarray,
    observation: np.ndarray,
    tail_threshold: np.ndarray,
) -> np.ndarray:
    """Threshold-weighted CRPS via the standard upper-tail censoring transform."""
    ensemble = np.asarray(samples, dtype=float)
    obs = np.asarray(observation, dtype=float)
    threshold = np.asarray(tail_threshold, dtype=float)
    return empirical_crps(
        np.maximum(ensemble, threshold[np.newaxis, ...]),
        np.maximum(obs, threshold),
    )
