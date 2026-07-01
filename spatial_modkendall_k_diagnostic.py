#!/usr/bin/env python3
"""Spatial top-k hotspot diagnostics for ACE2/ERA5 fields.

This is the grid-cell analogue of modkendall_k_diagnostic.py. Instead of
ranking years as the objects, each event/year ranks CONUS grid cells by the
ACE2 score and the ERA5 score, then sweeps the Zheng-Lo truncation k.

Outputs:
  spatial_modkendall_k_sweep.csv
      One row per event and candidate k.
  spatial_modkendall_event_summary.csv
      Peak/plateau k choices per event, within the configured choice fraction
      range. By default, choices ignore the tiniest top-cell fractions because
      one or two matching cells are not a useful spatial scale.
  spatial_modkendall_selected_points.csv
      Figure-1-style point data for the selected event.
  spatial_modkendall_k_diagnostic.png
      Multiple k traces, selected-event scatter, and selected-event top-cell map.

By default, the trace y-axis is top-k overlap SNR:
    (observed top-k overlap - expected random overlap)
    / random-overlap standard deviation.

The Zheng-Lo modified-Kendall z statistic is still written to the sweep CSV for
comparison, but is not the default plotting/selection score because its iid
object null can be too sharp for spatially autocorrelated CONUS grid cells.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

mpl_cache = Path(tempfile.gettempdir()) / "ace2_matplotlib_cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mod_kendall_metric import modified_kendall_null, rank_scores  # noqa: E402


DEFAULT_FREQ_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/lag_may/spatial_modkendall_k_diagnostic"
DEFAULT_FORCING_DIR = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"
DEFAULT_CONUS_LAT = (25.0, 53.0)
DEFAULT_CONUS_LON = (235.0, 295.0)
SCORE_KEYS = {
    "overlap_snr": ("overlap_snr", "top-k overlap SNR"),
    "overlap_lift": ("overlap_lift", "top-k overlap lift"),
    "overlap_excess_fraction": ("overlap_excess_fraction", "excess top-k overlap fraction"),
    "jaccard": ("top_jaccard", "top-k Jaccard overlap"),
    "modkendall_z": ("z", "modified-Kendall z(k)"),
}


@dataclass(frozen=True)
class SpatialEvent:
    index: int
    label: str
    pred: np.ndarray
    obs: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class EventSummary:
    event_index: int
    event: str
    n: int
    score_name: str
    k_peak: int
    score_peak: float
    k_plateau: int
    score_plateau: float
    top_fraction_peak: float
    top_fraction_plateau: float
    top_both_plateau: int
    modkendall_z_at_peak: float
    modkendall_z_at_plateau: float
    overlap_snr_at_peak: float
    overlap_snr_at_plateau: float
    overlap_lift_at_plateau: float


def parse_bounds(text: Optional[str], default: Optional[tuple[float, float]] = None):
    if text is None:
        return default
    lo, hi = text.split(",", 1)
    return float(lo), float(hi)


def parse_k_values(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None:
        return None
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals:
        raise ValueError("--k-values was provided but no integers were parsed")
    return np.asarray(vals, dtype=int)


def coord_name(da, candidates: Sequence[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in da.dims}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    lowered_coords = {name.lower(): name for name in da.coords}
    for cand in candidates:
        if cand in lowered_coords:
            return lowered_coords[cand]
    return None


def subset_lat(da, lat_name: str, bounds: Optional[tuple[float, float]]):
    if bounds is None:
        return da
    lo, hi = bounds
    lat = da[lat_name]
    ascending = bool(float(lat[0]) <= float(lat[-1]))
    return da.sel({lat_name: slice(lo, hi) if ascending else slice(hi, lo)})


def subset_lon(da, lon_name: str, bounds: Optional[tuple[float, float]]):
    if bounds is None:
        return da
    lo, hi = bounds
    lon = da[lon_name]
    if float(lon.max()) <= 180.0 and lo > 180.0:
        lo -= 360.0
        hi -= 360.0
    ascending = bool(float(lon[0]) <= float(lon[-1]))
    if lo <= hi:
        return da.sel({lon_name: slice(lo, hi) if ascending else slice(hi, lo)})
    return da.where((lon >= lo) | (lon <= hi), drop=True)


def format_coord_value(value) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def event_labels(stacked_da) -> list[str]:
    index = stacked_da.indexes.get("event")
    if index is not None:
        nlevels = getattr(index, "nlevels", 1)
        if nlevels > 1:
            names = [name or f"level{i}" for i, name in enumerate(index.names)]
            return [
                ", ".join(f"{name}={format_coord_value(val)}" for name, val in zip(names, item))
                for item in index
            ]
        name = None
        if getattr(index, "names", None):
            name = index.names[0]
        name = name or index.name or "event"
        values = [item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in index]
        return [f"{name}={format_coord_value(item)}" for item in values]
    return [f"event={i}" for i in range(stacked_da.sizes["event"])]


def target_lon_for_source(lon: np.ndarray, source_lon: np.ndarray) -> np.ndarray:
    target = np.asarray(lon, dtype=float).copy()
    if float(np.nanmax(source_lon)) > 180.0 and float(np.nanmin(target)) < 0.0:
        target = np.mod(target, 360.0)
    elif float(np.nanmax(source_lon)) <= 180.0 and float(np.nanmax(target)) > 180.0:
        target = ((target + 180.0) % 360.0) - 180.0
    return target


def load_land_mask_from_forcing(
    lat: np.ndarray,
    lon: np.ndarray,
    forcing_dir: Path,
) -> np.ndarray | None:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required for land/ocean masking") from exc

    forcing_file = next(forcing_dir.glob("forcing_*.nc"), None)
    if forcing_file is None:
        return None
    with xr.open_dataset(forcing_file) as ds:
        if "land_fraction" not in ds:
            raise ValueError(f"{forcing_file} does not contain land_fraction")
        lf = ds["land_fraction"]
        lat_name = "latitude" if "latitude" in lf.dims else "lat"
        lon_name = "longitude" if "longitude" in lf.dims else "lon"
        lon_target = target_lon_for_source(lon, np.asarray(lf[lon_name].values, dtype=float))
        lf2 = lf.interp({lat_name: lat, lon_name: lon_target}, method="nearest")
    return np.asarray(lf2.values > 0.5, dtype=bool)


def load_spatial_events(
    path: Path,
    pred_var: str,
    obs_var: str,
    lat_bounds: Optional[tuple[float, float]],
    lon_bounds: Optional[tuple[float, float]],
    cell_domain: str,
    forcing_dir: Path,
) -> tuple[list[SpatialEvent], np.ndarray, np.ndarray]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required for NetCDF input") from exc

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate the seasonal frequency NetCDF first, "
            "or pass --freq-nc to an existing ACE2/ERA5 field file."
        )

    with xr.open_dataset(path) as ds:
        if pred_var not in ds or obs_var not in ds:
            raise ValueError(f"{path} must contain variables {pred_var!r} and {obs_var!r}")

        pred_da = ds[pred_var]
        obs_da = ds[obs_var]
        lat_name = coord_name(pred_da, ("lat", "latitude", "grid_yt", "y"))
        lon_name = coord_name(pred_da, ("lon", "longitude", "grid_xt", "x"))
        if lat_name is None or lon_name is None:
            raise ValueError(f"could not identify latitude/longitude dims in {pred_var!r}")

        pred_da = subset_lat(pred_da, lat_name, lat_bounds)
        obs_da = subset_lat(obs_da, lat_name, lat_bounds)
        pred_da = subset_lon(pred_da, lon_name, lon_bounds)
        obs_da = subset_lon(obs_da, lon_name, lon_bounds)
        pred_da, obs_da = xr.align(pred_da, obs_da, join="inner")

        event_dims = [d for d in pred_da.dims if d not in (lat_name, lon_name)]
        if event_dims:
            pred_s = pred_da.stack(event=event_dims).transpose("event", lat_name, lon_name)
            obs_s = obs_da.stack(event=event_dims).transpose("event", lat_name, lon_name)
            labels = event_labels(pred_s)
        else:
            pred_s = pred_da.expand_dims(event=["field"]).transpose("event", lat_name, lon_name)
            obs_s = obs_da.expand_dims(event=["field"]).transpose("event", lat_name, lon_name)
            labels = ["field"]

        lat = np.asarray(pred_s[lat_name].values, dtype=float)
        lon = np.asarray(pred_s[lon_name].values, dtype=float)
        pred = np.asarray(pred_s.values, dtype=np.float64)
        obs = np.asarray(obs_s.values, dtype=np.float64)

    keep_mask = np.ones((lat.size, lon.size), dtype=bool)
    if cell_domain in ("land", "ocean"):
        land_mask = load_land_mask_from_forcing(lat, lon, forcing_dir)
        if land_mask is None:
            raise FileNotFoundError(
                f"cannot apply --cell-domain {cell_domain!r}: no forcing_*.nc found in {forcing_dir}"
            )
        keep_mask = land_mask if cell_domain == "land" else ~land_mask

    events = []
    for i, label in enumerate(labels):
        valid = np.isfinite(pred[i]) & np.isfinite(obs[i]) & keep_mask
        events.append(
            SpatialEvent(
                index=i,
                label=label,
                pred=pred[i],
                obs=obs[i],
                lat=lat,
                lon=lon,
                valid_mask=valid,
            )
        )
    return events, lat, lon


def candidate_k_values(
    n: int,
    *,
    k_min: int,
    k_max: Optional[int],
    explicit: Optional[np.ndarray],
    grid: str,
    n_k: int,
) -> np.ndarray:
    if n < 2:
        raise ValueError("need at least two valid grid cells")
    lo = max(2, int(k_min))
    hi = min(n, int(k_max) if k_max is not None else n)
    if lo > hi:
        raise ValueError(f"empty k range after clipping to n={n}: {lo}>{hi}")
    if explicit is not None:
        vals = explicit[(explicit >= lo) & (explicit <= hi)]
        if vals.size == 0:
            raise ValueError(f"--k-values has no candidates inside [{lo}, {hi}]")
        return vals.astype(int)
    if grid == "all":
        return np.arange(lo, hi + 1, dtype=int)
    vals = np.unique(np.rint(np.geomspace(lo, hi, max(2, int(n_k)))).astype(int))
    return vals[(vals >= lo) & (vals <= hi)]


def fenwick_add(tree: np.ndarray, idx: int, delta: int = 1) -> None:
    while idx < tree.size:
        tree[idx] += delta
        idx += idx & -idx


def fenwick_sum(tree: np.ndarray, idx: int) -> int:
    total = 0
    while idx > 0:
        total += int(tree[idx])
        idx -= idx & -idx
    return total


def dominance_counts(rank_x: np.ndarray, rank_y: np.ndarray) -> np.ndarray:
    """Count cells worse than each cell in both rankings.

    For each object i, D_i = #{j: rank_x[j] > rank_x[i] and rank_y[j] > rank_y[i]}.
    Then the truncated-rank agreement count at k is sum(D_i) over cells where
    both ranks are strictly above the truncation tie, i.e. max(rank_x, rank_y)<k.
    """
    rx = np.asarray(rank_x, dtype=float)
    ry = np.asarray(rank_y, dtype=float)
    if rx.shape != ry.shape:
        raise ValueError("rank arrays must have matching shapes")
    n = rx.size
    unique_y = np.unique(ry)
    y_pos = np.searchsorted(unique_y, ry) + 1
    order = np.argsort(-rx, kind="mergesort")
    out = np.zeros(n, dtype=np.int64)
    tree = np.zeros(unique_y.size + 1, dtype=np.int64)
    processed = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and rx[order[end]] == rx[order[start]]:
            end += 1
        group = order[start:end]
        for idx in group:
            out[idx] = processed - fenwick_sum(tree, int(y_pos[idx]))
        for idx in group:
            fenwick_add(tree, int(y_pos[idx]), 1)
            processed += 1
        start = end
    return out


def sweep_from_ranks(rank_pred: np.ndarray, rank_obs: np.ndarray, k_values: np.ndarray):
    n = int(rank_pred.size)
    dominance = dominance_counts(rank_pred, rank_obs)
    joint_rank = np.maximum(rank_pred, rank_obs)
    order = np.argsort(joint_rank, kind="mergesort")
    joint_sorted = joint_rank[order]
    dominance_csum = np.cumsum(dominance[order], dtype=np.float64)

    rows = []
    for k in k_values:
        kk = int(max(2, min(int(k), n)))
        idx = int(np.searchsorted(joint_sorted, kk, side="left"))
        agreements = float(dominance_csum[idx - 1]) if idx else 0.0
        rate = agreements / n / (n - 1)
        mean, var = modified_kendall_null(n, kk)
        z = (rate - mean) / sqrt(var) if var > 0 else float("nan")
        top_pred_count = int(np.count_nonzero(rank_pred < kk))
        top_obs_count = int(np.count_nonzero(rank_obs < kk))
        top_both_count = int(idx)
        expected_overlap = top_pred_count * top_obs_count / n
        if n > 1:
            overlap_var = (
                top_pred_count
                * top_obs_count
                * (n - top_pred_count)
                * (n - top_obs_count)
                / (n * n * (n - 1))
            )
        else:
            overlap_var = float("nan")
        overlap_excess = top_both_count - expected_overlap
        overlap_snr = overlap_excess / sqrt(overlap_var) if overlap_var > 0 else float("nan")
        overlap_lift = top_both_count / expected_overlap if expected_overlap > 0 else float("nan")
        top_union_count = top_pred_count + top_obs_count - top_both_count
        top_jaccard = top_both_count / top_union_count if top_union_count > 0 else float("nan")
        rows.append(
            {
                "k": kk,
                "top_cells_individually_ordered": kk - 1,
                "top_fraction": (kk - 1) / n,
                "z": float(z),
                "agreement_rate": float(rate),
                "null_mean": float(mean),
                "null_variance": float(var),
                "agreement_count": agreements,
                "top_pred_count": top_pred_count,
                "top_obs_count": top_obs_count,
                "top_both_count": top_both_count,
                "expected_top_overlap": float(expected_overlap),
                "top_overlap_excess": float(overlap_excess),
                "overlap_excess_fraction": float(overlap_excess / n),
                "overlap_lift": float(overlap_lift),
                "overlap_snr": float(overlap_snr),
                "top_jaccard": float(top_jaccard),
            }
        )
    return rows


def choose_k(rows, score_key: str, plateau_fraction: float) -> tuple[int, float, int, float, int]:
    return choose_k_in_fraction_range(rows, score_key, plateau_fraction, 0.0, 1.0)


def choose_k_in_fraction_range(
    rows,
    score_key: str,
    plateau_fraction: float,
    min_fraction: float,
    max_fraction: float,
) -> tuple[int, float, int, float, int]:
    choice_rows = [
        r for r in rows
        if min_fraction <= float(r["top_fraction"]) <= max_fraction
    ]
    if not choice_rows:
        choice_rows = rows
    score = np.asarray([r[score_key] for r in choice_rows], dtype=float)
    k = np.asarray([r["k"] for r in choice_rows], dtype=int)
    finite = np.isfinite(score)
    if not finite.any():
        raise ValueError(f"all {score_key} values are NaN")
    peak_idx = int(np.nanargmax(score))
    k_peak = int(k[peak_idx])
    score_peak = float(score[peak_idx])
    if score_peak > 0:
        near = np.where(score >= plateau_fraction * score_peak)[0]
        plateau_idx = int(near[0])
    else:
        plateau_idx = peak_idx
    return (
        k_peak,
        score_peak,
        int(k[plateau_idx]),
        float(score[plateau_idx]),
        int(choice_rows[plateau_idx]["top_both_count"]),
    )


def event_vectors(event: SpatialEvent):
    valid = event.valid_mask
    pred = event.pred[valid]
    obs = event.obs[valid]
    lat2d, lon2d = np.meshgrid(event.lat, event.lon, indexing="ij")
    return pred, obs, lat2d[valid], lon2d[valid]


def analyze_event(
    event: SpatialEvent,
    k_values: np.ndarray,
    tie_method: str,
    score_key: str,
    plateau_fraction: float,
    choice_min_fraction: float,
    choice_max_fraction: float,
):
    pred, obs, _lat_v, _lon_v = event_vectors(event)
    if pred.size < 4:
        raise ValueError(f"{event.label}: need at least 4 finite paired cells")
    if np.nanstd(pred) < 1e-12 or np.nanstd(obs) < 1e-12:
        raise ValueError(f"{event.label}: predicted and observed fields must both vary")

    rank_pred = rank_scores(pred, descending=True, tie_method=tie_method)
    rank_obs = rank_scores(obs, descending=True, tie_method=tie_method)
    rows = sweep_from_ranks(rank_pred, rank_obs, k_values)
    k_peak, score_peak, k_plateau, score_plateau, top_both_plateau = choose_k_in_fraction_range(
        rows, score_key, plateau_fraction, choice_min_fraction, choice_max_fraction
    )
    rows_by_k = {int(r["k"]): r for r in rows}
    peak_row = rows_by_k[k_peak]
    plateau_row = rows_by_k[k_plateau]
    n = int(pred.size)
    summary = EventSummary(
        event_index=event.index,
        event=event.label,
        n=n,
        score_name=score_key,
        k_peak=k_peak,
        score_peak=score_peak,
        k_plateau=k_plateau,
        score_plateau=score_plateau,
        top_fraction_peak=(k_peak - 1) / n,
        top_fraction_plateau=(k_plateau - 1) / n,
        top_both_plateau=top_both_plateau,
        modkendall_z_at_peak=float(peak_row["z"]),
        modkendall_z_at_plateau=float(plateau_row["z"]),
        overlap_snr_at_peak=float(peak_row["overlap_snr"]),
        overlap_snr_at_plateau=float(plateau_row["overlap_snr"]),
        overlap_lift_at_plateau=float(plateau_row["overlap_lift"]),
    )
    return rows, summary, rank_pred, rank_obs


def write_sweep_csv(path: Path, all_rows):
    fieldnames = [
        "event_index",
        "event",
        "n",
        "k",
        "top_cells_individually_ordered",
        "top_fraction",
        "z",
        "agreement_rate",
        "null_mean",
        "null_variance",
        "agreement_count",
        "top_pred_count",
        "top_obs_count",
        "top_both_count",
        "expected_top_overlap",
        "top_overlap_excess",
        "overlap_excess_fraction",
        "overlap_lift",
        "overlap_snr",
        "top_jaccard",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def write_summary_csv(path: Path, summaries: Sequence[EventSummary]):
    fieldnames = list(EventSummary.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item.__dict__)


def write_selected_points_csv(
    path: Path,
    event: SpatialEvent,
    rank_pred: np.ndarray,
    rank_obs: np.ndarray,
    selected_k: int,
):
    pred, obs, lat_v, lon_v = event_vectors(event)
    top_pred = rank_pred < selected_k
    top_obs = rank_obs < selected_k
    rows, cols = np.where(event.valid_mask)
    fieldnames = [
        "row",
        "col",
        "lat",
        "lon",
        "pred",
        "obs",
        "pred_rank",
        "obs_rank",
        "in_pred_top",
        "in_obs_top",
        "in_both_top",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vals in zip(rows, cols, lat_v, lon_v, pred, obs, rank_pred, rank_obs, top_pred, top_obs):
            row, col, lat, lon, p, o, rp, ro, in_pred, in_obs = vals
            writer.writerow(
                {
                    "row": int(row),
                    "col": int(col),
                    "lat": float(lat),
                    "lon": float(lon),
                    "pred": float(p),
                    "obs": float(o),
                    "pred_rank": float(rp),
                    "obs_rank": float(ro),
                    "in_pred_top": bool(in_pred),
                    "in_obs_top": bool(in_obs),
                    "in_both_top": bool(in_pred and in_obs),
                }
            )


def lon_for_plot(lon: np.ndarray) -> np.ndarray:
    return lon - 360.0 if float(np.nanmean(lon)) > 180.0 else lon


def select_event_index(
    summaries: Sequence[EventSummary],
    *,
    event_index: Optional[int],
    event_label: Optional[str],
) -> int:
    if event_index is not None:
        for item in summaries:
            if item.event_index == event_index:
                return item.event_index
        raise ValueError(f"--event-index {event_index} is outside the analyzed events")
    if event_label:
        matches = [item.event_index for item in summaries if event_label in item.event]
        if not matches:
            raise ValueError(f"no event label contains {event_label!r}")
        return matches[0]
    return max(summaries, key=lambda item: item.score_peak).event_index


def selected_k_from_summary(summary: EventSummary, mode: str) -> int:
    if mode == "peak":
        return summary.k_peak
    if mode == "plateau":
        return summary.k_plateau
    raise ValueError(mode)


def category_map(event: SpatialEvent, rank_pred: np.ndarray, rank_obs: np.ndarray, selected_k: int):
    cat = np.full(event.pred.shape, np.nan, dtype=float)
    top_pred = rank_pred < selected_k
    top_obs = rank_obs < selected_k
    values = np.zeros(rank_pred.size, dtype=float)
    values[top_pred & ~top_obs] = 1.0
    values[~top_pred & top_obs] = 2.0
    values[top_pred & top_obs] = 3.0
    cat[event.valid_mask] = values
    return cat


def plot_diagnostic(
    path: Path,
    all_rows_by_event,
    summaries: Sequence[EventSummary],
    selected_event: SpatialEvent,
    selected_rows,
    rank_pred: np.ndarray,
    rank_obs: np.ndarray,
    selected_k: int,
    pred_label: str,
    obs_label: str,
    score_key: str,
    score_label: str,
    max_scatter_points: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor="white")

    ax = axes[0]
    common_x = None
    z_stack = []
    for rows in all_rows_by_event:
        x = np.asarray([r["top_fraction"] for r in rows], dtype=float)
        score = np.asarray([r[score_key] for r in rows], dtype=float)
        ax.plot(x, score, color="#4C78A8", alpha=0.22, linewidth=0.8)
        if common_x is None:
            common_x = x
            z_stack.append(score)
        elif x.shape == common_x.shape and np.allclose(x, common_x):
            z_stack.append(score)
    if common_x is not None and len(z_stack) >= 2:
        z_arr = np.vstack(z_stack)
        med = np.nanmedian(z_arr, axis=0)
        q25 = np.nanpercentile(z_arr, 25, axis=0)
        q75 = np.nanpercentile(z_arr, 75, axis=0)
        ax.fill_between(common_x, q25, q75, color="#1F77B4", alpha=0.16, linewidth=0)
        ax.plot(common_x, med, color="#1F77B4", linewidth=2.2, label="median trace")
    sx = np.asarray([r["top_fraction"] for r in selected_rows], dtype=float)
    sz = np.asarray([r[score_key] for r in selected_rows], dtype=float)
    ax.plot(sx, sz, color="#D62728", linewidth=1.7, label="selected event")
    ax.axvline((selected_k - 1) / selected_event.valid_mask.sum(), color="#D62728",
               linestyle="--", linewidth=1.1)
    if score_key in ("overlap_snr", "z", "overlap_excess_fraction"):
        ax.axhline(0.0, color="0.55", linewidth=0.9)
    if score_key == "overlap_snr":
        ax.axhline(2.0, color="#2CA02C", linestyle=":", linewidth=1.0, label="SNR=2")
    elif score_key == "z":
        ax.axhline(1.96, color="#2CA02C", linestyle=":", linewidth=1.0, label="z=1.96")
    elif score_key == "overlap_lift":
        ax.axhline(1.0, color="0.55", linewidth=0.9, label="random lift=1")
    ax.set_xlabel("top CONUS grid-cell fraction, (k - 1) / n")
    ax.set_ylabel(score_label)
    ax.set_title("Spatial top-k traces")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    pred, obs, _lat_v, _lon_v = event_vectors(selected_event)
    top_pred = rank_pred < selected_k
    top_obs = rank_obs < selected_k
    both = top_pred & top_obs
    one = top_pred ^ top_obs
    lower = ~(both | one)
    rng = np.random.default_rng(42)

    def scatter_mask(mask, color, label, size, alpha):
        idx = np.flatnonzero(mask)
        if idx.size > max_scatter_points:
            idx = rng.choice(idx, size=max_scatter_points, replace=False)
        axes[1].scatter(pred[idx], obs[idx], s=size, color=color, alpha=alpha,
                        label=label, edgecolors="none")

    scatter_mask(lower, "#4C78A8", "lower in both", 13, 0.45)
    scatter_mask(one, "#F58518", "top in one ranking", 24, 0.78)
    scatter_mask(both, "#D62728", "top in both", 30, 0.9)
    if np.any(top_pred):
        axes[1].axvline(float(np.nanmin(pred[top_pred])), color="#D62728",
                        linestyle="--", linewidth=1)
    if np.any(top_obs):
        axes[1].axhline(float(np.nanmin(obs[top_obs])), color="#D62728",
                        linestyle="--", linewidth=1)
    axes[1].set_xlabel(pred_label)
    axes[1].set_ylabel(obs_label)
    axes[1].set_title(f"Figure-1-style grid-cell scatter\n{selected_event.label}, k={selected_k}")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    cat = category_map(selected_event, rank_pred, rank_obs, selected_k)
    lon_plot = lon_for_plot(selected_event.lon)
    extent = [
        float(lon_plot[0]) - 0.5,
        float(lon_plot[-1]) + 0.5,
        float(selected_event.lat[0]) - 0.5,
        float(selected_event.lat[-1]) + 0.5,
    ]
    cmap = ListedColormap(["#D9D9D9", "#4C78A8", "#F58518", "#D62728"])
    cmap.set_bad("#F7F7F7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = axes[2].imshow(cat, origin="lower", extent=extent, aspect="auto",
                        interpolation="nearest", cmap=cmap, norm=norm)
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title("Top-cell geometry")
    axes[2].grid(True, alpha=0.25, linewidth=0.3)
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.8, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["lower", "ACE2 top", "ERA5 top", "both top"])

    selected_summary = next(item for item in summaries if item.event_index == selected_event.index)
    fig.suptitle(
        "Spatial top-k hotspot diagnostic | "
        f"selected {selected_event.label} | "
        f"{score_key} peak k={selected_summary.k_peak} score={selected_summary.score_peak:.2f} | "
        f"plateau k={selected_summary.k_plateau} score={selected_summary.score_plateau:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq-nc", type=Path, default=DEFAULT_FREQ_NC,
                        help="NetCDF containing paired ACE2/ERA5 spatial fields")
    parser.add_argument("--pred-var", default="ace2_freq")
    parser.add_argument("--obs-var", default="era5_freq")
    parser.add_argument("--domain", choices=("conus", "global", "bounds"), default="conus")
    parser.add_argument("--cell-domain", choices=("all", "land", "ocean"), default="all",
                        help="rank all cells, land cells only, or ocean cells only")
    parser.add_argument("--forcing-dir", type=Path, default=DEFAULT_FORCING_DIR,
                        help="directory containing forcing_*.nc with land_fraction")
    parser.add_argument("--lat-bounds", help="latitude bounds as min,max")
    parser.add_argument("--lon-bounds", help="longitude bounds as min,max in dataset convention")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int)
    parser.add_argument("--k-values", help="comma-separated explicit k values")
    parser.add_argument("--k-grid", choices=("all", "log"), default="all")
    parser.add_argument("--n-k", type=int, default=160,
                        help="number of log-spaced k values when --k-grid=log")
    parser.add_argument("--tie-method", choices=("average", "ordinal"), default="average")
    parser.add_argument("--score", choices=tuple(SCORE_KEYS.keys()), default="overlap_snr",
                        help="score used for the trace y-axis and peak/plateau k selection")
    parser.add_argument("--plateau-fraction", type=float, default=0.95,
                        help="choose the smallest k with score >= this fraction of peak score")
    parser.add_argument("--choice-min-fraction", type=float, default=0.01,
                        help="ignore smaller top-cell fractions when choosing peak/plateau k")
    parser.add_argument("--choice-max-fraction", type=float, default=1.0,
                        help="ignore larger top-cell fractions when choosing peak/plateau k")
    parser.add_argument("--selected-k", choices=("plateau", "peak"), default="plateau")
    parser.add_argument("--event-index", type=int)
    parser.add_argument("--event-label", help="substring match for selecting the figure event")
    parser.add_argument("--max-scatter-points", type=int, default=30000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not (0.0 < args.plateau_fraction <= 1.0):
        raise ValueError("--plateau-fraction must be in (0, 1]")
    if not (0.0 <= args.choice_min_fraction <= args.choice_max_fraction <= 1.0):
        raise ValueError("--choice-min-fraction and --choice-max-fraction must satisfy 0 <= min <= max <= 1")
    score_key, score_label = SCORE_KEYS[args.score]

    if args.domain == "conus":
        lat_bounds = parse_bounds(args.lat_bounds, DEFAULT_CONUS_LAT)
        lon_bounds = parse_bounds(args.lon_bounds, DEFAULT_CONUS_LON)
    elif args.domain == "bounds":
        lat_bounds = parse_bounds(args.lat_bounds)
        lon_bounds = parse_bounds(args.lon_bounds)
        if lat_bounds is None and lon_bounds is None:
            raise ValueError("--domain bounds requires --lat-bounds and/or --lon-bounds")
    else:
        lat_bounds = parse_bounds(args.lat_bounds)
        lon_bounds = parse_bounds(args.lon_bounds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events, lat, lon = load_spatial_events(
        args.freq_nc,
        args.pred_var,
        args.obs_var,
        lat_bounds,
        lon_bounds,
        args.cell_domain,
        args.forcing_dir,
    )
    if not events:
        raise ValueError("no events found in input")

    valid_counts = [int(e.valid_mask.sum()) for e in events]
    n_ref = max(valid_counts)
    k_values = candidate_k_values(
        n_ref,
        k_min=args.k_min,
        k_max=args.k_max,
        explicit=parse_k_values(args.k_values),
        grid=args.k_grid,
        n_k=args.n_k,
    )
    print(
        f"Loaded {args.freq_nc} with {len(events)} event(s), "
        f"grid={len(lat)}x{len(lon)}, cell-domain={args.cell_domain}, "
        f"valid cells/event={min(valid_counts)}-{max(valid_counts)}",
        flush=True,
    )
    print(f"Sweeping {len(k_values)} k values from {int(k_values[0])} to {int(k_values[-1])}", flush=True)
    print(
        "Choosing peak/plateau k within top-cell fraction range "
        f"[{args.choice_min_fraction:.3f}, {args.choice_max_fraction:.3f}]",
        flush=True,
    )
    print(f"Trace/choice score: {args.score} ({score_label})", flush=True)

    all_csv_rows = []
    all_rows_by_event = []
    summaries = []
    rank_cache = {}

    for event in events:
        local_k = k_values[k_values <= int(event.valid_mask.sum())]
        rows, summary, rank_pred, rank_obs = analyze_event(
            event,
            local_k,
            args.tie_method,
            score_key,
            args.plateau_fraction,
            args.choice_min_fraction,
            args.choice_max_fraction,
        )
        all_rows_by_event.append(rows)
        summaries.append(summary)
        rank_cache[event.index] = (rank_pred, rank_obs)
        for row in rows:
            out = {
                "event_index": event.index,
                "event": event.label,
                "n": summary.n,
                **row,
            }
            all_csv_rows.append(out)
        print(
            f"  {event.label}: peak k={summary.k_peak} {args.score}={summary.score_peak:.3f}; "
            f"plateau k={summary.k_plateau} {args.score}={summary.score_plateau:.3f}",
            flush=True,
        )

    selected_idx = select_event_index(
        summaries, event_index=args.event_index, event_label=args.event_label
    )
    selected_event = events[selected_idx]
    selected_summary = next(item for item in summaries if item.event_index == selected_idx)
    selected_k = selected_k_from_summary(selected_summary, args.selected_k)
    selected_rows = all_rows_by_event[selected_idx]
    rank_pred, rank_obs = rank_cache[selected_idx]

    sweep_csv = args.out_dir / "spatial_modkendall_k_sweep.csv"
    summary_csv = args.out_dir / "spatial_modkendall_event_summary.csv"
    points_csv = args.out_dir / "spatial_modkendall_selected_points.csv"
    fig_path = args.out_dir / "spatial_modkendall_k_diagnostic.png"
    write_sweep_csv(sweep_csv, all_csv_rows)
    write_summary_csv(summary_csv, summaries)
    write_selected_points_csv(points_csv, selected_event, rank_pred, rank_obs, selected_k)
    plot_diagnostic(
        fig_path,
        all_rows_by_event,
        summaries,
        selected_event,
        selected_rows,
        rank_pred,
        rank_obs,
        selected_k,
        args.pred_var,
        args.obs_var,
        score_key,
        score_label,
        args.max_scatter_points,
    )

    print(
        f"Selected event: {selected_event.label}; "
        f"{args.selected_k} k={selected_k}; "
        f"top fraction={(selected_k - 1) / selected_summary.n:.4f}",
        flush=True,
    )
    print(f"wrote {fig_path}", flush=True)
    print(f"wrote {sweep_csv}", flush=True)
    print(f"wrote {summary_csv}", flush=True)
    print(f"wrote {points_csv}", flush=True)


if __name__ == "__main__":
    main()
