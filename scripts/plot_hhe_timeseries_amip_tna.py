#!/usr/bin/env python3
"""Recreate the raw/detrended regional HHE-frequency time-series figure.

Required inputs are the gridded ERA5 JJA HHE-frequency file and the compact
per-year AMIP versus AMIP-TNA evaluations.  ACE2 lead-0, lead-1, and historical
series are optional because this repository does not always contain direct
counterparts to all SPEAR products in the reference figure.

All gridded frequency inputs are expected as fractions of JJA days.  They are
converted to percent after cosine-latitude, land-only regional averaging.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import xarray as xr
from scipy import stats


def coordinate_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in da.dims or name in da.coords:
            return name
    raise KeyError(f"none of {candidates} found in {da.dims}")


def box_mask(lat: np.ndarray, lon: np.ndarray, box: list[float]) -> np.ndarray:
    south, north, west, east = box
    lon360 = np.asarray(lon, dtype=float) % 360.0
    if west % 360.0 <= east % 360.0:
        x = (lon360 >= west % 360.0) & (lon360 <= east % 360.0)
    else:
        x = (lon360 >= west % 360.0) | (lon360 <= east % 360.0)
    return (
        (np.asarray(lat)[:, None] >= south)
        & (np.asarray(lat)[:, None] <= north)
        & x[None, :]
    )


def load_land(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        land = ds["land_fraction"]
        if "time" in land.dims:
            land = land.isel(time=0)
        return (
            np.asarray(ds[lat_name], dtype=float),
            np.asarray(ds[lon_name], dtype=float),
            np.asarray(land, dtype=float) > 0.5,
        )


def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(field)
    weights = np.cos(np.deg2rad(lat))[:, None] * valid
    denominator = np.sum(weights)
    return float(np.nansum(field * weights) / denominator) if denominator > 0 else np.nan


def regional_series(
    path: Path,
    variable: str,
    box: list[float],
    land_lat: np.ndarray,
    land_lon: np.ndarray,
    land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {path}; variables={list(ds.data_vars)}")
        da = ds[variable]
        year_name = coordinate_name(da, ("year", "years"))
        if da.ndim == 1:
            return np.asarray(da[year_name], dtype=int), 100.0 * np.asarray(da, dtype=float)
        lat_name = coordinate_name(da, ("lat", "latitude"))
        lon_name = coordinate_name(da, ("lon", "longitude"))
        da = da.transpose(year_name, lat_name, lon_name)
        lat = np.asarray(da[lat_name], dtype=float)
        lon = np.asarray(da[lon_name], dtype=float)
        if da.shape[1:] != land.shape or not (
            np.allclose(lat, land_lat) and np.allclose(lon % 360.0, land_lon % 360.0)
        ):
            raise ValueError(f"grid mismatch between {path} and land-mask forcing")
        mask = box_mask(lat, lon, box) & land
        values = np.asarray(da, dtype=float)
        result = np.asarray([weighted_mean(values[i], lat, mask) for i in range(len(values))])
        return np.asarray(da[year_name], dtype=int), 100.0 * result


def amip_series(
    evaluation_dir: Path,
    variable: str,
    box: list[float],
    land_lat: np.ndarray,
    land_lon: np.ndarray,
    land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    records: dict[int, float] = {}
    for path in sorted(evaluation_dir.glob("amip_global_minus_tna_*.nc")):
        match = re.search(r"_(\d{4})\.nc$", path.name)
        if not match:
            continue
        year = int(match.group(1))
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                continue
            da = ds[variable]
            lat_name = coordinate_name(da, ("lat", "latitude"))
            lon_name = coordinate_name(da, ("lon", "longitude"))
            lat = np.asarray(da[lat_name], dtype=float)
            lon = np.asarray(da[lon_name], dtype=float)
            if da.shape != land.shape or not (
                np.allclose(lat, land_lat) and np.allclose(lon % 360.0, land_lon % 360.0)
            ):
                raise ValueError(f"grid mismatch between {path} and land-mask forcing")
            mask = box_mask(lat, lon, box) & land
            records[year] = 100.0 * weighted_mean(np.asarray(da, dtype=float), lat, mask)
    years = np.asarray(sorted(records), dtype=int)
    return years, np.asarray([records[y] for y in years], dtype=float)


def tagged_evaluation_series(
    evaluation_root: Path,
    tag: str,
    variable: str,
    box: list[float],
    land_lat: np.ndarray,
    land_lon: np.ndarray,
    land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one map per year from evaluation_root/tag/tag_YEAR.nc."""

    records: dict[int, float] = {}
    directory = evaluation_root / tag
    for path in sorted(directory.glob(f"{tag}_*.nc")):
        match = re.search(r"_(\d{4})\.nc$", path.name)
        if not match:
            continue
        year = int(match.group(1))
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                continue
            da = ds[variable]
            lat_name = coordinate_name(da, ("lat", "latitude"))
            lon_name = coordinate_name(da, ("lon", "longitude"))
            lat = np.asarray(da[lat_name], dtype=float)
            lon = np.asarray(da[lon_name], dtype=float)
            if da.shape != land.shape or not (
                np.allclose(lat, land_lat)
                and np.allclose(lon % 360.0, land_lon % 360.0)
            ):
                raise ValueError(f"grid mismatch between {path} and land-mask forcing")
            mask = box_mask(lat, lon, box) & land
            records[year] = 100.0 * weighted_mean(
                np.asarray(da, dtype=float), lat, mask
            )
    years = np.asarray(sorted(records), dtype=int)
    return years, np.asarray([records[year] for year in years], dtype=float)


def detrend(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return result
    x = years[valid].astype(float)
    y = values[valid]
    slope, intercept = np.polyfit(x - x.mean(), y, 1)
    result[valid] = y - (slope * (x - x.mean()) + intercept)
    return result


def correlation(
    reference_years: np.ndarray,
    reference: np.ndarray,
    years: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, int]:
    ref = {int(y): float(v) for y, v in zip(reference_years, reference) if np.isfinite(v)}
    pairs = [(ref[int(y)], float(v)) for y, v in zip(years, values) if int(y) in ref and np.isfinite(v)]
    if len(pairs) < 3:
        return np.nan, np.nan, len(pairs)
    a, b = np.asarray(pairs, dtype=float).T
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan, len(pairs)
    r = stats.pearsonr(a, b)
    return float(r.statistic), float(r.pvalue), len(pairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era5-nc", required=True, type=Path)
    p.add_argument("--era5-var", default="era5_hi_freq")
    p.add_argument("--amip-evaluation-dir", type=Path)
    p.add_argument(
        "--amip-nc",
        type=Path,
        help="optional multi-year AMIP HHE-frequency NetCDF; preferred over per-year evaluations",
    )
    p.add_argument("--amip-var", default="ace2_hi_freq")
    p.add_argument("--land-mask-forcing", required=True, type=Path)
    p.add_argument("--lead0-nc", type=Path)
    p.add_argument("--lead0-var", default="ace2_hi_freq")
    p.add_argument("--lead0-label", default="ACE2 lead 0")
    p.add_argument("--lead1-nc", type=Path)
    p.add_argument("--lead1-var", default="ace2_hi_freq")
    p.add_argument("--lead1-label", default="ACE2")
    p.add_argument("--historical-nc", type=Path)
    p.add_argument("--historical-var", default="ace2_hi_freq")
    p.add_argument("--historical-label", default="ACE2 historical")
    p.add_argument(
        "--control-evaluation-root",
        type=Path,
        help="root containing TAG/TAG_YEAR.nc compact causal evaluations",
    )
    p.add_argument("--control-tag", default="evolving_may_aug_global")
    p.add_argument("--control-var", default="low_hhe_frequency")
    p.add_argument("--control-label", default="Climatological SST control")
    p.add_argument(
        "--experiment-evaluation-root",
        type=Path,
        help="root containing TAG/TAG_YEAR.nc for an additional completed experiment",
    )
    p.add_argument("--experiment-tag", default="evolving_may_aug_global")
    p.add_argument("--experiment-var", default="high_hhe_frequency")
    p.add_argument("--experiment-label", default="Evolving composite SST")
    p.add_argument("--amip-label", default="AMIP")
    p.add_argument("--amip-tna-label", default="AMIP-TNA")
    p.add_argument(
        "--omit-amip-global",
        action="store_true",
        help="omit the global-AMIP line while retaining AMIP-TNA",
    )
    p.add_argument(
        "--omit-amip-tna",
        action="store_true",
        help="plot ERA5/ACE2/AMIP without the AMIP-TNA series",
    )
    p.add_argument("--index-box", nargs=4, type=float, default=[23, 38, 260, 283])
    p.add_argument("--region-label", default="SEUS")
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument(
        "--common-years",
        action="store_true",
        help="plot only years present in every supplied series",
    )
    p.add_argument(
        "--require-four-distinct",
        action="store_true",
        help=(
            "require ERA5, ACE2, climatological control, and either the supplied "
            "experiment or AMIP-TNA, with at least two common years"
        ),
    )
    p.add_argument(
        "--figure-title",
        default=None,
        help="optional overall title; defaults to the legacy AMIP-style title",
    )
    p.add_argument(
        "--reference-style",
        action="store_true",
        help="match the supplied two-panel SPEAR time-series figure styling",
    )
    p.add_argument(
        "--mean-adjust-causal-to-ace2",
        action="store_true",
        help=(
            "apply one shared constant offset to control/experiment so the control "
            "mean matches ACE2; anomalies, their contrast, and correlations are unchanged"
        ),
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    land_lat, land_lon, land = load_land(args.land_mask_forcing)
    era5_years, era5 = regional_series(
        args.era5_nc, args.era5_var, args.index_box, land_lat, land_lon, land
    )
    series: list[dict] = [
        {"key": "era5", "label": "ERA5", "years": era5_years, "values": era5,
         "stat_label": "ERA5", "color": "#111111", "linewidth": 2.8},
    ]

    optional = [
        (args.lead0_nc, args.lead0_var, "lead0", args.lead0_label, args.lead0_label, "#e31a1c"),
        (args.lead1_nc, args.lead1_var, "lead1", args.lead1_label, args.lead1_label, "#9e1b32"),
        (args.historical_nc, args.historical_var, "historical", args.historical_label, args.historical_label, "#2455d6"),
    ]
    for path, variable, key, label, stat_label, color in optional:
        if path is None:
            continue
        years, values = regional_series(path, variable, args.index_box, land_lat, land_lon, land)
        series.append({"key": key, "label": label, "stat_label": stat_label,
                       "years": years, "values": values,
                       "color": color, "linewidth": 2.2})

    if args.control_evaluation_root is not None:
        years, values = tagged_evaluation_series(
            args.control_evaluation_root, args.control_tag, args.control_var,
            args.index_box, land_lat, land_lon, land,
        )
        if len(years):
            series.append({"key": "control", "label": args.control_label,
                           "stat_label": args.control_label, "years": years, "values": values,
                           "color": "#2455d6", "linewidth": 2.2})

    if args.experiment_evaluation_root is not None:
        years, values = tagged_evaluation_series(
            args.experiment_evaluation_root, args.experiment_tag, args.experiment_var,
            args.index_box, land_lat, land_lon, land,
        )
        if len(years):
            series.append({"key": "experiment", "label": args.experiment_label,
                           "stat_label": args.experiment_label, "years": years, "values": values,
                           "color": "#19a64a", "linewidth": 2.2})

    if not args.omit_amip_global:
        if args.amip_nc is not None:
            years, values = regional_series(
                args.amip_nc, args.amip_var, args.index_box, land_lat, land_lon, land
            )
            series.append({"key": "amip", "label": args.amip_label,
                           "stat_label": args.amip_label, "years": years, "values": values,
                           "color": "#19a64a", "linewidth": 2.2})
        elif args.amip_evaluation_dir is not None:
            years, values = amip_series(
                args.amip_evaluation_dir, "high_hhe_frequency", args.index_box,
                land_lat, land_lon, land
            )
            if len(years):
                series.append({"key": "amip", "label": args.amip_label,
                               "stat_label": args.amip_label, "years": years, "values": values,
                               "color": "#19a64a", "linewidth": 2.2})
        else:
            raise ValueError("supply either --amip-nc or --amip-evaluation-dir")

    if not args.omit_amip_tna:
        if args.amip_evaluation_dir is None:
            raise ValueError("AMIP-TNA requires --amip-evaluation-dir")
        years, values = amip_series(
            args.amip_evaluation_dir, "low_hhe_frequency", args.index_box,
            land_lat, land_lon, land
        )
        if len(years):
            series.append({"key": "amip_tna", "label": args.amip_tna_label,
                           "stat_label": args.amip_tna_label, "years": years, "values": values,
                           "color": "#d22bd2", "linewidth": 2.2})

    if len(series) == 1:
        raise RuntimeError("no model series found; AMIP evaluations and optional model files are absent")

    if args.require_four_distinct:
        required = {"era5", "lead1", "control"}
        missing = required.difference(item["key"] for item in series)
        fourth = {"experiment", "amip_tna"}.intersection(item["key"] for item in series)
        if not fourth:
            missing.add("experiment-or-amip_tna")
        if missing:
            raise RuntimeError(f"four-line plot is missing series: {sorted(missing)}")
        if len(series) != 4:
            raise RuntimeError(
                "--require-four-distinct expects exactly four plotted series; "
                f"found {[item['key'] for item in series]}"
            )

    start = args.start_year if args.start_year is not None else min(int(s["years"].min()) for s in series)
    end = args.end_year if args.end_year is not None else max(int(s["years"].max()) for s in series)
    for item in series:
        keep = (item["years"] >= start) & (item["years"] <= end)
        item["years"] = item["years"][keep]
        item["values"] = item["values"][keep]

    if args.common_years:
        common = set(int(year) for year in series[0]["years"])
        for item in series[1:]:
            common.intersection_update(int(year) for year in item["years"])
        common_years = np.asarray(sorted(common), dtype=int)
        if args.require_four_distinct and len(common_years) < 2:
            coverage = {item["label"]: item["years"].tolist() for item in series}
            raise RuntimeError(
                "need at least two common years for the four-line time series; "
                f"found {common_years.tolist()}; coverage={coverage}"
            )
        for item in series:
            lookup = {int(year): value for year, value in zip(item["years"], item["values"])}
            item["years"] = common_years.copy()
            item["values"] = np.asarray([lookup[int(year)] for year in common_years])

    mean_adjustments: dict[str, float] = {}
    if args.mean_adjust_causal_to_ace2:
        try:
            reference = next(item for item in series if item["key"] == "lead1")
            control = next(item for item in series if item["key"] == "control")
        except StopIteration as exc:
            raise RuntimeError("mean adjustment requires ACE2/lead1 and control series") from exc
        reference_mean = float(np.nanmean(reference["values"]))
        # Use one shared bias offset, diagnosed from the control.  Applying the
        # identical constant to both causal arms preserves their exact contrast.
        shared_offset = reference_mean - float(np.nanmean(control["values"]))
        for item in series:
            if item["key"] not in {"control", "experiment"}:
                continue
            item["values"] = item["values"] + shared_offset
            mean_adjustments[item["key"]] = shared_offset

    for item in series:
        item["detrended"] = detrend(item["years"], item["values"])

    if args.reference_style:
        reference_colors = {
            "era5": "#111111",
            "lead1": "#ef1010",
            "control": "#0b27df",
            "experiment": "#08e628",
            "amip_tna": "#ed12ed",
        }
        for item in series:
            item["color"] = reference_colors.get(item["key"], item["color"])
            item["linewidth"] = 2.7 if item["key"] == "era5" else 2.35

    era5_item = next(x for x in series if x["key"] == "era5")
    statistics = {"raw": {}, "detrended": {}}
    for item in series[1:]:
        r, pv, n = correlation(era5_item["years"], era5_item["values"], item["years"], item["values"])
        statistics["raw"][item["key"]] = {"r": r, "p": pv, "n": n}
        r, pv, n = correlation(
            era5_item["years"], era5_item["detrended"], item["years"], item["detrended"]
        )
        statistics["detrended"][item["key"]] = {"r": r, "p": pv, "n": n}

    figure_size = (13.5, 11.3) if args.reference_style else (13.5, 9.0)
    fig, axes = plt.subplots(
        2, 1, figsize=figure_size, sharex=True,
        constrained_layout=not args.reference_style,
    )
    if args.reference_style:
        fig.subplots_adjust(left=0.075, right=0.76, top=0.955, bottom=0.07, hspace=0.39)
        panels = [
            ("values", f"Time series of HHE in {args.region_label}", "%"),
            ("detrended", f"Time series of HHE in {args.region_label}, detrended", "%"),
        ]
    else:
        panels = [
            ("values", f"JJA HHE frequency over {args.region_label}", "HHE frequency (% of JJA days)"),
            ("detrended", f"Linearly detrended JJA HHE frequency over {args.region_label}",
             "HHE-frequency anomaly (percentage points)"),
        ]
    for panel_index, (field, title, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        for item in series:
            marker = "o" if len(item["years"]) == 1 else None
            ax.plot(item["years"], item[field], color=item["color"], linewidth=item["linewidth"],
                    marker=marker, markersize=6, label=item["label"])
        ax.axhline(0, color="#65717c", linewidth=0.8, alpha=0.7) if panel_index else None
        ax.set_title(
            title,
            fontsize=17 if args.reference_style else 15,
            fontweight="bold",
            pad=8 if args.reference_style else 10,
        )
        ax.set_ylabel(ylabel, fontsize=15 if args.reference_style else None)
        if args.reference_style:
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#303030")
                spine.set_linewidth(0.85)
            ax.tick_params(
                axis="both", which="major", direction="in", top=True, right=True,
                length=7, width=0.8, labelsize=13,
            )
            ax.text(
                0.035, 0.95, f"({chr(97 + panel_index)})", transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top",
            )
        else:
            ax.grid(axis="y", color="#d7dde3", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.015, 0.94, f"({chr(97 + panel_index)})", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="top")
        stat_key = "raw" if panel_index == 0 else "detrended"
        lines = []
        for item in series[1:]:
            record = statistics[stat_key][item["key"]]
            if np.isfinite(record["r"]):
                star = "*" if record["p"] < 0.05 else ""
                if args.reference_style:
                    lines.append((item["color"], f"R={record['r']:.2f}{star}"))
                else:
                    lines.append((item["color"], f"{item['stat_label']}: r={record['r']:+.2f}{star}  (n={record['n']})"))
            else:
                text = "R=NA" if args.reference_style else f"{item['stat_label']}: r=NA  (n={record['n']})"
                lines.append((item["color"], text))
        y = 0.90 if args.reference_style else 0.93
        for color, text in lines:
            ax.text(
                0.50 if args.reference_style else 0.99,
                y,
                text,
                transform=ax.transAxes,
                ha="left" if args.reference_style else "right",
                va="top",
                color=color,
                fontsize=14 if args.reference_style else 9.5,
                fontweight="bold",
            )
            y -= 0.065 if args.reference_style else 0.052

    if args.reference_style:
        raw = np.concatenate([
            item["values"][np.isfinite(item["values"])] for item in series
            if np.any(np.isfinite(item["values"]))
        ])
        raw_low = 5.0 * np.floor(float(raw.min()) / 5.0)
        raw_high = 5.0 * np.ceil(float(raw.max()) / 5.0)
        if raw_high - raw_low < 10.0:
            raw_high = raw_low + 10.0
        axes[0].set_ylim(raw_low, raw_high)
        axes[0].yaxis.set_major_locator(MultipleLocator(5.0))

        anomalies = np.concatenate([
            item["detrended"][np.isfinite(item["detrended"])] for item in series
            if np.any(np.isfinite(item["detrended"]))
        ])
        anomaly_low = 5.0 * np.floor(float(anomalies.min()) / 5.0)
        anomaly_high = 5.0 * np.ceil(float(anomalies.max()) / 5.0)
        if anomaly_high - anomaly_low < 10.0:
            anomaly_low -= 5.0
            anomaly_high += 5.0
        axes[1].set_ylim(anomaly_low, anomaly_high)
        axes[1].yaxis.set_major_locator(MultipleLocator(5.0))

        axes[0].legend(
            loc="upper left", bbox_to_anchor=(1.015, 0.985),
            frameon=True, fancybox=False, edgecolor="#303030",
            facecolor="white", framealpha=1.0, fontsize=11.5,
            handlelength=2.6, handletextpad=0.5, borderpad=0.4,
        )
        axes[1].legend(
            loc="upper left", bbox_to_anchor=(1.015, 0.985),
            frameon=True, fancybox=False, edgecolor="#303030",
            facecolor="white", framealpha=1.0, fontsize=11.5,
            handlelength=2.6, handletextpad=0.5, borderpad=0.4,
        )
    else:
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.06, 0.97), frameon=False, fontsize=9.5)

    axes[1].set_xlabel("Year", fontsize=15 if args.reference_style else None)
    axes[1].set_xlim(start - 0.5, end + 0.5)
    if args.reference_style:
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axes[0].tick_params(axis="x", labelbottom=True)
    if not args.reference_style:
        overall_title = args.figure_title or (
            f"ACE2 AMIP-style HHE time-series comparison | HI ≥ 105°F | box={args.index_box}"
        )
        fig.suptitle(overall_title, fontsize=13, color="#334155")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    statistics_json = {
        panel: {
            key: {
                "r": None if not np.isfinite(record["r"]) else float(record["r"]),
                "p": None if not np.isfinite(record["p"]) else float(record["p"]),
                "n": int(record["n"]),
            }
            for key, record in records.items()
        }
        for panel, records in statistics.items()
    }
    summary = {
        "region_label": args.region_label,
        "index_box": args.index_box,
        "years_shown": [start, end],
        "series": {item["key"]: item["years"].tolist() for item in series},
        "correlations_with_era5": statistics_json,
        "mean_adjustments_percentage_points": mean_adjustments,
        "mean_adjustment_note": (
            "one shared constant offset aligns the control to the ACE2 May-hindcast "
            "common-period mean; anomalies, trends, experiment-minus-control differences, "
            "and correlations unchanged"
            if mean_adjustments else None
        ),
        "asterisk": "two-sided Pearson correlation p < 0.05",
        "output": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
