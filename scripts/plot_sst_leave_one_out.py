#!/usr/bin/env python3
"""Plot compact multi-year leave-one-basin-out SST attribution results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr


BASINS = {
    "tropical_pacific": "Tropical Pacific",
    "north_pacific": "North Pacific",
    "tropical_atlantic": "Tropical Atlantic",
    "north_atlantic": "North Atlantic",
}
MAP_EXTENT = (-130.0, -60.0, 20.0, 52.0)
INDEX_BOX = (23.0, 38.0, 260.0, 283.0)
_COASTS = None


# Absolute HHE frequency now begins at pure white rather than YlOrRd's pale
# yellow.  The incremental-effect palette also has an explicit white midpoint,
# so cells with no change are visually neutral in every map.
FREQUENCY_CMAP = LinearSegmentedColormap.from_list(
    "hhe_frequency_white_to_red",
    [
        (0.00, "#ffffff"),
        (0.12, "#fff4d6"),
        (0.35, "#fdd17a"),
        (0.58, "#f98e45"),
        (0.78, "#ed3b2f"),
        (1.00, "#87002a"),
    ],
)
EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "hhe_effect_blue_white_red",
    [
        (0.00, "#2166ac"),
        (0.30, "#92c5de"),
        (0.50, "#ffffff"),
        (0.70, "#f4a582"),
        (1.00, "#b2182b"),
    ],
)


def lon180_order(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    converted = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(converted)
    return converted[order], order


def coastlines(ax: plt.Axes) -> None:
    global _COASTS
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box

        if _COASTS is None:
            path = shpreader.natural_earth(
                resolution="110m", category="physical", name="land"
            )
            _COASTS = list(shpreader.Reader(path).geometries())
        xmin, xmax, ymin, ymax = MAP_EXTENT
        viewport = box(xmin, ymin, xmax, ymax)

        def draw(geometry) -> None:
            if geometry is None or geometry.is_empty:
                return
            if hasattr(geometry, "geoms"):
                for part in geometry.geoms:
                    draw(part)
            elif hasattr(geometry, "exterior"):
                x, y = geometry.exterior.xy
                ax.plot(x, y, color="#4b5563", linewidth=0.55, zorder=5)

        for geometry in _COASTS:
            draw(geometry.intersection(viewport))
    except Exception as exc:
        if _COASTS is None:
            print(f"WARNING: coastlines unavailable ({exc})", file=sys.stderr)
            _COASTS = []


def decorate_map(ax: plt.Axes) -> None:
    xmin, xmax, ymin, ymax = MAP_EXTENT
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(color="#d1d5db", linewidth=0.35, alpha=0.55)
    coastlines(ax)
    south, north, west, east = INDEX_BOX
    west = west - 360.0 if west > 180.0 else west
    east = east - 360.0 if east > 180.0 else east
    ax.add_patch(
        Rectangle(
            (west, south), east - west, north - south,
            fill=False, edgecolor="#009e73", linewidth=1.8, zorder=7,
        )
    )


def land_mask(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        field = dataset["land_fraction"]
        if "time" in field.dims:
            field = field.isel(time=0)
        source_lat = np.asarray(dataset["latitude"], dtype=float)
        source_lon = np.asarray(dataset["longitude"], dtype=float) % 360.0
        values = np.asarray(field, dtype=float)
    lat_index = np.array([int(np.argmin(np.abs(source_lat - x))) for x in lat])
    lon_index = np.array([
        int(np.argmin(np.minimum(np.abs(source_lon - x), 360.0 - np.abs(source_lon - x))))
        for x in lon
    ])
    return np.take(np.take(values, lat_index, axis=0), lon_index, axis=1) > 0.5


def nice_limit(values: list[np.ndarray], percentile: float, floor: float) -> float:
    pieces = [
        np.abs(value[np.isfinite(value)]) for value in values
        if np.any(np.isfinite(value))
    ]
    if not pieces:
        return floor
    finite = np.concatenate(pieces)
    raw = max(float(np.percentile(finite, percentile)), floor)
    step = 0.25 if raw < 2.0 else 0.5
    return math.ceil(raw / step) * step


def panel_figure(
    dataset: xr.Dataset,
    basin_label: str,
    mask: np.ndarray,
    output: Path,
) -> None:
    lat = np.asarray(dataset["lat"], dtype=float)
    lon = np.asarray(dataset["lon"], dtype=float) % 360.0
    x, order = lon180_order(lon)
    global_frequency = 100.0 * np.asarray(dataset["high_hhe_frequency"], dtype=float)
    without_frequency = 100.0 * np.asarray(dataset["low_hhe_frequency"], dtype=float)
    effect = 100.0 * np.asarray(dataset["high_minus_low_hhe_frequency"], dtype=float)
    global_frequency = np.where(mask, global_frequency, np.nan)[:, order]
    without_frequency = np.where(mask, without_frequency, np.nan)[:, order]
    effect = np.where(mask, effect, np.nan)[:, order]
    absolute_max = nice_limit([global_frequency, without_frequency], 99.0, 5.0)
    effect_max = nice_limit([effect], 98.0, 0.5)

    years = np.asarray(dataset["year"], dtype=int)
    annual = 100.0 * np.asarray(dataset["regional_effect_by_year"], dtype=float)
    mean = 100.0 * float(dataset["regional_effect_mean"])
    lower = 100.0 * float(dataset["regional_effect_ci_95_lower"])
    upper = 100.0 * float(dataset["regional_effect_ci_95_upper"])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    cmap_frequency = FREQUENCY_CMAP.copy()
    cmap_frequency.set_bad("#f3f4f6")
    cmap_effect = EFFECT_CMAP.copy()
    cmap_effect.set_bad("#f3f4f6")

    image = axes[0, 0].pcolormesh(
        x, lat, global_frequency, shading="nearest", cmap=cmap_frequency,
        vmin=0.0, vmax=absolute_max,
    )
    decorate_map(axes[0, 0])
    axes[0, 0].set_title("a  Global evolving SST anomaly", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 0], label="HHE frequency (% of JJA days)")

    image = axes[0, 1].pcolormesh(
        x, lat, without_frequency, shading="nearest", cmap=cmap_frequency,
        vmin=0.0, vmax=absolute_max,
    )
    decorate_map(axes[0, 1])
    axes[0, 1].set_title(
        f"b  Global evolving except {basin_label}", loc="left", fontweight="bold"
    )
    fig.colorbar(image, ax=axes[0, 1], label="HHE frequency (% of JJA days)")

    image = axes[1, 0].pcolormesh(
        x, lat, effect, shading="nearest", cmap=cmap_effect,
        vmin=-effect_max, vmax=effect_max,
    )
    decorate_map(axes[1, 0])
    axes[1, 0].set_title(
        f"c  Incremental effect of {basin_label}", loc="left", fontweight="bold"
    )
    fig.colorbar(image, ax=axes[1, 0], label="HHE-frequency change (percentage points)")

    ax = axes[1, 1]
    colors = np.where(annual >= 0.0, "#c83e3a", "#347bb8")
    ax.bar(years, annual, color=colors, width=0.72)
    ax.axhline(0.0, color="#374151", linewidth=0.9)
    ax.axhline(mean, color="#6f3bae", linestyle="--", linewidth=1.8)
    ax.fill_between(
        [years.min() - 0.5, years.max() + 0.5], lower, upper,
        color="#6f3bae", alpha=0.12,
    )
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("regional effect (percentage points)")
    ax.set_title("d  Regional effect by atmospheric year", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.5)
    ax.text(
        0.98, 0.97,
        f"mean = {mean:+.2f} pp\n95% year-bootstrap CI [{lower:+.2f}, {upper:+.2f}]",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    period = str(dataset.attrs.get("combined_period", f"{years.min()}-{years.max()}"))
    pairs = int(dataset.attrs.get("n_map_member_year_pairs", len(years) * 25))
    fig.suptitle(
        f"ACE2 leave-one-basin-out SST attribution: {basin_label}\n"
        f"global evolving minus global evolving except basin | {period} | {pairs} member-year cases",
        fontsize=14, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output}", flush=True)


def summary_figure(records: list[dict], output: Path) -> None:
    labels = [record["label"] for record in records]
    means = np.asarray([record["mean"] for record in records])
    lower = np.asarray([record["lower"] for record in records])
    upper = np.asarray([record["upper"] for record in records])
    positions = np.arange(len(records))
    colors = np.where(means >= 0.0, "#c83e3a", "#347bb8")

    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.21)
    ax.barh(positions, means, color=colors, height=0.56, zorder=3)
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack([
            np.maximum(means - lower, 0.0),
            np.maximum(upper - means, 0.0),
        ]),
        fmt="none", ecolor="#20242a", elinewidth=1.5, capsize=4,
        capthick=1.3, zorder=4,
    )
    ax.axvline(0.0, color="#374151", linewidth=0.9, zorder=2)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("regional HHE-frequency effect (percentage points)")
    ax.set_title(
        "Incremental contribution of each basin to the global evolving-SST response\n"
        "global evolving minus global evolving without the named basin",
        loc="left", fontweight="bold", fontsize=13.5, pad=12,
    )
    ax.grid(axis="x", color="#d1d5db", linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

    finite_bounds = np.r_[means, lower[np.isfinite(lower)], upper[np.isfinite(upper)], 0.0]
    data_min = float(np.min(finite_bounds))
    data_max = float(np.max(finite_bounds))
    span = max(data_max - data_min, 0.25)
    label_pad = 0.04 * span
    axis_pad = 0.20 * span
    ax.set_xlim(data_min - axis_pad, data_max + axis_pad)

    for position, value, ci_lower, ci_upper in zip(positions, means, lower, upper):
        if value >= 0.0:
            anchor = max(value, ci_upper if np.isfinite(ci_upper) else value) + label_pad
            alignment = "left"
        else:
            anchor = min(value, ci_lower if np.isfinite(ci_lower) else value) - label_pad
            alignment = "right"
        ax.text(
            anchor,
            position,
            f"{value:+.2f} pp",
            ha=alignment,
            va="center",
            fontsize=10.5,
            fontweight="semibold",
            clip_on=False,
        )
    fig.text(
        0.22, 0.055,
        "Error bars: 95% bootstrap CI across atmospheric base years. "
        "Positive values mean the named basin increases regional HHE frequency.",
        ha="left", va="bottom", fontsize=9.2, color="#4b5563",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, dpi=200, bbox_inches="tight", pad_inches=0.18, facecolor="white"
    )
    plt.close(fig)
    print(f"wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--period-tag", default="1995_2005")
    parser.add_argument("--land-mask-forcing", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for basin, label in BASINS.items():
        tag = f"evolving_may_aug_without_{basin}"
        path = args.input_dir / f"{tag}_{args.period_tag}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as source:
            dataset = source.load()
        mask = land_mask(
            args.land_mask_forcing,
            np.asarray(dataset["lat"], dtype=float),
            np.asarray(dataset["lon"], dtype=float) % 360.0,
        )
        panel_figure(dataset, label, mask, args.out_dir / f"{tag}_{args.period_tag}.png")
        records.append({
            "basin": basin,
            "label": label,
            "mean": 100.0 * float(dataset["regional_effect_mean"]),
            "lower": 100.0 * float(dataset["regional_effect_ci_95_lower"]),
            "upper": 100.0 * float(dataset["regional_effect_ci_95_upper"]),
        })

    summary_path = args.out_dir / f"leave_one_basin_out_summary_{args.period_tag}.png"
    summary_figure(records, summary_path)
    (args.out_dir / f"leave_one_basin_out_summary_{args.period_tag}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
