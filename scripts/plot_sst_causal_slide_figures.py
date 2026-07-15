#!/usr/bin/env python3
"""Create clean, slide-ready figures for the ACE2 SST attribution study.

This script reads the compact NetCDF products written by
``evaluate_sst_composite_causal.py``.  It does not crop the original four-panel
diagnostic PNGs.  Every map and member plot is redrawn directly from the data.

Expected evaluation files (for BASE_YEAR=2000)
------------------------------------------------
    persistent_mar_global_2000.nc
    persistent_apr_global_2000.nc
    persistent_may_global_2000.nc
    evolving_may_aug_global_2000.nc
    evolving_may_aug_tropical_pacific_2000.nc
    evolving_may_aug_north_pacific_2000.nc
    evolving_may_aug_tropical_atlantic_2000.nc
    evolving_may_aug_north_atlantic_2000.nc

The output directory contains eight presentation assets plus a CSV/JSON table
of the plotted regional statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


MONTHS = ("mar", "apr", "may", "jun", "jul", "aug")
MONTH_LABELS = {
    "mar": "March (3-month lead)",
    "apr": "April (2-month lead)",
    "may": "May (1-month lead)",
    "jun": "June",
    "jul": "July",
    "aug": "August",
}

GLOBAL_SPECS = (
    ("persistent_mar_global", "Persistent March", "03_persistent_march_response"),
    ("persistent_apr_global", "Persistent April", "04_persistent_april_response"),
    ("persistent_may_global", "Persistent May", "05_persistent_may_response"),
    ("evolving_may_aug_global", "Evolving May–August", "06_evolving_may_august_response"),
)

BASIN_SPECS = (
    ("evolving_may_aug_tropical_pacific", "Tropical Pacific"),
    ("evolving_may_aug_north_pacific", "North Pacific"),
    ("evolving_may_aug_tropical_atlantic", "Tropical Atlantic"),
    ("evolving_may_aug_north_atlantic", "North Atlantic"),
)

NAVY = "#172936"
MUTED = "#66727D"
GRID = "#D8DDE1"
RED = "#BE3E4A"
BLUE = "#3479B9"
PURPLE = "#6D43A6"
GREEN = "#008A5B"
GRAY = "#8B969E"
LIGHT_RED = "#F5E7E9"
MAP_EXTENT = (-125.0, -66.0, 22.0, 50.5)
TARGET_BOX = (23.0, 38.0, 260.0, 283.0)


@dataclass
class Experiment:
    key: str
    label: str
    path: Path
    lat: np.ndarray
    lon: np.ndarray
    response_pp: np.ndarray
    member_ids: np.ndarray
    member_effects_pp: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.member_effects_pp))

    @property
    def median(self) -> float:
        return float(np.median(self.member_effects_pp))

    @property
    def n_positive(self) -> int:
        return int(np.count_nonzero(self.member_effects_pp > 0.0))

    @property
    def n_members(self) -> int:
        return int(self.member_effects_pp.size)


_COAST_GEOMETRIES = None
_COAST_WARNING_SHOWN = False


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_experiment(path: Path, key: str, label: str) -> Experiment:
    require_file(path, "evaluation NetCDF")
    with xr.open_dataset(path) as ds:
        required = {"high_minus_low_hhe_frequency", "regional_effect_by_case"}
        missing = required.difference(ds.data_vars)
        if missing:
            raise KeyError(f"{path} is missing variables: {sorted(missing)}")
        lat = np.asarray(ds["lat"], dtype=float)
        lon = np.asarray(ds["lon"], dtype=float) % 360.0
        response = 100.0 * np.asarray(ds["high_minus_low_hhe_frequency"], dtype=float)
        effects = 100.0 * np.asarray(ds["regional_effect_by_case"], dtype=float)
        if "case_member" in ds:
            member_ids = np.asarray(ds["case_member"], dtype=int)
        else:
            member_ids = np.arange(effects.size, dtype=int)

    valid = np.isfinite(effects)
    effects, member_ids = effects[valid], member_ids[valid]
    if effects.size == 0:
        raise ValueError(f"No finite paired-member effects in {path}")
    order = np.argsort(member_ids)
    return Experiment(
        key=key,
        label=label,
        path=path,
        lat=lat,
        lon=lon,
        response_pp=response,
        member_ids=member_ids[order],
        member_effects_pp=effects[order],
    )


def _coord_values(ds: xr.Dataset, candidates: tuple[str, ...]) -> np.ndarray | None:
    for name in candidates:
        if name in ds:
            return np.asarray(ds[name], dtype=float)
    return None


def load_land_mask(path: Path | None, lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
    """Load and nearest-align the forcing land mask to an evaluation grid."""
    if path is None or not path.is_file():
        print(
            "WARNING: no land-mask forcing file found; response maps will include all grid cells.",
            file=sys.stderr,
        )
        return None

    with xr.open_dataset(path) as ds:
        if "land_fraction" not in ds:
            raise KeyError(f"land_fraction is not present in {path}")
        da = ds["land_fraction"]
        if "time" in da.dims:
            da = da.isel(time=0)
        values = np.asarray(da, dtype=float).squeeze()
        source_lat = _coord_values(ds, ("latitude", "lat"))
        source_lon = _coord_values(ds, ("longitude", "lon"))

    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D land_fraction field in {path}; got {values.shape}")
    if source_lat is None or source_lon is None:
        if values.shape != (lat.size, lon.size):
            raise ValueError(
                f"Land mask {values.shape} cannot be aligned to evaluation grid "
                f"{(lat.size, lon.size)} without coordinates"
            )
        return values > 0.5

    source_lon = source_lon % 360.0
    lat_index = np.array([int(np.argmin(np.abs(source_lat - x))) for x in lat])
    lon_index = np.array(
        [int(np.argmin(np.minimum(np.abs(source_lon - x), 360.0 - np.abs(source_lon - x)))) for x in lon]
    )
    lat_error = float(np.max(np.abs(source_lat[lat_index] - lat)))
    raw_lon_error = np.abs(source_lon[lon_index] - lon)
    lon_error = float(np.max(np.minimum(raw_lon_error, 360.0 - raw_lon_error)))
    if lat_error > 0.1 or lon_error > 0.1:
        raise ValueError(
            f"Land mask coordinates do not match evaluation coordinates "
            f"(max errors: lat={lat_error:g}, lon={lon_error:g})"
        )
    return np.take(np.take(values, lat_index, axis=0), lon_index, axis=1) > 0.5


def coast_geometries():
    global _COAST_GEOMETRIES, _COAST_WARNING_SHOWN
    if _COAST_GEOMETRIES is not None:
        return _COAST_GEOMETRIES
    try:
        import cartopy.io.shapereader as shpreader

        shp = shpreader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
        _COAST_GEOMETRIES = list(shpreader.Reader(shp).geometries())
    except Exception as exc:  # plotting should still finish without cartopy data
        _COAST_GEOMETRIES = []
        if not _COAST_WARNING_SHOWN:
            print(f"WARNING: coastlines unavailable ({exc})", file=sys.stderr)
            _COAST_WARNING_SHOWN = True
    return _COAST_GEOMETRIES


def draw_coastlines(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    geometries = coast_geometries()
    if not geometries:
        return
    try:
        from shapely.geometry import box
    except Exception:
        return
    viewport = box(xlim[0], ylim[0], xlim[1], ylim[1])

    def draw_geometry(geom) -> None:
        if geom is None or geom.is_empty:
            return
        if hasattr(geom, "geoms"):
            for child in geom.geoms:
                draw_geometry(child)
        elif hasattr(geom, "exterior"):
            x, y = geom.exterior.xy
            ax.plot(x, y, color="#4A535A", linewidth=0.55, zorder=6)
            for ring in geom.interiors:
                x, y = ring.xy
                ax.plot(x, y, color="#4A535A", linewidth=0.45, zorder=6)

    for geom in geometries:
        try:
            draw_geometry(geom.intersection(viewport))
        except Exception:
            continue


def lon180_and_order(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon180)
    return lon180[order], order


def response_cmap():
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F3F4")
    return cmap


def robust_limit(fields: Iterable[np.ndarray], percentile: float, floor: float) -> float:
    finite = []
    for field in fields:
        values = np.abs(np.asarray(field, dtype=float))
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return floor
    limit = max(float(np.percentile(np.concatenate(finite), percentile)), floor)
    step = 0.25 if limit < 3.0 else 0.5
    return float(math.ceil(limit / step) * step)


def response_fields_for_scale(
    experiments: Iterable[Experiment], land: np.ndarray | None
) -> Iterable[np.ndarray]:
    for experiment in experiments:
        if land is None:
            yield experiment.response_pp
        else:
            yield np.where(land, experiment.response_pp, np.nan)


def add_target_box(ax: plt.Axes) -> None:
    south, north, west, east = TARGET_BOX
    west180 = west - 360.0 if west > 180.0 else west
    east180 = east - 360.0 if east > 180.0 else east
    ax.add_patch(
        Rectangle(
            (west180, south),
            east180 - west180,
            north - south,
            fill=False,
            edgecolor=GREEN,
            linewidth=1.8,
            zorder=8,
        )
    )


def draw_response_map(
    ax: plt.Axes,
    experiment: Experiment,
    land: np.ndarray | None,
    vmax: float,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    lon180, order = lon180_and_order(experiment.lon)
    field = experiment.response_pp[:, order]
    if land is not None:
        field = np.where(land[:, order], field, np.nan)
    mesh = ax.pcolormesh(
        lon180,
        experiment.lat,
        field,
        shading="nearest",
        cmap=response_cmap(),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        rasterized=True,
    )
    xmin, xmax, ymin, ymax = MAP_EXTENT
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    draw_coastlines(ax, (xmin, xmax), (ymin, ymax))
    add_target_box(ax)
    ax.set_xticks([-120, -110, -100, -90, -80, -70])
    ax.set_yticks([25, 30, 35, 40, 45, 50])
    ax.tick_params(labelsize=9, length=3)
    ax.set_xlabel("Longitude" if show_xlabel else "")
    ax.set_ylabel("Latitude" if show_ylabel else "")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def draw_member_bars(ax: plt.Axes, experiment: Experiment) -> None:
    effects = experiment.member_effects_pp
    x = np.arange(effects.size)
    colors = np.where(effects >= 0.0, RED, BLUE)
    ax.bar(x, effects, color=colors, width=0.76, edgecolor="none", zorder=3)
    ax.axhline(0.0, color=NAVY, linewidth=0.9, zorder=4)
    ax.axhline(
        experiment.mean,
        color=PURPLE,
        linewidth=1.7,
        linestyle=(0, (4, 2)),
        label=f"Mean {experiment.mean:+.2f} pp",
        zorder=5,
    )
    ax.axhline(
        experiment.median,
        color=NAVY,
        linewidth=1.3,
        linestyle=(0, (1, 2)),
        label=f"Median {experiment.median:+.2f} pp",
        zorder=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"m{i:02d}" for i in experiment.member_ids], rotation=60, ha="right")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Regional HHE change (pp)")
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(
        0.01,
        0.98,
        f"{experiment.n_positive}/{experiment.n_members} members positive",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=MUTED,
    )


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"wrote {output}", flush=True)


def plot_composite(composite_path: Path, output: Path, dpi: int) -> None:
    require_file(composite_path, "ACE2 SST composite NetCDF")
    with xr.open_dataset(composite_path) as ds:
        lat = np.asarray(ds["lat"], dtype=float)
        lon = np.asarray(ds["lon"], dtype=float) % 360.0
        fields = {
            month: np.asarray(ds[f"sst_high_minus_low_{month}"], dtype=float)
            for month in MONTHS
        }

    lon180, order = lon180_and_order(lon)
    vmax = robust_limit(fields.values(), percentile=98.5, floor=0.25)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.35), constrained_layout=True)
    mesh = None
    for i, (ax, month) in enumerate(zip(axes.flat, MONTHS)):
        field = fields[month][:, order]
        mesh = ax.pcolormesh(
            lon180,
            lat,
            field,
            shading="nearest",
            cmap=response_cmap(),
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            rasterized=True,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        draw_coastlines(ax, (-180, 180), (-60, 75))
        ax.set_title(MONTH_LABELS[month], loc="left", color=NAVY, pad=5)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.tick_params(labelsize=8, length=2.5)
        if i // 3 == 1:
            ax.set_xlabel("Longitude")
        if i % 3 == 0:
            ax.set_ylabel("Latitude")
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert mesh is not None
    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.075,
        aspect=45,
    )
    cbar.set_label("ACE2 high-HHE minus low-HHE SST composite (K)")
    cbar.outline.set_visible(False)
    save_figure(fig, output, dpi)


def plot_global_summary(experiments: list[Experiment], output: Path, dpi: int) -> None:
    means = np.array([exp.mean for exp in experiments])
    medians = np.array([exp.median for exp in experiments])
    labels = ["Persistent\nMarch", "Persistent\nApril", "Persistent\nMay", "Evolving\nMay–August"]
    x = np.arange(len(experiments))
    colors = [GRAY, GRAY, GRAY, RED]

    fig, ax = plt.subplots(figsize=(10.6, 5.25), constrained_layout=True)
    bars = ax.bar(x, means, color=colors, width=0.62, edgecolor="none", zorder=3)
    ax.scatter(
        x,
        medians,
        marker="D",
        s=58,
        color=NAVY,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="Member median",
    )
    ax.axhline(0.0, color=NAVY, linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Regional perturbation minus control (pp)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_span = max(float(np.max(np.abs(means))), 0.5)
    ax.set_ylim(min(-0.12, float(np.min(medians)) - 0.15), y_span * 1.34)
    for bar, exp in zip(bars, experiments):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.045,
            f"{exp.mean:+.3f} pp",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.03,
            f"{exp.n_positive}/{exp.n_members} +",
            ha="center",
            va="bottom",
            fontsize=9,
            color="white" if bar.get_height() > 0.22 else NAVY,
            fontweight="semibold",
            zorder=6,
        )
    ax.text(
        0.99,
        0.98,
        "Bars: paired-member mean   ◆: paired-member median",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.5,
    )
    save_figure(fig, output, dpi)


def plot_response_pair(
    experiment: Experiment,
    land: np.ndarray | None,
    map_vmax: float,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.15),
        gridspec_kw={"width_ratios": (1.05, 1.0)},
        constrained_layout=True,
    )
    mesh = draw_response_map(axes[0], experiment, land, map_vmax)
    axes[0].set_title("Spatial HHE response", loc="left")
    cbar = fig.colorbar(mesh, ax=axes[0], orientation="horizontal", pad=0.10, fraction=0.055)
    cbar.set_label("Perturbation minus control (percentage points)")
    cbar.outline.set_visible(False)

    draw_member_bars(axes[1], experiment)
    axes[1].set_title("Paired lag-member effects", loc="left")
    save_figure(fig, output, dpi)


def plot_basin_maps(
    experiments: list[Experiment],
    land: np.ndarray | None,
    output: Path,
    dpi: int,
) -> None:
    vmax = robust_limit(
        response_fields_for_scale(experiments, land), percentile=99.5, floor=0.25
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.65), constrained_layout=True)
    mesh = None
    for i, (ax, exp) in enumerate(zip(axes.flat, experiments)):
        mesh = draw_response_map(
            ax,
            exp,
            land,
            vmax,
            show_xlabel=(i // 2 == 1),
            show_ylabel=(i % 2 == 0),
        )
        ax.set_title(
            f"{exp.label}   {exp.mean:+.3f} pp mean",
            loc="left",
            fontsize=12,
        )
        ax.text(
            0.99,
            0.98,
            f"{exp.n_positive}/{exp.n_members} positive",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )
    assert mesh is not None
    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.065,
        aspect=45,
    )
    cbar.set_label("Regional HHE-frequency response (percentage points)")
    cbar.outline.set_visible(False)
    save_figure(fig, output, dpi)


def plot_basin_summary(
    global_experiment: Experiment,
    basin_experiments: list[Experiment],
    output: Path,
    dpi: int,
) -> None:
    experiments = [global_experiment, *basin_experiments]
    labels = ["Global evolving", *(exp.label for exp in basin_experiments)]
    means = np.array([exp.mean for exp in experiments])
    medians = np.array([exp.median for exp in experiments])
    y = np.arange(len(experiments))
    colors = [RED] + ["#C8755D" if value > 0 else BLUE for value in means[1:]]

    fig, ax = plt.subplots(figsize=(10.9, 5.35), constrained_layout=True)
    bars = ax.barh(y, means, color=colors, height=0.58, edgecolor="none", zorder=3)
    ax.scatter(
        medians,
        y,
        marker="D",
        s=52,
        color=NAVY,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Member median",
    )
    ax.axvline(0.0, color=NAVY, linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Regional perturbation minus control (percentage points)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    left = min(-0.45, float(np.min(np.r_[means, medians])) - 0.20)
    # Leave a separate right-hand text column for member agreement so it never
    # collides with the global bar-end value.
    right = max(1.65, float(np.max(np.r_[means, medians])) + 0.60)
    ax.set_xlim(left, right)
    for bar, exp in zip(bars, experiments):
        value = exp.mean
        offset = 0.035 if value >= 0 else -0.035
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f} pp",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
        )
        ax.text(
            right - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{exp.n_positive}/{exp.n_members} positive",
            ha="right",
            va="center",
            fontsize=9.5,
            color=MUTED,
        )
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.0,
        -0.12,
        "Each basin experiment perturbs only the named basin; basin effects are not expected to sum linearly.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=9.5,
    )
    save_figure(fig, output, dpi)


def write_statistics(
    output_dir: Path,
    global_experiments: list[Experiment],
    basin_experiments: list[Experiment],
) -> None:
    experiments = [*global_experiments, *basin_experiments]
    rows = [
        {
            "key": exp.key,
            "label": exp.label,
            "mean_percentage_points": exp.mean,
            "median_percentage_points": exp.median,
            "positive_members": exp.n_positive,
            "n_members": exp.n_members,
            "equivalent_jja_days_from_mean": exp.mean * 0.92,
            "source": str(exp.path),
        }
        for exp in experiments
    ]
    json_path = output_dir / "figure_statistics.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    csv_path = output_dir / "figure_statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("outputs/lag_may/sst_causal_v2"),
        help="SST experiment root (default: outputs/lag_may/sst_causal_v2)",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help="Directory containing compact evaluation .nc files; default EXP_ROOT/evaluation",
    )
    parser.add_argument(
        "--composite",
        type=Path,
        default=None,
        help="Composite NetCDF; default EXP_ROOT/ace2_hhe_sst_composites_mar_aug.nc",
    )
    parser.add_argument(
        "--land-mask-forcing",
        type=Path,
        default=None,
        help="forcing_YEAR.nc containing land_fraction; default EXP_ROOT/inputs/control/forcing/forcing_YEAR.nc",
    )
    parser.add_argument("--base-year", type=int, default=2000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; default EXP_ROOT/slide_figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    exp_root = args.exp_root.resolve()
    evaluation_dir = (args.evaluation_dir or exp_root / "evaluation").resolve()
    composite = (args.composite or exp_root / "ace2_hhe_sst_composites_mar_aug.nc").resolve()
    land_path = (
        args.land_mask_forcing
        or exp_root / "inputs" / "control" / "forcing" / f"forcing_{args.base_year}.nc"
    ).resolve()
    output_dir = (args.out_dir or exp_root / "slide_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_experiments = [
        load_experiment(
            evaluation_dir / f"{key}_{args.base_year}.nc",
            key,
            label,
        )
        for key, label, _ in GLOBAL_SPECS
    ]
    basin_experiments = [
        load_experiment(
            evaluation_dir / f"{key}_{args.base_year}.nc",
            key,
            label,
        )
        for key, label in BASIN_SPECS
    ]

    reference = global_experiments[0]
    for exp in [*global_experiments[1:], *basin_experiments]:
        if exp.response_pp.shape != reference.response_pp.shape:
            raise ValueError(f"Grid shape mismatch: {reference.path} versus {exp.path}")
        if not np.allclose(exp.lat, reference.lat) or not np.allclose(exp.lon, reference.lon):
            raise ValueError(f"Grid coordinate mismatch: {reference.path} versus {exp.path}")
    land = load_land_mask(land_path, reference.lat, reference.lon)

    print(f"evaluation directory: {evaluation_dir}")
    print(f"composite: {composite}")
    print(f"land mask: {land_path if land is not None else 'not used'}")
    print(f"output directory: {output_dir}")

    plot_composite(composite, output_dir / "01_sst_composite_march_august.png", args.dpi)
    plot_global_summary(
        global_experiments,
        output_dir / "02_global_experiment_summary.png",
        args.dpi,
    )

    global_map_vmax = robust_limit(
        response_fields_for_scale(global_experiments, land),
        percentile=99.5,
        floor=0.25,
    )
    for experiment, (_, _, output_stem) in zip(global_experiments, GLOBAL_SPECS):
        plot_response_pair(
            experiment,
            land,
            global_map_vmax,
            output_dir / f"{output_stem}.png",
            args.dpi,
        )

    plot_basin_maps(
        basin_experiments,
        land,
        output_dir / "07_individual_basin_response_maps.png",
        args.dpi,
    )
    plot_basin_summary(
        global_experiments[-1],
        basin_experiments,
        output_dir / "08_individual_basin_effect_summary.png",
        args.dpi,
    )
    write_statistics(output_dir, global_experiments, basin_experiments)
    print("All slide figures completed successfully.", flush=True)


if __name__ == "__main__":
    main()
