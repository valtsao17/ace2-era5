#!/usr/bin/env python3
"""Workflow 5: marked spatiotemporal extreme-event evaluation.

The default mode reads Workflow 3's cached daily ERA5/ACE2 Tmax fields, refits
fit-only frozen calendar-window thresholds, constructs relative-extreme and
absolute-hazard tracks, labels 3D events, matches events one-to-one within every
member, and scores event support, counts, arrivals, nulls, and tail dependence.

The former pixelwise temporal-neighborhood calculation is retained only as
``--workflow-mode legacy-coverage``. Its diagnostic is

    P_tau(x, t) = mean_m I[exists t' in t +/- tau where member m has an event at x].

That field is neighborhood coverage and is never used as the primary matched
event probability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/ace2_era5_mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/ace2_era5_xdg_cache"))
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EVENT_SEGMENTATION_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVENT_SEGMENTATION_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from run_daily_first_shape_sweep import (  # noqa: E402
    DEFAULT_OUT_ROOT as DEFAULT_DAILY_FIRST_ROOT,
    DailyCase,
    DailyFirstPaths,
    ace2_daily_path,
    all_daily_cases,
    barrier,
    era5_daily_path,
    event_slug,
    format_map_axis,
    lon_for_plot,
    mask_path,
    mpi_info,
    parse_neighborhood_pixels,
    threshold_path,
)
from run_daily_first_spacetime_tolerance import (  # noqa: E402
    MaskCube,
    binary_overlap_metrics,
    dilate_mask,
    finite_numeric,
    load_mask_cube,
    neighbor_indices,
    overlap_code,
    summarize_group,
)
from run_weekly_full_sweep import parse_years, rank_print, rank_subset  # noqa: E402
from weekly_event_metrics import cell_area_km2  # noqa: E402


DEFAULT_OUT_ROOT = DEFAULT_DAILY_FIRST_ROOT / "workflow5_temporal_support"
STAGES = ("metrics", "paper-metrics", "figures")
DEFAULT_STAGES = ("metrics", "figures")
MARKED_STAGES = (
    "all",
    "thresholds",
    "catalogs",
    "matching",
    "coverage",
    "process",
    "nulls",
    "paper-metrics",
    "figures",
)
FIGURE_NAMES = ("support-sweep", "timeseries", "spatial-timeseries", "effect-distributions", "sample-panels")
FIGURE_ALIASES = {"all": FIGURE_NAMES}
PROB_CMAP = LinearSegmentedColormap.from_list(
    "event_probability_white_red",
    ["#ffffff", "#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#67000d"],
)
OVERLAP_CMAP = ListedColormap(["#ffffff", "#2b83ba", "#fdae61", "#1a9850"])
OVERLAP_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], OVERLAP_CMAP.N)
OVERLAP_LABELS = ("none", "ERA5 only", "ACE2 member support only", "both")


@dataclass(frozen=True)
class SupportPaths:
    out_root: Path
    tables: Path
    figures: Path
    parts: Path
    logs: Path


def normalize_cli_tokens(tokens: list[str]) -> list[str]:
    return [token.strip().replace("\u00a0", "") for token in tokens]


def parse_int_list(values: list[str] | tuple[str, ...] | str, *, name: str) -> list[int]:
    raw_values = [values] if isinstance(values, str) else list(values)
    out: list[int] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
            out.append(value)
    return sorted(set(out or [0]))


def parse_float_list(values: list[str] | tuple[str, ...] | str, *, name: str) -> list[float]:
    raw_values = [values] if isinstance(values, str) else list(values)
    out: list[float] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
            out.append(value)
    return sorted(set(out or [0.5]))


def parse_unbounded_float_list(values: list[str] | tuple[str, ...] | str, *, name: str) -> list[float]:
    raw_values = [values] if isinstance(values, str) else list(values)
    out: list[float] = []
    for raw in raw_values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            out.append(float(token))
    if not out:
        raise ValueError(f"No {name} supplied")
    return sorted(set(out))


def slug_number(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def threshold_suffix(value: float) -> str:
    return f"p{slug_number(value)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Workflow 5: full ACE2-vs-ERA5 marked spatiotemporal event evaluation "
            "from existing Workflow 3 daily fields. The former pixelwise temporal "
            "coverage diagnostic remains available as --workflow-mode legacy-coverage."
        )
    )
    parser.add_argument(
        "--workflow-mode",
        choices=("marked-events", "legacy-coverage"),
        default="marked-events",
        help="Run the full marked-event workflow (default) or the former coverage-only diagnostic.",
    )
    parser.add_argument("--daily-first-root", type=Path, default=DEFAULT_DAILY_FIRST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--fit-years", default="1981:2000")
    parser.add_argument("--validation-years", default="2001:2010")
    parser.add_argument("--test-years", default="2011:2022")
    parser.add_argument(
        "--marked-stages",
        nargs="+",
        choices=MARKED_STAGES,
        default=["all"],
        help="Stages for the full marked-event workflow.",
    )
    parser.add_argument(
        "--marked-max-cases",
        type=int,
        default=None,
        help="Debug-only daily-case limit applied separately within every year/period.",
    )
    parser.add_argument(
        "--field-variant",
        choices=("smoothed", "unsmoothed"),
        default="smoothed",
        help="Use cached smoothed daily Tmax or run the required unsmoothed sensitivity.",
    )
    parser.add_argument(
        "--calendar-window-days",
        type=int,
        default=7,
        help="Half-width of the fit-only calendar-day threshold window.",
    )
    parser.add_argument("--comparison-operator", choices=(">", ">="), default=">")
    parser.add_argument(
        "--lag-group-size",
        type=int,
        default=0,
        help="Consecutive ACE2 members per threshold pool; zero pools all members.",
    )
    parser.add_argument(
        "--fit-gpd-pot",
        action="store_true",
        help="Optionally fit runs-declustered GPD/POT diagnostics to fit-period daily excesses.",
    )
    parser.add_argument("--gpd-run-length-days", type=int, default=3)
    parser.add_argument("--gpd-min-peaks", type=int, default=20)
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--min-duration-days", type=int, default=2)
    parser.add_argument("--min-event-area-km2", type=float, default=10_000.0)
    parser.add_argument(
        "--validation-connectivities",
        nargs="+",
        default=None,
        help="Candidate 3D connectivities selected using validation years only.",
    )
    parser.add_argument(
        "--validation-min-duration-days",
        nargs="+",
        default=None,
        help="Candidate minimum durations selected using validation years only.",
    )
    parser.add_argument(
        "--validation-min-event-area-km2",
        nargs="+",
        default=None,
        help="Candidate minimum daily areas selected using validation years only.",
    )
    parser.add_argument(
        "--frozen-region-map",
        type=Path,
        default=None,
        help="Optional validated ERA5-only region artifact. Unfrozen candidates are rejected.",
    )
    parser.add_argument(
        "--match-radii-km",
        nargs="+",
        default=["0", "250", "500", "1000"],
    )
    parser.add_argument(
        "--match-tolerances-days",
        nargs="+",
        default=["0", "1", "3", "7"],
    )
    parser.add_argument("--primary-match-radius-km", type=float, default=500.0)
    parser.add_argument("--primary-match-tolerance-days", type=int, default=3)
    parser.add_argument("--match-weight-timing", type=float, default=1.0)
    parser.add_argument("--match-weight-distance", type=float, default=1.0)
    parser.add_argument("--match-weight-iou", type=float, default=2.0)
    parser.add_argument("--match-weight-duration", type=float, default=0.5)
    parser.add_argument("--match-weight-area", type=float, default=0.5)
    parser.add_argument("--match-weight-intensity", type=float, default=0.5)
    parser.add_argument(
        "--process-windows",
        choices=("weekly", "monthly", "seasonal"),
        default="weekly",
        help=(
            "Fixed windows for count/arrival/reliability scoring. Weekly is the "
            "default because whole-domain monthly q90 occurrence is saturated."
        ),
    )
    parser.add_argument(
        "--process-region-mode",
        choices=("auto", "catalog", "grid"),
        default="auto",
        help=(
            "Spatial units for count/arrival scoring. Auto uses a supplied frozen "
            "catalog region map, otherwise predeclared regular geographic blocks."
        ),
    )
    parser.add_argument(
        "--process-grid-degrees",
        type=float,
        default=10.0,
        help="Latitude/longitude width of predeclared process blocks in grid mode.",
    )
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--null-replicates", type=int, default=50)
    parser.add_argument(
        "--independence-surrogate-replicates",
        type=int,
        default=0,
        help="Optional cellwise marginal-preserving independence null replicates.",
    )
    parser.add_argument("--minimum-null-shift-days", type=int, default=7)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--local-are-quantile", type=float, default=0.95)
    parser.add_argument("--force-marked", action="store_true")
    parser.add_argument("--years", default="1981:2022")
    parser.add_argument("--week-days", type=int, default=7)
    parser.add_argument("--include-partial-final-week", action="store_true")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--temporal-neighborhood-days",
        nargs="+",
        default=["0", "1", "2", "3"],
        help="Temporal windows tau, in days, for P_tau.",
    )
    parser.add_argument(
        "--spatial-radii-pixels",
        nargs="+",
        default=["0", "1", "2", "3", "5"],
        help=(
            "Optional spatial radii r. For r>0, each member mask is dilated before "
            "temporal union, yielding nearby support P_{r,tau}."
        ),
    )
    parser.add_argument(
        "--probability-thresholds",
        nargs="+",
        default=["0.25", "0.5", "0.75"],
        help="Thresholds applied to P_tau for binary IoU/F1 diagnostics.",
    )
    parser.add_argument(
        "--paper-sources",
        nargs="+",
        choices=("era5", "ace2"),
        default=["era5", "ace2"],
        help="Continuous-field sources used for ARE/chi dependence summaries.",
    )
    parser.add_argument(
        "--paper-quantiles",
        nargs="+",
        default=["0.9", "0.95", "0.98"],
        help="Upper-tail quantiles u for chi_ij(u) and ARE(u).",
    )
    parser.add_argument(
        "--twcrps-tail-quantile",
        type=float,
        default=0.9,
        help="Upper-tail cutoff quantile for tail-weighted CRPS.",
    )
    parser.add_argument(
        "--twcrps-tail-threshold-source",
        choices=("ace2_climatology", "case_ensemble"),
        default="ace2_climatology",
        help=(
            "Use Workflow 3 ACE2 climatological threshold maps when the tail quantile "
            "matches --percentile, or use each case's ACE2 ensemble quantile."
        ),
    )
    parser.add_argument(
        "--chi-distance-bins-km",
        nargs="+",
        default=["0", "250", "500", "1000", "2000", "4000"],
        help="Distance-bin edges in km for summaries of pairwise chi_ij(u).",
    )
    parser.add_argument(
        "--chi-max-pairs-per-bin",
        type=int,
        default=1000,
        help="Maximum randomly sampled grid-cell pairs per distance bin for chi summaries.",
    )
    parser.add_argument(
        "--are-max-reference-points",
        type=int,
        default=150,
        help="Maximum randomly sampled reference grid cells for ARE summaries.",
    )
    parser.add_argument(
        "--are-reference-stride",
        type=int,
        default=1,
        help="Optional stride over candidate reference cells before random ARE sampling.",
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument(
        "--figure-names",
        nargs="+",
        default=["all"],
        help="Figure subset. Use 'all' or any of: " + ", ".join(FIGURE_NAMES),
    )
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--force-paper-metrics", action="store_true")
    parser.add_argument("--force-figures", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Check that the configured Workflow 3 root contains every input "
            "needed by the selected stages, print the result, and exit."
        ),
    )
    parser.add_argument("--sample-date", default=None, help="Optional YYYY-MM-DD reference day for sample panels.")
    parser.add_argument("--sample-cases", type=int, default=1)
    parser.add_argument("--sample-members", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args(normalize_cli_tokens(sys.argv[1:]))


def selected_figure_names(args: argparse.Namespace) -> tuple[str, ...]:
    selected: list[str] = []
    for raw in args.figure_names:
        for token in str(raw).split(","):
            name = token.strip().lower().replace("_", "-")
            if not name:
                continue
            if name in FIGURE_ALIASES:
                for aliased in FIGURE_ALIASES[name]:
                    if aliased not in selected:
                        selected.append(aliased)
                continue
            if name not in FIGURE_NAMES:
                valid = ", ".join(("all", *FIGURE_NAMES))
                raise ValueError(f"Unknown figure name {token!r}. Valid values: {valid}")
            if name not in selected:
                selected.append(name)
    if not selected:
        return FIGURE_ALIASES["all"]
    return tuple(name for name in FIGURE_NAMES if name in selected)


def make_paths(out_root: Path) -> SupportPaths:
    paths = SupportPaths(
        out_root=out_root,
        tables=out_root / "tables",
        figures=out_root / "figures",
        parts=out_root / "tables" / "parts",
        logs=out_root / "logs",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def daily_first_input_paths(out_root: Path) -> DailyFirstPaths:
    """Describe an existing Workflow 3 tree without creating input directories."""
    return DailyFirstPaths(
        out_root=out_root,
        daily_era5=out_root / "daily_fields" / "era5",
        daily_ace2=out_root / "daily_fields" / "ace2",
        thresholds=out_root / "thresholds",
        masks=out_root / "masks",
        tables=out_root / "tables",
        figures=out_root / "figures",
        logs=out_root / "logs",
    )


def required_daily_first_artifacts(
    daily_cases: list[DailyCase],
    daily_first_paths,
    args: argparse.Namespace,
) -> dict[str, list[Path]]:
    """Return the Workflow 3 artifacts required by the selected Workflow 5 stages."""
    required: dict[str, list[Path]] = {}
    figure_names = selected_figure_names(args) if "figures" in args.stages else ()
    if "metrics" in args.stages or "sample-panels" in figure_names:
        required["daily event masks"] = [mask_path(daily_first_paths, case, args) for case in daily_cases]
    if "paper-metrics" in args.stages:
        required["ERA5 smoothed daily fields"] = [
            era5_daily_path(daily_first_paths, case, args) for case in daily_cases
        ]
        required["ACE2 smoothed daily fields"] = [
            ace2_daily_path(daily_first_paths, case, args) for case in daily_cases
        ]
        if args.twcrps_tail_threshold_source == "ace2_climatology":
            month_days = sorted({(case.month, case.day) for case in daily_cases})
            required["frozen threshold fields"] = [
                threshold_path(daily_first_paths, month, day, args) for month, day in month_days
            ]
    return required


def validate_daily_first_artifacts(
    daily_cases: list[DailyCase],
    daily_first_paths,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Fail before computation when the Workflow 3 root is absent or incomplete."""
    root = Path(args.daily_first_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Workflow 3 root does not exist: {root}\n"
            "Pass --daily-first-root pointing at daily_first_shape_sweep."
        )
    required = required_daily_first_artifacts(daily_cases, daily_first_paths, args)
    counts: dict[str, int] = {}
    missing_by_kind: dict[str, list[Path]] = {}
    for kind, paths in required.items():
        missing = [path for path in paths if not path.is_file()]
        counts[kind] = len(paths) - len(missing)
        if missing:
            missing_by_kind[kind] = missing
    if missing_by_kind:
        lines = [f"Incomplete Workflow 3 inputs under {root}:"]
        for kind, missing in missing_by_kind.items():
            lines.append(f"- {kind}: missing {len(missing)} of {len(required[kind])}")
            lines.extend(f"  {path}" for path in missing[:4])
            if len(missing) > 4:
                lines.append(f"  ... and {len(missing) - 4} more")
        raise FileNotFoundError("\n".join(lines))
    return counts


def metric_slug(args: argparse.Namespace) -> str:
    return event_slug(args)


def daily_part_path(paths: SupportPaths, rank: int, args: argparse.Namespace) -> Path:
    return paths.parts / f"daily_temporal_support_rank{rank:03d}_{metric_slug(args)}.csv"


def daily_metrics_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"daily_temporal_support_metrics_{metric_slug(args)}.csv"


def weekly_summary_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"weekly_temporal_support_summary_{metric_slug(args)}.csv"


def overall_summary_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"overall_temporal_support_summary_{metric_slug(args)}.csv"


def manifest_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"workflow5_temporal_support_manifest_{metric_slug(args)}.json"


def twcrps_part_path(paths: SupportPaths, rank: int, args: argparse.Namespace) -> Path:
    return paths.parts / f"paper_twcrps_daily_rank{rank:03d}_{metric_slug(args)}.csv"


def twcrps_daily_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"paper_twcrps_daily_{metric_slug(args)}.csv"


def twcrps_summary_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"paper_twcrps_summary_{metric_slug(args)}.csv"


def chi_summary_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"paper_chi_distance_summary_{metric_slug(args)}.csv"


def are_reference_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"paper_are_reference_metrics_{metric_slug(args)}.csv"


def are_summary_path(paths: SupportPaths, args: argparse.Namespace) -> Path:
    return paths.tables / f"paper_are_summary_{metric_slug(args)}.csv"


def area_sum(mask: np.ndarray, area: np.ndarray) -> float:
    return float(np.nansum(np.where(mask, area, 0.0)))


def weighted_sum(values: np.ndarray, valid: np.ndarray, area: np.ndarray) -> float:
    return float(np.nansum(np.where(valid, values * area, 0.0)))


def weighted_mean(values: np.ndarray, valid: np.ndarray, area: np.ndarray) -> float:
    return safe_ratio(weighted_sum(values, valid, area), area_sum(valid, area))


def safe_ratio(num: float, den: float, *, empty_value: float = float("nan")) -> float:
    return float(num / den) if den > 0 else empty_value


@dataclass(frozen=True)
class ProbabilityFields:
    probability: np.ndarray
    valid: np.ndarray
    event_count: np.ndarray
    valid_count: np.ndarray


def temporal_member_support(
    cube: MaskCube,
    index: int,
    member_index: int,
    tau_days: int,
    radius_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One member vote: event anywhere in the temporal/spatial neighborhood."""
    support = np.zeros_like(cube.era5_mask[index], dtype=bool)
    valid = np.zeros_like(cube.era5_valid[index], dtype=bool)
    for _, neighbor_index in neighbor_indices(cube, index, tau_days):
        member_valid = cube.ace2_valid[neighbor_index, member_index]
        member_mask = dilate_mask(cube.ace2_mask[neighbor_index, member_index], member_valid, radius_pixels)
        support |= member_mask & member_valid
        valid |= member_valid
    return support, valid


def temporal_support_probability(
    cube: MaskCube,
    index: int,
    tau_days: int,
    radius_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    fields = temporal_support_probability_fields(cube, index, tau_days, radius_pixels)
    return fields.probability, fields.valid


def temporal_support_probability_fields(
    cube: MaskCube,
    index: int,
    tau_days: int,
    radius_pixels: int,
) -> ProbabilityFields:
    event_count = np.zeros_like(cube.era5_mask[index], dtype=float)
    valid_count = np.zeros_like(cube.era5_mask[index], dtype=float)
    for member_index in range(len(cube.members)):
        support, valid = temporal_member_support(cube, index, member_index, tau_days, radius_pixels)
        event_count += np.where(valid & support, 1.0, 0.0)
        valid_count += np.where(valid, 1.0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        probability = np.where(valid_count > 0, event_count / valid_count, np.nan)
    valid = valid_count > 0
    return ProbabilityFields(probability=probability, valid=valid, event_count=event_count, valid_count=valid_count)


def temporal_member_day_probability_fields(
    cube: MaskCube,
    index: int,
    tau_days: int,
    radius_pixels: int,
) -> ProbabilityFields:
    event_count = np.zeros_like(cube.era5_mask[index], dtype=float)
    valid_count = np.zeros_like(cube.era5_mask[index], dtype=float)
    for member_index in range(len(cube.members)):
        for _, neighbor_index in neighbor_indices(cube, index, tau_days):
            member_valid = cube.ace2_valid[neighbor_index, member_index]
            member_mask = dilate_mask(cube.ace2_mask[neighbor_index, member_index], member_valid, radius_pixels)
            event_count += np.where(member_valid & member_mask, 1.0, 0.0)
            valid_count += np.where(member_valid, 1.0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        probability = np.where(valid_count > 0, event_count / valid_count, np.nan)
    valid = valid_count > 0
    return ProbabilityFields(probability=probability, valid=valid, event_count=event_count, valid_count=valid_count)


def soft_probability_metrics(obs: np.ndarray, probability: np.ndarray, valid: np.ndarray, area: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=bool) & valid
    non_event = (~obs) & valid
    p = np.where(valid, np.asarray(probability, dtype=float), np.nan)
    p = np.clip(p, 0.0, 1.0)

    obs_area = area_sum(obs, area)
    non_event_area = area_sum(non_event, area)
    valid_area = area_sum(valid, area)
    expected_area = weighted_sum(p, valid, area)
    soft_intersection = weighted_sum(np.where(obs, p, 0.0), valid, area)
    soft_union = weighted_sum(np.where(valid, obs.astype(float) + p - obs.astype(float) * p, 0.0), valid, area)
    false_alarm_mass = weighted_sum(np.where(non_event, p, 0.0), valid, area)
    miss_mass = weighted_sum(np.where(obs, 1.0 - p, 0.0), valid, area)
    soft_recall = safe_ratio(soft_intersection, obs_area)
    soft_precision = safe_ratio(soft_intersection, expected_area)
    soft_f1 = safe_ratio(2.0 * soft_recall * soft_precision, soft_recall + soft_precision)
    brier = safe_ratio(weighted_sum((p - obs.astype(float)) ** 2, valid, area), valid_area)
    event_brier = safe_ratio(weighted_sum(np.where(obs, (1.0 - p) ** 2, 0.0), valid, area), obs_area)
    nonevent_brier = safe_ratio(weighted_sum(np.where(non_event, p**2, 0.0), valid, area), non_event_area)
    return {
        "era5_event_area_km2": obs_area,
        "ace2_expected_support_area_km2": expected_area,
        "support_area_ratio_ace2_over_era5": safe_ratio(expected_area, obs_area),
        "soft_intersection_area_km2": soft_intersection,
        "soft_union_area_km2": soft_union,
        "soft_iou": 1.0 if soft_union == 0 else safe_ratio(soft_intersection, soft_union),
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "soft_f1": soft_f1,
        "false_alarm_probability_mass_km2": false_alarm_mass,
        "false_alarm_mass_fraction": safe_ratio(false_alarm_mass, expected_area),
        "mean_false_alarm_probability": safe_ratio(false_alarm_mass, non_event_area),
        "miss_probability_mass_km2": miss_mass,
        "mean_event_probability": soft_recall,
        "mean_miss_probability": safe_ratio(miss_mass, obs_area),
        "area_weighted_brier": brier,
        "event_brier": event_brier,
        "nonevent_brier": nonevent_brier,
    }


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def temporal_support_metric_row(
    cube: MaskCube,
    index: int,
    tau_days: int,
    radius_pixels: int,
    probability_thresholds: list[float],
) -> dict[str, object]:
    case = cube.cases[index]
    obs = cube.era5_mask[index]
    obs_valid = cube.era5_valid[index]
    support_fields = temporal_support_probability_fields(cube, index, tau_days, radius_pixels)
    probability = support_fields.probability
    probability_valid = support_fields.valid
    valid = obs_valid & probability_valid & np.isfinite(probability)
    member_day_fields = temporal_member_day_probability_fields(cube, index, tau_days, radius_pixels)
    member_day_probability = member_day_fields.probability
    member_day_valid = obs_valid & member_day_fields.valid & np.isfinite(member_day_probability)
    neighbors = neighbor_indices(cube, index, tau_days)

    row: dict[str, object] = {
        "date": str(case.date.date()),
        "year": int(case.year),
        "week_index": int(case.week),
        "day_of_week": int(case.day_of_week),
        "calendar_month_day": f"{case.month:02d}-{case.day:02d}",
        "tau_days": int(tau_days),
        "radius_pixels": int(radius_pixels),
        "nominal_temporal_window_days": int(2 * tau_days + 1),
        "actual_temporal_window_days": int(len(neighbors)),
        "mean_supporting_members": weighted_mean(support_fields.event_count, valid, cube.area),
        "mean_valid_members": weighted_mean(support_fields.valid_count, valid, cube.area),
        "event_area_mean_supporting_members": weighted_mean(support_fields.event_count, obs & valid, cube.area),
        "event_area_mean_valid_members": weighted_mean(support_fields.valid_count, obs & valid, cube.area),
        "mean_event_member_days": weighted_mean(member_day_fields.event_count, member_day_valid, cube.area),
        "mean_valid_member_days": weighted_mean(member_day_fields.valid_count, member_day_valid, cube.area),
        "event_area_mean_event_member_days": weighted_mean(member_day_fields.event_count, obs & member_day_valid, cube.area),
        "event_area_mean_valid_member_days": weighted_mean(member_day_fields.valid_count, obs & member_day_valid, cube.area),
        "area_weighted_member_day_probability": weighted_mean(member_day_probability, member_day_valid, cube.area),
    }
    row.update(soft_probability_metrics(obs, probability, valid, cube.area))
    row.update(
        prefixed_metrics(
            soft_probability_metrics(obs, member_day_probability, member_day_valid, cube.area),
            "member_day_",
        )
    )
    for threshold in probability_thresholds:
        pred = np.asarray(probability >= threshold, dtype=bool) & valid
        metrics = binary_overlap_metrics(obs, pred, valid, cube.area)
        suffix = threshold_suffix(threshold)
        row[f"binary_iou_{suffix}"] = metrics["exact_iou"]
        row[f"binary_f1_{suffix}"] = metrics["exact_dice_f1"]
        row[f"binary_precision_{suffix}"] = metrics["exact_precision"]
        row[f"binary_recall_{suffix}"] = metrics["exact_recall"]
        row[f"binary_area_ratio_{suffix}"] = metrics["area_ratio_ace2_over_era5"]
        member_day_pred = np.asarray(member_day_probability >= threshold, dtype=bool) & member_day_valid
        member_day_metrics = binary_overlap_metrics(obs, member_day_pred, member_day_valid, cube.area)
        row[f"member_day_binary_iou_{suffix}"] = member_day_metrics["exact_iou"]
        row[f"member_day_binary_f1_{suffix}"] = member_day_metrics["exact_dice_f1"]
        row[f"member_day_binary_precision_{suffix}"] = member_day_metrics["exact_precision"]
        row[f"member_day_binary_recall_{suffix}"] = member_day_metrics["exact_recall"]
        row[f"member_day_binary_area_ratio_{suffix}"] = member_day_metrics["area_ratio_ace2_over_era5"]
    return row


def run_metric_part(cube: MaskCube, case_indices: list[int], paths: SupportPaths, args: argparse.Namespace, rank: int) -> Path:
    out = daily_part_path(paths, rank, args)
    if out.exists() and not args.force_metrics:
        return out
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    thresholds = parse_float_list(args.probability_thresholds, name="probability thresholds")
    rows: list[dict[str, object]] = []
    iterator = tqdm(case_indices, desc=f"rank {rank} workflow5 metrics", disable=rank != 0)
    for index in iterator:
        for tau in taus:
            for radius in radii:
                rows.append(temporal_support_metric_row(cube, index, tau, radius, thresholds))
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def key_metric_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "soft_",
        "binary_",
        "member_day_",
        "false_alarm_",
        "mean_",
        "event_area_",
        "area_weighted_",
        "event_brier",
        "nonevent_brier",
    )
    cols = [
        "era5_event_area_km2",
        "ace2_expected_support_area_km2",
        "support_area_ratio_ace2_over_era5",
        "miss_probability_mass_km2",
        "actual_temporal_window_days",
        "nominal_temporal_window_days",
    ]
    cols.extend(col for col in df.columns if col.startswith(prefixes))
    return [col for col in dict.fromkeys(cols) if col in df.columns]


def combine_and_summarize(parts: list[Path], paths: SupportPaths, args: argparse.Namespace) -> dict[str, str]:
    existing = [path for path in parts if path.exists() and path.stat().st_size > 0]
    if not existing:
        raise FileNotFoundError("No Workflow 5 metric part files were written.")
    daily_df = pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.sort_values(["year", "week_index", "day_of_week", "tau_days", "radius_pixels"])
    daily_csv = daily_metrics_path(paths, args)
    daily_df.to_csv(daily_csv, index=False)

    metrics = key_metric_columns(daily_df)
    weekly_rows: list[dict[str, object]] = []
    for (year, week, tau, radius), group in daily_df.groupby(["year", "week_index", "tau_days", "radius_pixels"], dropna=False):
        row = {
            "year": int(year),
            "week_index": int(week),
            "week_start": str(pd.Timestamp(group["date"].min()).date()),
            "week_end": str(pd.Timestamp(group["date"].max()).date()),
            "tau_days": int(tau),
            "radius_pixels": int(radius),
            "n_daily_rows": int(len(group)),
        }
        row.update(summarize_group(group, metrics, prefix="weekly_"))
        weekly_rows.append(row)
    weekly_df = pd.DataFrame(weekly_rows).sort_values(["year", "week_index", "tau_days", "radius_pixels"])
    weekly_csv = weekly_summary_path(paths, args)
    weekly_df.to_csv(weekly_csv, index=False)

    overall_rows: list[dict[str, object]] = []
    for (tau, radius), group in daily_df.groupby(["tau_days", "radius_pixels"], dropna=False):
        row = {"tau_days": int(tau), "radius_pixels": int(radius), "n_daily_rows": int(len(group))}
        row.update(summarize_group(group, metrics))
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values(["tau_days", "radius_pixels"])
    overall_csv = overall_summary_path(paths, args)
    overall_df.to_csv(overall_csv, index=False)

    return {
        "daily_temporal_support_metrics": str(daily_csv),
        "weekly_temporal_support_summary": str(weekly_csv),
        "overall_temporal_support_summary": str(overall_csv),
    }


def paper_quantiles(args: argparse.Namespace) -> list[float]:
    quantiles = parse_float_list(args.paper_quantiles, name="paper quantiles")
    for quantile in quantiles:
        if quantile <= 0.0 or quantile >= 1.0:
            raise ValueError(f"Paper quantiles must be strictly between 0 and 1, got {quantile}")
    return quantiles


def chi_distance_edges(args: argparse.Namespace) -> list[float]:
    edges = parse_unbounded_float_list(args.chi_distance_bins_km, name="chi distance-bin edges")
    if len(edges) < 2:
        raise ValueError("At least two --chi-distance-bins-km edges are required.")
    if any(edge < 0 for edge in edges):
        raise ValueError("Distance-bin edges must be nonnegative.")
    if any(edges[idx + 1] <= edges[idx] for idx in range(len(edges) - 1)):
        raise ValueError("Distance-bin edges must be strictly increasing.")
    return edges


def tail_weighted_crps(
    samples: np.ndarray,
    obs: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    obs = np.asarray(obs, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    sample_valid = np.isfinite(samples)
    valid_count = sample_valid.sum(axis=0).astype(float)
    valid = np.isfinite(obs) & np.isfinite(threshold) & (valid_count > 0)
    threshold_3d = threshold[np.newaxis, :, :]
    censored_samples = np.where(sample_valid & np.isfinite(threshold_3d), np.maximum(samples, threshold_3d), np.nan)
    censored_obs = np.where(valid, np.maximum(obs, threshold), np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        term1 = np.nanmean(np.abs(censored_samples - censored_obs[np.newaxis, :, :]), axis=0)
        pairwise = np.abs(censored_samples[:, np.newaxis, :, :] - censored_samples[np.newaxis, :, :, :])
        term2 = np.nanmean(pairwise, axis=(0, 1))
    crps = term1 - 0.5 * term2
    return np.where(valid, crps, np.nan), valid_count


def twcrps_tail_threshold(
    case: DailyCase,
    samples: np.ndarray,
    daily_first_paths,
    args: argparse.Namespace,
) -> np.ndarray:
    q = float(args.twcrps_tail_quantile)
    if q <= 0.0 or q >= 1.0:
        raise ValueError(f"--twcrps-tail-quantile must be strictly between 0 and 1, got {q}")
    if args.twcrps_tail_threshold_source == "case_ensemble":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            return np.nanquantile(samples, q, axis=0).astype(np.float32)
    expected = float(args.percentile) / 100.0
    if not np.isclose(q, expected):
        raise ValueError(
            "--twcrps-tail-threshold-source ace2_climatology requires "
            "--twcrps-tail-quantile to match --percentile/100."
        )
    with xr.open_dataset(threshold_path(daily_first_paths, case.month, case.day, args)) as ds:
        return ds["ace2_threshold_C"].values.astype(np.float32)


def twcrps_metric_row(case: DailyCase, daily_first_paths, args: argparse.Namespace) -> dict[str, object]:
    with xr.open_dataset(era5_daily_path(daily_first_paths, case, args)) as era5_ds:
        obs = era5_ds["era5_smoothed_daily_tmax_C"].values.astype(np.float32)
        lat = era5_ds["lat"].values.astype(float)
        lon = era5_ds["lon"].values.astype(float)
    with xr.open_dataset(ace2_daily_path(daily_first_paths, case, args)) as ace2_ds:
        samples = ace2_ds["ace2_smoothed_daily_tmax_C"].values.astype(np.float32)
    threshold = twcrps_tail_threshold(case, samples, daily_first_paths, args)
    crps, valid_members = tail_weighted_crps(samples, obs, threshold)
    area = cell_area_km2(lat, lon)
    valid = np.isfinite(crps)
    era5_tail = valid & np.isfinite(obs) & np.isfinite(threshold) & (obs > threshold)
    vals = crps[valid]
    row: dict[str, object] = {
        "date": str(case.date.date()),
        "year": int(case.year),
        "week_index": int(case.week),
        "day_of_week": int(case.day_of_week),
        "calendar_month_day": f"{case.month:02d}-{case.day:02d}",
        "twcrps_tail_quantile": float(args.twcrps_tail_quantile),
        "twcrps_tail_threshold_source": args.twcrps_tail_threshold_source,
        "n_valid_pixels": int(np.count_nonzero(valid)),
        "valid_area_km2": area_sum(valid, area),
        "era5_tail_area_km2": area_sum(era5_tail, area),
        "mean_valid_ace2_members_per_pixel": weighted_mean(valid_members, valid, area),
        "area_weighted_twcrps": weighted_mean(crps, valid, area),
        "area_weighted_twcrps_on_era5_tail": weighted_mean(crps, era5_tail, area),
    }
    if vals.size:
        row.update(
            {
                "twcrps_mean_unweighted": float(np.nanmean(vals)),
                "twcrps_median": float(np.nanmedian(vals)),
                "twcrps_q25": float(np.nanpercentile(vals, 25)),
                "twcrps_q75": float(np.nanpercentile(vals, 75)),
                "twcrps_q90": float(np.nanpercentile(vals, 90)),
                "twcrps_max": float(np.nanmax(vals)),
            }
        )
    else:
        row.update(
            {
                "twcrps_mean_unweighted": np.nan,
                "twcrps_median": np.nan,
                "twcrps_q25": np.nan,
                "twcrps_q75": np.nan,
                "twcrps_q90": np.nan,
                "twcrps_max": np.nan,
            }
        )
    return row


def run_twcrps_part(
    daily_cases: list[DailyCase],
    case_indices: list[int],
    daily_first_paths,
    paths: SupportPaths,
    args: argparse.Namespace,
    rank: int,
) -> Path:
    out = twcrps_part_path(paths, rank, args)
    if out.exists() and not args.force_paper_metrics:
        return out
    rows: list[dict[str, object]] = []
    iterator = tqdm(case_indices, desc=f"rank {rank} twCRPS", disable=rank != 0)
    for index in iterator:
        rows.append(twcrps_metric_row(daily_cases[index], daily_first_paths, args))
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def combine_twcrps_parts(parts: list[Path], paths: SupportPaths, args: argparse.Namespace) -> dict[str, str]:
    existing = [path for path in parts if path.exists() and path.stat().st_size > 0]
    if not existing:
        raise FileNotFoundError("No twCRPS metric part files were written.")
    daily_df = pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.sort_values(["year", "week_index", "day_of_week"])
    daily_csv = twcrps_daily_path(paths, args)
    daily_df.to_csv(daily_csv, index=False)
    metrics = [
        col
        for col in daily_df.columns
        if (
            col.startswith("twcrps_")
            or col.startswith("area_weighted_twcrps")
            or col in ("valid_area_km2", "era5_tail_area_km2", "mean_valid_ace2_members_per_pixel")
        )
        and pd.api.types.is_numeric_dtype(daily_df[col])
    ]
    summary_rows: list[dict[str, object]] = []
    row: dict[str, object] = {"n_daily_rows": int(len(daily_df))}
    row.update(summarize_group(daily_df, metrics))
    summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = twcrps_summary_path(paths, args)
    summary_df.to_csv(summary_csv, index=False)
    return {"paper_twcrps_daily": str(daily_csv), "paper_twcrps_summary": str(summary_csv)}


def load_continuous_source_matrix(
    daily_cases: list[DailyCase],
    daily_first_paths,
    args: argparse.Namespace,
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    for case in tqdm(daily_cases, desc=f"load {source} continuous fields"):
        if source == "era5":
            path = era5_daily_path(daily_first_paths, case, args)
            with xr.open_dataset(path) as ds:
                values = ds["era5_smoothed_daily_tmax_C"].values.astype(np.float32)
                if lat is None:
                    lat = ds["lat"].values.astype(float)
                    lon = ds["lon"].values.astype(float)
                rows.append(values.reshape(1, -1))
        elif source == "ace2":
            path = ace2_daily_path(daily_first_paths, case, args)
            with xr.open_dataset(path) as ds:
                values = ds["ace2_smoothed_daily_tmax_C"].values.astype(np.float32)
                if lat is None:
                    lat = ds["lat"].values.astype(float)
                    lon = ds["lon"].values.astype(float)
                rows.append(values.reshape(values.shape[0], -1))
        else:
            raise ValueError(f"Unknown paper metric source {source!r}")
    if not rows or lat is None or lon is None:
        raise ValueError(f"No continuous fields loaded for source {source!r}")
    return np.concatenate(rows, axis=0).astype(np.float32), lat, lon


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    dlambda = np.deg2rad(dlon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def sample_pairs_by_distance(
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    valid_cells: np.ndarray,
    edges: list[float],
    max_pairs_per_bin: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_pairs_per_bin <= 0 or valid_cells.size < 2:
        return pd.DataFrame(columns=["cell_i", "cell_j", "distance_km", "bin_start_km", "bin_end_km"])
    n_bins = len(edges) - 1
    pairs: list[list[float]] = []
    counts = np.zeros(n_bins, dtype=int)
    attempts = 0
    max_attempts = max(250_000, max_pairs_per_bin * n_bins * 250)
    batch_size = min(100_000, max(10_000, max_pairs_per_bin * n_bins * 20))
    while np.any(counts < max_pairs_per_bin) and attempts < max_attempts:
        attempts += batch_size
        i = rng.choice(valid_cells, size=batch_size, replace=True)
        j = rng.choice(valid_cells, size=batch_size, replace=True)
        different = i != j
        if not np.any(different):
            continue
        i = i[different]
        j = j[different]
        distances = haversine_vectorized(lat_flat[i], lon_flat[i], lat_flat[j], lon_flat[j])
        bin_indices = np.searchsorted(edges, distances, side="right") - 1
        for bin_index in range(n_bins):
            need = max_pairs_per_bin - counts[bin_index]
            if need <= 0:
                continue
            hits = np.flatnonzero(bin_indices == bin_index)
            if hits.size == 0:
                continue
            chosen = hits[:need]
            for idx in chosen:
                pairs.append([int(i[idx]), int(j[idx]), float(distances[idx]), float(edges[bin_index]), float(edges[bin_index + 1])])
            counts[bin_index] += int(chosen.size)
    return pd.DataFrame(pairs, columns=["cell_i", "cell_j", "distance_km", "bin_start_km", "bin_end_km"])


def summarize_numeric(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    key = f"{prefix}_" if prefix else ""
    if vals.size == 0:
        return {
            f"{key}mean": np.nan,
            f"{key}median": np.nan,
            f"{key}q25": np.nan,
            f"{key}q75": np.nan,
            f"{key}min": np.nan,
            f"{key}max": np.nan,
        }
    return {
        f"{key}mean": float(np.mean(vals)),
        f"{key}median": float(np.median(vals)),
        f"{key}q25": float(np.percentile(vals, 25)),
        f"{key}q75": float(np.percentile(vals, 75)),
        f"{key}min": float(np.min(vals)),
        f"{key}max": float(np.max(vals)),
    }


def chi_distance_summary(
    source: str,
    values: np.ndarray,
    pairs_df: pd.DataFrame,
    quantiles: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if pairs_df.empty:
        return rows
    pair_i = pairs_df["cell_i"].to_numpy(dtype=int)
    pair_j = pairs_df["cell_j"].to_numpy(dtype=int)
    for quantile in quantiles:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            threshold = np.nanquantile(values, quantile, axis=0)
        exceed = np.isfinite(values) & (values > threshold[np.newaxis, :])
        denom = exceed.sum(axis=0).astype(float)
        both_counts = np.array([np.count_nonzero(exceed[:, i] & exceed[:, j]) for i, j in zip(pair_i, pair_j)], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            chi_i_given_j = np.where(denom[pair_i] > 0, both_counts / denom[pair_i], np.nan)
            chi_j_given_i = np.where(denom[pair_j] > 0, both_counts / denom[pair_j], np.nan)
            symmetric_chi = 0.5 * (chi_i_given_j + chi_j_given_i)
        temp = pairs_df.copy()
        temp["chi"] = symmetric_chi
        for (bin_start, bin_end), group in temp.groupby(["bin_start_km", "bin_end_km"], dropna=False):
            chi_vals = pd.to_numeric(group["chi"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            row: dict[str, object] = {
                "source": source,
                "quantile": float(quantile),
                "distance_bin_start_km": float(bin_start),
                "distance_bin_end_km": float(bin_end),
                "n_pairs": int(len(group)),
                "n_finite_pairs": int(chi_vals.size),
                "mean_distance_km": float(group["distance_km"].mean()) if len(group) else np.nan,
            }
            row.update({f"chi_{key}": value for key, value in summarize_numeric(chi_vals).items()})
            rows.append(row)
    return rows


def are_reference_metrics(
    source: str,
    values: np.ndarray,
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    area_flat: np.ndarray,
    reference_cells: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quantile in quantiles:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            threshold = np.nanquantile(values, quantile, axis=0)
        exceed = np.isfinite(values) & (values > threshold[np.newaxis, :])
        for ref in reference_cells:
            ref_exceed = exceed[:, ref]
            denom = int(np.count_nonzero(ref_exceed))
            if denom <= 0:
                are_km = np.nan
                joint_area_km2 = np.nan
            else:
                joint_counts = exceed[ref_exceed].sum(axis=0).astype(float)
                joint_area_km2 = float(np.nansum(area_flat * joint_counts))
                are_km = float(np.sqrt(joint_area_km2 / (np.pi * denom)))
            rows.append(
                {
                    "source": source,
                    "quantile": float(quantile),
                    "reference_cell": int(ref),
                    "reference_lat": float(lat_flat[ref]),
                    "reference_lon": float(lon_flat[ref]),
                    "n_reference_exceedances": int(denom),
                    "joint_exceedance_area_sum_km2": joint_area_km2,
                    "are_km": are_km,
                }
            )
    return rows


def summarize_are_rows(are_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if are_df.empty:
        return pd.DataFrame(rows)
    for (source, quantile), group in are_df.groupby(["source", "quantile"], dropna=False):
        vals = finite_numeric(group, "are_km").dropna().to_numpy(dtype=float)
        row: dict[str, object] = {
            "source": source,
            "quantile": float(quantile),
            "n_reference_points": int(len(group)),
            "n_finite_reference_points": int(vals.size),
        }
        row.update({f"are_km_{key}": value for key, value in summarize_numeric(vals).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "quantile"])


def run_dependence_paper_metrics(
    daily_cases: list[DailyCase],
    daily_first_paths,
    paths: SupportPaths,
    args: argparse.Namespace,
) -> dict[str, str]:
    chi_csv = chi_summary_path(paths, args)
    are_ref_csv = are_reference_path(paths, args)
    are_csv = are_summary_path(paths, args)
    if (
        chi_csv.exists()
        and are_ref_csv.exists()
        and are_csv.exists()
        and not args.force_paper_metrics
    ):
        return {
            "paper_chi_distance_summary": str(chi_csv),
            "paper_are_reference_metrics": str(are_ref_csv),
            "paper_are_summary": str(are_csv),
        }
    quantiles = paper_quantiles(args)
    edges = chi_distance_edges(args)
    rng = np.random.default_rng(int(args.random_seed) + 5505)
    chi_rows: list[dict[str, object]] = []
    are_rows: list[dict[str, object]] = []
    for source in args.paper_sources:
        values, lat, lon = load_continuous_source_matrix(daily_cases, daily_first_paths, args, source)
        area = cell_area_km2(lat, lon).reshape(-1)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        lat_flat = lat2d.reshape(-1).astype(float)
        lon_flat = lon2d.reshape(-1).astype(float)
        valid_cells = np.flatnonzero(np.any(np.isfinite(values), axis=0))
        pairs_df = sample_pairs_by_distance(
            lat_flat,
            lon_flat,
            valid_cells,
            edges,
            int(args.chi_max_pairs_per_bin),
            rng,
        )
        chi_rows.extend(chi_distance_summary(source, values, pairs_df, quantiles))
        stride = max(1, int(args.are_reference_stride))
        reference_candidates = valid_cells[::stride]
        if args.are_max_reference_points > 0 and reference_candidates.size > args.are_max_reference_points:
            reference_cells = np.sort(
                rng.choice(reference_candidates, size=int(args.are_max_reference_points), replace=False)
            )
        else:
            reference_cells = reference_candidates
        are_rows.extend(are_reference_metrics(source, values, lat_flat, lon_flat, area, reference_cells, quantiles))
    chi_df = pd.DataFrame(chi_rows)
    if not chi_df.empty:
        chi_df = chi_df.sort_values(["source", "quantile", "distance_bin_start_km", "distance_bin_end_km"])
    chi_df.to_csv(chi_csv, index=False)
    are_df = pd.DataFrame(are_rows)
    if not are_df.empty:
        are_df = are_df.sort_values(["source", "quantile", "reference_cell"])
    are_df.to_csv(are_ref_csv, index=False)
    summarize_are_rows(are_df).to_csv(are_csv, index=False)
    return {
        "paper_chi_distance_summary": str(chi_csv),
        "paper_are_reference_metrics": str(are_ref_csv),
        "paper_are_summary": str(are_csv),
    }


def heatmap_metric(daily_df: pd.DataFrame, metric: str, taus: list[int], radii: list[int]) -> np.ndarray:
    values = np.full((len(radii), len(taus)), np.nan, dtype=float)
    for i, radius in enumerate(radii):
        for j, tau in enumerate(taus):
            subset = daily_df[(daily_df["tau_days"] == tau) & (daily_df["radius_pixels"] == radius)]
            vals = finite_numeric(subset, metric).dropna()
            if not vals.empty:
                values[i, j] = float(vals.median())
    return values


def plot_support_sweep(daily_df: pd.DataFrame, paths: SupportPaths, args: argparse.Namespace) -> Path:
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    specs = [
        ("soft_recall", "event support / soft recall"),
        ("soft_precision", "probability-mass precision"),
        ("soft_f1", "soft F1"),
        ("soft_iou", "soft IoU"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    last_mesh = None
    for ax, (metric, title) in zip(axes.ravel(), specs):
        values = heatmap_metric(daily_df, metric, taus, radii)
        last_mesh = ax.imshow(values, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(taus)))
        ax.set_xticklabels([str(tau) for tau in taus])
        ax.set_yticks(np.arange(len(radii)))
        ax.set_yticklabels([str(radius) for radius in radii])
        ax.set_xlabel("temporal neighborhood ±tau (days)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(radii)):
            for j in range(len(taus)):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="0.1")
    axes[0, 0].set_ylabel("spatial radius r (grid pixels)")
    axes[1, 0].set_ylabel("spatial radius r (grid pixels)")
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=axes, orientation="horizontal", fraction=0.045, pad=0.07)
        cbar.set_label("median score across daily cases", fontsize=9)
    fig.suptitle("Workflow 5: ACE2 temporal-neighborhood support for ERA5 daily events", fontsize=13, fontweight="bold")
    out = paths.figures / f"workflow5_temporal_support_sweep_{metric_slug(args)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timeseries(weekly_df: pd.DataFrame, paths: SupportPaths, args: argparse.Namespace) -> Path:
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    radius = 0 if 0 in radii else min(radii)
    df = weekly_df[weekly_df["radius_pixels"] == radius].copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.2), constrained_layout=True, sharex=True)
    colors = ["#2166ac", "#1a9850", "#fdae61", "#762a83", "#d6604d", "#4d4d4d"]
    for ax, metric, ylabel in [
        (axes[0], "weekly_soft_recall_mean", "weekly mean event support"),
        (axes[1], "weekly_soft_f1_mean", "weekly mean soft F1"),
    ]:
        for idx, tau in enumerate(taus):
            sub = df[df["tau_days"] == tau].sort_values("week_start")
            x = np.arange(len(sub))
            y = finite_numeric(sub, metric).to_numpy(dtype=float)
            ax.plot(x, y, color=colors[idx % len(colors)], linewidth=0.9, label=f"±{tau}d")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(frameon=False, fontsize=8, ncol=min(len(taus), 6))
    axes[1].set_xlabel("weekly case, chronological order")
    if len(df) > 0:
        first_tau = min(taus)
        ref = df[df["tau_days"] == first_tau].sort_values("week_start")
        tick_idx = np.linspace(0, len(ref) - 1, min(12, len(ref)), dtype=int)
        axes[1].set_xticks(tick_idx)
        axes[1].set_xticklabels([str(pd.Timestamp(ref.iloc[i]["week_start"]).date()) for i in tick_idx], rotation=30, ha="right")
    fig.suptitle(f"Workflow 5 temporal support through time, exact pixels (r={radius})", fontsize=13, fontweight="bold")
    out = paths.figures / f"workflow5_temporal_support_timeseries_{metric_slug(args)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spatial_timeseries(weekly_df: pd.DataFrame, paths: SupportPaths, args: argparse.Namespace) -> Path:
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    tau = 0 if 0 in taus else min(taus)
    df = weekly_df[weekly_df["tau_days"] == tau].copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.2), constrained_layout=True, sharex=True)
    colors = ["#2166ac", "#1a9850", "#fdae61", "#762a83", "#d6604d", "#4d4d4d", "#80cdc1"]
    for ax, metric, ylabel in [
        (axes[0], "weekly_soft_recall_mean", "weekly mean event support"),
        (axes[1], "weekly_soft_f1_mean", "weekly mean soft F1"),
    ]:
        for idx, radius in enumerate(radii):
            sub = df[df["radius_pixels"] == radius].sort_values("week_start")
            x = np.arange(len(sub))
            y = finite_numeric(sub, metric).to_numpy(dtype=float)
            ax.plot(x, y, color=colors[idx % len(colors)], linewidth=0.9, label=f"r={radius}")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(frameon=False, fontsize=8, ncol=min(len(radii), 7))
    axes[1].set_xlabel("weekly case, chronological order")
    if len(df) > 0:
        first_radius = min(radii)
        ref = df[df["radius_pixels"] == first_radius].sort_values("week_start")
        tick_idx = np.linspace(0, len(ref) - 1, min(12, len(ref)), dtype=int)
        axes[1].set_xticks(tick_idx)
        axes[1].set_xticklabels([str(pd.Timestamp(ref.iloc[i]["week_start"]).date()) for i in tick_idx], rotation=30, ha="right")
    fig.suptitle(f"Workflow 5 spatial support through time, same-day only (±{tau}d)", fontsize=13, fontweight="bold")
    out = paths.figures / f"workflow5_spatial_support_timeseries_{metric_slug(args)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def finite_array(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = finite_numeric(df, col).dropna().to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def grouped_distribution_boxplot(
    ax,
    groups: list[int],
    series: list[int | None],
    values_for,
    *,
    series_label,
    colors: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    show_legend: bool = False,
) -> None:
    x_positions = np.arange(len(groups), dtype=float)
    n_series = max(1, len(series))
    width = 0.52 if n_series == 1 else min(0.11, 0.66 / n_series)
    offsets = np.array([0.0]) if n_series == 1 else (np.arange(n_series) - (n_series - 1) / 2.0) * width * 1.42
    legend_handles = []
    any_data = False
    for series_idx, series_value in enumerate(series):
        data: list[np.ndarray] = []
        positions: list[float] = []
        for group_idx, group_value in enumerate(groups):
            vals = values_for(group_value, series_value)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(float(x_positions[group_idx] + offsets[series_idx]))
        if not data:
            continue
        any_data = True
        color = colors[series_idx % len(colors)]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "0.1", "linewidth": 0.9},
            whiskerprops={"color": color, "linewidth": 0.7},
            capprops={"color": color, "linewidth": 0.7},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.42 if n_series > 1 else 0.55)
        if n_series > 1:
            legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=6, alpha=0.55, label=series_label(series_value)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(group) for group in groups])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    if not any_data:
        ax.text(0.5, 0.5, "metric columns not found", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=8, title="spatial radius", title_fontsize=8, ncol=2)


def plot_effect_distributions(daily_df: pd.DataFrame, paths: SupportPaths, args: argparse.Namespace) -> Path:
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    colors = ["#2166ac", "#1a9850", "#fdae61", "#762a83", "#d6604d", "#4d4d4d", "#80cdc1"]

    def subset_values(metric: str, tau: int, radius: int) -> np.ndarray:
        sub = daily_df[(daily_df["tau_days"] == tau) & (daily_df["radius_pixels"] == radius)]
        return finite_array(sub, metric)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18.2, 8.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.75, 1.0, 1.0]},
    )
    for row, metric, ylabel in [(0, "soft_iou", "soft IoU"), (1, "soft_f1", "soft F1")]:
        grouped_distribution_boxplot(
            axes[row, 0],
            taus,
            radii,
            lambda tau, radius: subset_values(metric, int(tau), int(radius)),
            series_label=lambda radius: f"r={int(radius)}",
            colors=colors,
            xlabel="temporal neighborhood ±tau (days)",
            ylabel=ylabel,
            title="Space + time support",
            show_legend=row == 0,
        )
        grouped_distribution_boxplot(
            axes[row, 1],
            taus,
            [None],
            lambda tau, _radius: subset_values(metric, int(tau), 0 if 0 in radii else min(radii)),
            series_label=lambda _radius: "temporal only",
            colors=["#2166ac"],
            xlabel="temporal neighborhood ±tau (days)",
            ylabel=ylabel,
            title="Temporal support only",
        )
        grouped_distribution_boxplot(
            axes[row, 2],
            radii,
            [None],
            lambda radius, _series: subset_values(metric, 0 if 0 in taus else min(taus), int(radius)),
            series_label=lambda _series: "spatial only",
            colors=["#1a9850"],
            xlabel="spatial radius r (grid pixels)",
            ylabel=ylabel,
            title="Spatial support only",
        )
    fig.suptitle("Workflow 5 soft IoU/F1 under spatial, temporal, and spatiotemporal support", fontsize=13, fontweight="bold")
    out = paths.figures / f"workflow5_effect_distributions_{metric_slug(args)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def choose_case_indices(cube: MaskCube, args: argparse.Namespace) -> list[int]:
    if args.sample_date:
        index = cube.date_to_index.get(pd.Timestamp(args.sample_date))
        if index is None:
            raise ValueError(f"--sample-date {args.sample_date} is not in the configured daily cases.")
        return [int(index)]
    rng = np.random.default_rng(args.random_seed + 51)
    n_cases = min(max(args.sample_cases, 0), len(cube.cases))
    if n_cases == 0:
        return []
    return sorted(int(i) for i in rng.choice(len(cube.cases), size=n_cases, replace=False))


def plot_sample_panels(cube: MaskCube, paths: SupportPaths, args: argparse.Namespace) -> list[Path]:
    case_indices = choose_case_indices(cube, args)
    if not case_indices:
        return []
    taus = parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days")
    radii = parse_neighborhood_pixels(args.spatial_radii_pixels)
    radius = 0 if 0 in radii else min(radii)
    rng = np.random.default_rng(args.random_seed + 53)
    chosen_members = rng.choice(
        np.arange(len(cube.members)),
        size=min(args.sample_members, len(cube.members)),
        replace=False,
    )
    outputs: list[Path] = []
    lon_plot = lon_for_plot(cube.lon)
    for case_index in case_indices:
        case = cube.cases[case_index]
        obs = cube.era5_mask[case_index]
        obs_valid = cube.era5_valid[case_index]
        nrows = len(taus)
        ncols = 2 + len(chosen_members)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.35 * ncols, 2.85 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        era5_mesh = None
        prob_mesh = None
        overlap_mesh = None
        for row, tau in enumerate(taus):
            if row == 0:
                ax = axes[row, 0]
                era5_mesh = ax.pcolormesh(
                    lon_plot,
                    cube.lat,
                    np.where(obs_valid, obs.astype(float), np.nan),
                    shading="auto",
                    cmap=ListedColormap(["#ffffff", "#d73027"]),
                    vmin=0,
                    vmax=1,
                )
                ax.set_title(f"ERA5 event mask\n{case.date.date()}", fontsize=8.5, fontweight="bold")
                format_map_axis(ax, lon_plot, cube.lat)
            else:
                axes[row, 0].axis("off")

            probability, probability_valid = temporal_support_probability(cube, case_index, tau, radius)
            valid = obs_valid & probability_valid & np.isfinite(probability)
            soft = soft_probability_metrics(obs, probability, valid, cube.area)
            ax = axes[row, 1]
            prob_mesh = ax.pcolormesh(lon_plot, cube.lat, probability, shading="auto", cmap=PROB_CMAP, vmin=0, vmax=1)
            if np.any(obs & obs_valid):
                ax.contour(
                    lon_plot,
                    cube.lat,
                    np.where(obs_valid, obs.astype(float), np.nan),
                    levels=[0.5],
                    colors="0.1",
                    linewidths=0.75,
                    alpha=0.85,
                    zorder=8,
                )
            ax.set_title(f"ACE2 P_tau\n±{tau}d, sF1={soft['soft_f1']:.2f}", fontsize=8.5, fontweight="bold")
            ax.text(
                -0.14,
                0.5,
                f"±{tau}d",
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
            format_map_axis(ax, lon_plot, cube.lat)

            for col, member_index in enumerate(chosen_members, start=2):
                support, support_valid = temporal_member_support(cube, case_index, int(member_index), tau, radius)
                member_valid = obs_valid & support_valid
                overlap = overlap_code(obs, support, member_valid)
                metrics = binary_overlap_metrics(obs, support, member_valid, cube.area)
                ax = axes[row, col]
                overlap_mesh = ax.pcolormesh(lon_plot, cube.lat, overlap, shading="auto", cmap=OVERLAP_CMAP, norm=OVERLAP_NORM)
                ax.set_title(
                    f"ACE2 m{int(cube.members[int(member_index)]):02d}\nunion IoU={metrics['exact_iou']:.2f}",
                    fontsize=8.5,
                    fontweight="bold",
                )
                format_map_axis(ax, lon_plot, cube.lat)
        if era5_mesh is not None:
            cbar = fig.colorbar(era5_mesh, ax=axes[0, 0], orientation="horizontal", fraction=0.055, pad=0.055)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["no event", "event"])
            cbar.ax.tick_params(labelsize=7)
        if prob_mesh is not None:
            cbar = fig.colorbar(prob_mesh, ax=axes[:, 1], orientation="horizontal", fraction=0.045, pad=0.055)
            cbar.set_label("ACE2 member fraction with event somewhere in window", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        if overlap_mesh is not None and len(chosen_members) > 0:
            cbar = fig.colorbar(overlap_mesh, ax=axes[:, 2:], orientation="horizontal", fraction=0.035, pad=0.055)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(OVERLAP_LABELS)
            cbar.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"Workflow 5 temporal support for ERA5 event day {case.date.date()} (r={radius})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig.text(
            0.5,
            -0.004,
            "P_tau counts each member once per pixel if it has an event anywhere inside the temporal window.",
            ha="center",
            va="top",
            fontsize=8.5,
        )
        out = paths.figures / f"workflow5_temporal_support_sample_{case.date.date()}_{metric_slug(args)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def run_figures(cube: MaskCube | None, paths: SupportPaths, args: argparse.Namespace) -> list[str]:
    selected = selected_figure_names(args)
    daily_csv = daily_metrics_path(paths, args)
    weekly_csv = weekly_summary_path(paths, args)
    if not daily_csv.exists() or not weekly_csv.exists():
        raise FileNotFoundError("Workflow 5 metric tables are missing. Run --stages metrics first.")
    daily_df = pd.read_csv(daily_csv)
    weekly_df = pd.read_csv(weekly_csv)
    outputs: list[Path] = []
    if "support-sweep" in selected:
        outputs.append(plot_support_sweep(daily_df, paths, args))
    if "timeseries" in selected:
        outputs.append(plot_timeseries(weekly_df, paths, args))
    if "spatial-timeseries" in selected:
        outputs.append(plot_spatial_timeseries(weekly_df, paths, args))
    if "effect-distributions" in selected:
        outputs.append(plot_effect_distributions(daily_df, paths, args))
    if "sample-panels" in selected:
        if cube is None:
            years = parse_years(args.years)
            daily_cases = all_daily_cases(years, args)
            cube = load_mask_cube(daily_cases, daily_first_input_paths(args.daily_first_root), args)
        outputs.extend(plot_sample_panels(cube, paths, args))
    return [str(path) for path in outputs]


def write_manifest(
    paths: SupportPaths,
    args: argparse.Namespace,
    years: list[int],
    n_cases: int,
    table_outputs: dict[str, str],
    figure_outputs: list[str],
) -> None:
    manifest = {
        "workflow": "Workflow 5 temporal-neighborhood ensemble support",
        "source": "Workflow 3 daily masks; no Tmax, threshold, smoothing, or mask computation is rerun.",
        "definition": {
            "member_vote": (
                "S_{m,r,tau}(x,t)=I[member m has an event within +/- tau days and within r grid pixels of x]. "
                "Each member contributes at most one vote per pixel."
            ),
            "event_probability": "P_{r,tau}(x,t)=mean_m S_{m,r,tau}(x,t).",
            "member_day_probability": (
                "P_member_day(x,t)=sum_m sum_{t' in window} I[event_{m,t'}(x)] / "
                "sum_m sum_{t' in window} I[valid_{m,t'}(x)]. This normalizes by the actual member-day slots."
            ),
            "soft_iou": "sum_x area_x O(x,t)P(x,t) / sum_x area_x [O(x,t)+P(x,t)-O(x,t)P(x,t)].",
            "soft_recall": "mean ACE2 support probability over ERA5 event pixels.",
            "soft_precision": "fraction of ACE2 probability mass that falls inside ERA5 event pixels.",
            "soft_f1": "harmonic mean of soft precision and soft recall.",
            "thresholded_metrics": "Binary IoU/F1 after thresholding P_{r,tau} at configured probability thresholds.",
            "paper_metrics": (
                "Optional paper-metrics stage computes tail-weighted CRPS from continuous Workflow 3 daily fields, "
                "plus distance-binned chi_ij(u) and ARE(u) dependence summaries."
            ),
        },
        "years": f"{years[0]}:{years[-1]}",
        "n_daily_cases": int(n_cases),
        "temporal_neighborhood_days": parse_int_list(args.temporal_neighborhood_days, name="temporal neighborhood days"),
        "spatial_radii_pixels": parse_neighborhood_pixels(args.spatial_radii_pixels),
        "probability_thresholds": parse_float_list(args.probability_thresholds, name="probability thresholds"),
        "paper_sources": list(args.paper_sources),
        "paper_quantiles": paper_quantiles(args),
        "twcrps_tail_quantile": float(args.twcrps_tail_quantile),
        "twcrps_tail_threshold_source": args.twcrps_tail_threshold_source,
        "daily_first_root": str(args.daily_first_root),
        "out_root": str(paths.out_root),
        "tables": table_outputs,
        "figures": figure_outputs,
    }
    manifest_path(paths, args).write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.workflow_mode == "marked-events":
        from marked_event_core import ensure_disjoint_periods
        from marked_event_workflow import (
            cases_for_years,
            input_paths,
            run_marked_event_workflow,
            validate_field_inputs,
        )

        fit_years = parse_years(args.fit_years)
        validation_years = parse_years(args.validation_years)
        test_years = parse_years(args.test_years)
        ensure_disjoint_periods(fit_years, validation_years, test_years)
        args.paper_quantiles_parsed = paper_quantiles(args)
        args.match_radii_km_parsed = parse_unbounded_float_list(
            args.match_radii_km,
            name="match radii (km)",
        )
        if any(value < 0 for value in args.match_radii_km_parsed):
            raise ValueError("--match-radii-km values must be nonnegative.")
        args.match_tolerances_days_parsed = parse_int_list(
            args.match_tolerances_days,
            name="match temporal tolerances",
        )
        args.chi_distance_bins_km_parsed = chi_distance_edges(args)
        args.validation_connectivities_parsed = parse_int_list(
            args.validation_connectivities or [str(args.connectivity)],
            name="validation connectivities",
        )
        if any(value not in (6, 18, 26) for value in args.validation_connectivities_parsed):
            raise ValueError("--validation-connectivities values must be 6, 18, or 26.")
        args.validation_min_duration_days_parsed = parse_int_list(
            args.validation_min_duration_days or [str(args.min_duration_days)],
            name="validation minimum durations",
        )
        args.validation_min_event_area_km2_parsed = parse_unbounded_float_list(
            args.validation_min_event_area_km2 or [str(args.min_event_area_km2)],
            name="validation minimum event areas",
        )
        if any(value < 0 for value in args.validation_min_event_area_km2_parsed):
            raise ValueError("--validation-min-event-area-km2 values must be nonnegative.")
        if float(args.process_grid_degrees) <= 0:
            raise ValueError("--process-grid-degrees must be positive.")
        if not any(
            np.isclose(value, float(args.primary_match_radius_km))
            for value in args.match_radii_km_parsed
        ):
            raise ValueError("--primary-match-radius-km must be included in --match-radii-km.")
        if int(args.primary_match_tolerance_days) not in args.match_tolerances_days_parsed:
            raise ValueError(
                "--primary-match-tolerance-days must be included in --match-tolerances-days."
            )
        if not any(np.isclose(float(args.local_are_quantile), value) for value in args.paper_quantiles_parsed):
            raise ValueError("--local-are-quantile must be included in --paper-quantiles.")
        if args.validate_only:
            all_years = sorted(set(fit_years + validation_years + test_years))
            cases = cases_for_years(all_years, args, apply_debug_limit=True)
            validate_field_inputs(cases, input_paths(args.daily_first_root), args)
            print(
                f"Workflow 5 marked-event input validation passed: "
                f"{len(cases)} daily cases under {args.daily_first_root}"
            )
            return
        artifacts = run_marked_event_workflow(args)
        print("Workflow 5 full marked-event evaluation complete.")
        print(f"Manifest: {artifacts['manifest']}")
        return

    comm, rank, size = mpi_info()
    years = parse_years(args.years)
    daily_cases = all_daily_cases(years, args)
    daily_first_paths = daily_first_input_paths(args.daily_first_root)
    input_counts = validate_daily_first_artifacts(daily_cases, daily_first_paths, args)
    if args.validate_only:
        rank_print(rank, f"Workflow 3 input validation passed: {args.daily_first_root}")
        for kind, count in input_counts.items():
            rank_print(rank, f"  {kind}: {count} files")
        return
    paths = make_paths(args.out_root)

    rank_print(rank, f"MPI ranks: {size}")
    rank_print(rank, f"Workflow 3 daily-mask root: {args.daily_first_root}")
    rank_print(rank, f"Workflow 5 output root: {args.out_root}")
    rank_print(rank, f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    rank_print(rank, f"Daily cases: {len(daily_cases)}")
    rank_print(rank, f"Temporal neighborhoods: {parse_int_list(args.temporal_neighborhood_days, name='temporal neighborhood days')} days")
    rank_print(rank, f"Spatial radii: {parse_neighborhood_pixels(args.spatial_radii_pixels)} grid pixels")
    if "paper-metrics" in args.stages:
        rank_print(rank, f"Paper metric sources: {', '.join(args.paper_sources)}")
        rank_print(rank, f"Paper metric quantiles: {paper_quantiles(args)}")

    cube: MaskCube | None = None
    table_outputs: dict[str, str] = {}
    if "metrics" in args.stages:
        cube = load_mask_cube(daily_cases, daily_first_paths, args)
        local_indices = rank_subset(list(range(len(daily_cases))), rank, size)
        part = run_metric_part(cube, local_indices, paths, args, rank)
        parts = [daily_part_path(paths, part_rank, args) for part_rank in range(size)]
        if rank == 0 and part not in parts:
            parts.append(part)
    barrier(comm)

    if "metrics" in args.stages and rank == 0:
        parts = [daily_part_path(paths, part_rank, args) for part_rank in range(size)]
        table_outputs = combine_and_summarize(parts, paths, args)
    barrier(comm)

    if "paper-metrics" in args.stages:
        local_indices = rank_subset(list(range(len(daily_cases))), rank, size)
        part = run_twcrps_part(daily_cases, local_indices, daily_first_paths, paths, args, rank)
        parts = [twcrps_part_path(paths, part_rank, args) for part_rank in range(size)]
        if rank == 0 and part not in parts:
            parts.append(part)
    barrier(comm)

    if "paper-metrics" in args.stages and rank == 0:
        parts = [twcrps_part_path(paths, part_rank, args) for part_rank in range(size)]
        table_outputs.update(combine_twcrps_parts(parts, paths, args))
        table_outputs.update(run_dependence_paper_metrics(daily_cases, daily_first_paths, paths, args))
    barrier(comm)

    figure_outputs: list[str] = []
    if "figures" in args.stages and rank == 0:
        figure_outputs = run_figures(cube, paths, args)
    barrier(comm)

    if rank == 0:
        if not table_outputs:
            candidates = {
                "daily_temporal_support_metrics": daily_metrics_path(paths, args),
                "weekly_temporal_support_summary": weekly_summary_path(paths, args),
                "overall_temporal_support_summary": overall_summary_path(paths, args),
                "paper_twcrps_daily": twcrps_daily_path(paths, args),
                "paper_twcrps_summary": twcrps_summary_path(paths, args),
                "paper_chi_distance_summary": chi_summary_path(paths, args),
                "paper_are_reference_metrics": are_reference_path(paths, args),
                "paper_are_summary": are_summary_path(paths, args),
            }
            table_outputs = {name: str(path) for name, path in candidates.items() if path.exists()}
        write_manifest(paths, args, years, len(daily_cases), table_outputs, figure_outputs)
        rank_print(rank, "Workflow 5 temporal-support diagnostics complete.")
        rank_print(rank, f"Manifest: {manifest_path(paths, args)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        main()
