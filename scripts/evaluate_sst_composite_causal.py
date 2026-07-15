#!/usr/bin/env python3
"""Evaluate paired high/low SST ACE2 runs using JJA HHE frequency.

Inference is across base years (independent atmospheric states), not across the
25 lag members. Members are averaged within each base year before the paired
t-test, sign-flip test, and bootstrap confidence interval are computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from hhe_ace2 import HI_THRESH, member_jja_hi  # noqa: E402
from sst_teleconnection_jja_sliding7d import _draw_coast_and_borders  # noqa: E402


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":", 1))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_members(spec: str) -> list[int]:
    return list(range(25)) if spec == "all" else [int(x) for x in spec.split(",")]


def lag_times(year: int, month: int, day: int) -> list[datetime]:
    center = datetime(year, month, day)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(25)]


def weighted_region(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    box: list[float], land: np.ndarray | None) -> float:
    south, north, west, east = box
    lon360 = lon % 360.0
    lm = (lat[:, None] >= south) & (lat[:, None] <= north)
    if west % 360 <= east % 360:
        xm = (lon360[None, :] >= west % 360) & (lon360[None, :] <= east % 360)
    else:
        xm = (lon360[None, :] >= west % 360) | (lon360[None, :] <= east % 360)
    mask = lm & xm & np.isfinite(field)
    if land is not None:
        mask &= land
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    return float(np.nansum(field * w) / np.sum(w))


def load_land(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        da = ds["land_fraction"]
        if "time" in da.dims:
            da = da.isel(time=0)
        return np.asarray(da) > 0.5


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    n = len(values)
    if n <= 20:
        codes = np.arange(1 << n, dtype=np.uint64)[:, None]
        bits = ((codes >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        means = ((2 * bits - 1) * values).mean(axis=1)
    else:
        signs = rng.choice((-1.0, 1.0), size=(draws, n))
        means = (signs * values).mean(axis=1)
    return float((np.count_nonzero(np.abs(means) >= observed) + 1) / (len(means) + 1))


def json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def render_figure(
    out_path: Path,
    high: np.ndarray,
    low: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    land: np.ndarray | None,
    box: list[float],
    years: list[int],
    year_effects: np.ndarray,
    case_years: list[int],
    case_members: list[int],
    case_effects: list[float],
    high_label: str,
    low_label: str,
) -> None:
    """Render treatment means, causal contrast, and paired regional effects."""
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    order = np.argsort(lon180)
    x = lon180[order]
    high_pct = 100.0 * high[:, order]
    low_pct = 100.0 * low[:, order]
    diff_pct = high_pct - low_pct
    if land is not None:
        land_plot = land[:, order]
        high_pct = np.where(land_plot, high_pct, np.nan)
        low_pct = np.where(land_plot, low_pct, np.nan)
        diff_pct = np.where(land_plot, diff_pct, np.nan)

    finite_freq = np.concatenate([high_pct[np.isfinite(high_pct)], low_pct[np.isfinite(low_pct)]])
    freq_vmax = float(np.nanpercentile(finite_freq, 99)) if finite_freq.size else 1.0
    freq_vmax = max(freq_vmax, 1.0)
    finite_diff = np.abs(diff_pct[np.isfinite(diff_pct)])
    diff_vmax = float(np.nanpercentile(finite_diff, 99)) if finite_diff.size else 1.0
    diff_vmax = max(diff_vmax, 0.25)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    extent_x, extent_y = (-130.0, -60.0), (20.0, 52.0)
    south, north, west, east = box
    west180 = west - 360.0 if west > 180.0 else west
    east180 = east - 360.0 if east > 180.0 else east

    def map_panel(ax, field, title, cmap, vmin, vmax, cbar_label):
        mesh = ax.pcolormesh(x, lat, field, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim(*extent_x); ax.set_ylim(*extent_y)
        _draw_coast_and_borders(ax, extent_x, extent_y)
        ax.add_patch(Rectangle(
            (west180, south), east180 - west180, north - south,
            fill=False, edgecolor="#00a651", linewidth=2.0, zorder=6,
        ))
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.84, label=cbar_label)

    map_panel(axes[0, 0], high_pct, f"a  {high_label} arm", "YlOrRd", 0, freq_vmax,
              "HHE frequency (% of JJA days)")
    map_panel(axes[0, 1], low_pct, f"b  {low_label} arm", "YlOrRd", 0, freq_vmax,
              "HHE frequency (% of JJA days)")
    map_panel(axes[1, 0], diff_pct, f"c  Causal contrast: {high_label} minus {low_label}", "RdBu_r",
              -diff_vmax, diff_vmax, "HHE-frequency change (percentage points)")

    ax = axes[1, 1]
    if len(years) == 1:
        vals = 100.0 * np.asarray(case_effects)
        labels = [f"m{m:02d}" for m in case_members]
        title = f"d  Paired member effects, base year {years[0]}"
        xlabel = "lag member (descriptive, not independent replicates)"
    else:
        vals = 100.0 * year_effects
        labels = [str(y) for y in years]
        title = "d  Regional effect by independent base year"
        xlabel = "base year"
    colors = np.where(vals >= 0, "#c43c39", "#3478b8")
    ax.bar(np.arange(len(vals)), vals, color=colors, width=0.78)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(float(np.mean(vals)), color="#6a3d9a", linewidth=1.8, linestyle="--",
               label=f"mean = {np.mean(vals):.2f} pp")
    ax.set_xticks(np.arange(len(vals)), labels, rotation=60 if len(vals) > 12 else 0)
    ax.set_ylabel("regional high-minus-low effect (percentage points)")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"ACE2 response to prescribed SST: {high_label} − {low_label} | HI ≥ {HI_THRESH:g}°F | "
        f"{len(case_effects)} paired member-year cases",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--high-runs", required=True)
    p.add_argument("--low-runs", required=True)
    p.add_argument("--years", required=True)
    p.add_argument("--members", default="all")
    p.add_argument("--init-month", type=int, default=5)
    p.add_argument("--init-day", type=int, default=1)
    p.add_argument("--high-label", default="High-HHE-like SST")
    p.add_argument("--low-label", default="Low-HHE-like SST")
    p.add_argument("--index-box", nargs=4, type=float, default=[23, 38, 260, 283])
    p.add_argument("--land-mask-forcing", default=None,
                   help="A forcing_YEAR.nc file used to restrict the regional index to land")
    p.add_argument("--min-members", type=int, default=20)
    p.add_argument("--bootstrap-draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--out", required=True)
    p.add_argument("--figure", default=None,
                   help="PNG path; default is --out with a .png suffix")
    args = p.parse_args()

    years, members = parse_years(args.years), parse_members(args.members)
    high_root, low_root = Path(args.high_runs), Path(args.low_runs)
    land = load_land(Path(args.land_mask_forcing)) if args.land_mask_forcing else None
    high_sum = low_sum = None
    case_count = 0
    effects_by_year: list[float] = []
    kept_years: list[int] = []
    members_by_year: list[int] = []
    case_years: list[int] = []
    case_members: list[int] = []
    case_effects: list[float] = []
    lat = lon = None

    for year in years:
        year_high, year_low, regional, paired_member_ids = [], [], [], []
        times = lag_times(year, args.init_month, args.init_day)
        for member in members:
            hp = high_root / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc"
            lp = low_root / str(year) / f"member_{member:02d}" / "autoregressive_predictions.nc"
            if not hp.exists() or not lp.exists():
                continue
            h = member_jja_hi(hp, times[member])
            l = member_jja_hi(lp, times[member])
            if h is None or l is None:
                continue
            hf = np.asarray((h >= HI_THRESH).mean("time"), dtype=np.float32)
            lf = np.asarray((l >= HI_THRESH).mean("time"), dtype=np.float32)
            if lat is None:
                lat, lon = np.asarray(h["lat"], dtype=float), np.asarray(h["lon"], dtype=float)
                high_sum = np.zeros_like(hf, dtype=np.float64)
                low_sum = np.zeros_like(lf, dtype=np.float64)
                if land is not None and land.shape != hf.shape:
                    raise ValueError(f"Land mask shape {land.shape} != output grid {hf.shape}")
            year_high.append(hf)
            year_low.append(lf)
            regional.append(weighted_region(hf - lf, lat, lon, args.index_box, land))
            paired_member_ids.append(member)
        if len(regional) < args.min_members:
            print(f"{year}: only {len(regional)} paired members; excluded from inference", flush=True)
            continue
        for hf, lf in zip(year_high, year_low):
            high_sum += hf
            low_sum += lf
            case_count += 1
        case_years.extend([year] * len(regional))
        case_members.extend(paired_member_ids)
        case_effects.extend(regional)
        effects_by_year.append(float(np.mean(regional)))
        kept_years.append(year)
        members_by_year.append(len(regional))
        print(f"{year}: n={len(regional)} regional high-low={100*np.mean(regional):.3f} percentage points", flush=True)

    if not kept_years or high_sum is None or case_count == 0:
        raise RuntimeError("No complete paired cases found")
    effects = np.asarray(effects_by_year)
    rng = np.random.default_rng(args.seed)
    if len(effects) >= 2:
        t_result = stats.ttest_1samp(effects, 0.0)
        boot = rng.choice(effects, size=(args.bootstrap_draws, len(effects)), replace=True).mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
        flip_p = sign_flip_p(effects, rng, args.bootstrap_draws)
        t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
    else:
        t_stat = t_p = flip_p = float("nan")
        ci = np.array([np.nan, np.nan])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_high = (high_sum / case_count).astype(np.float32)
    mean_low = (low_sum / case_count).astype(np.float32)
    ds = xr.Dataset(
        {
            "high_hhe_frequency": (("lat", "lon"), mean_high),
            "low_hhe_frequency": (("lat", "lon"), mean_low),
            "high_minus_low_hhe_frequency": (("lat", "lon"), mean_high - mean_low),
            "regional_effect_by_year": (("year",), effects.astype(np.float32)),
            "n_paired_members": (("year",), np.asarray(members_by_year, dtype=np.int16)),
            "regional_effect_by_case": (("case",), np.asarray(case_effects, dtype=np.float32)),
            "case_year": (("case",), np.asarray(case_years, dtype=np.int16)),
            "case_member": (("case",), np.asarray(case_members, dtype=np.int16)),
        },
        coords={"lat": lat, "lon": lon, "year": kept_years},
        attrs={
            "frequency_units": "fraction of JJA days",
            "regional_index_box": json.dumps(args.index_box),
            "inference_unit": "base year; lag members averaged within year",
            "n_map_member_year_pairs": case_count,
            "high_arm_label": args.high_label,
            "low_arm_label": args.low_label,
            "contrast": f"{args.high_label} minus {args.low_label}",
        },
    )
    ds.to_netcdf(out_path)
    figure_path = Path(args.figure) if args.figure else out_path.with_suffix(".png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    render_figure(
        figure_path, mean_high, mean_low, lat, lon, land, args.index_box,
        kept_years, effects, case_years, case_members, case_effects,
        args.high_label, args.low_label,
    )
    summary = {
        "years": kept_years,
        "n_years": len(kept_years),
        "n_map_member_year_pairs": case_count,
        "regional_high_minus_low_percentage_points": 100 * float(effects.mean()),
        "regional_contrast_percentage_points": 100 * float(effects.mean()),
        "bootstrap_95pct_ci_percentage_points": [json_number(x) for x in 100 * ci],
        "paired_year_t_statistic": json_number(t_stat),
        "paired_year_t_p_value": json_number(t_p),
        "year_block_sign_flip_p_value": json_number(flip_p),
        "index_box": args.index_box,
        "initialization": f"{args.init_month:02d}-{args.init_day:02d}",
        "contrast": f"{args.high_label} minus {args.low_label}",
        "figure": str(figure_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
