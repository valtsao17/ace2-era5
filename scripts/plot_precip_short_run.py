#!/usr/bin/env python3
"""Quick visualization for the short ACE2 precipitation inference.

Reads PRATEsfc from outputs/lag_may/runs_precip_short/<year>/member_*/ and
plots CONUS mean precipitation rate in mm/day for the three-year all-member run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    CONUS_LAT_SLICE,
    CONUS_LON_SLICE,
    _plain_map_axes,
)
from seasonal_jja_skill import cos_lat_mean, load_land_mask, domain_scores_label  # noqa: E402

RUN_DIR = PROJECT_ROOT / "outputs/lag_may/runs_precip_short"
FIG_DIR = RUN_DIR / "figures"
OUT_PNG = FIG_DIR / "precip_short_mean_conus.png"
OUT_JSON = FIG_DIR / "precip_short_mean_conus_summary.json"
SECONDS_PER_DAY = 86400.0


def _accumulate_map(acc, arr):
    good = np.isfinite(arr)
    acc["sum"] += np.nansum(np.where(good, arr, 0.0), axis=(0, 1))
    acc["count"] += good.sum(axis=(0, 1))


def _finish(acc):
    return np.divide(
        acc["sum"],
        acc["count"],
        out=np.full_like(acc["sum"], np.nan, dtype=np.float64),
        where=acc["count"] > 0,
    ).astype(np.float32)


def load_precip_means():
    files = sorted(RUN_DIR.glob("*/member_*/autoregressive_predictions.nc"))
    if not files:
        raise FileNotFoundError(f"No precipitation predictions found under {RUN_DIR}")

    lat = lon = None
    total = None
    by_year = {}
    file_counts = {}

    for path in files:
        year = path.parts[-3]
        with xr.open_dataset(path) as ds:
            if "PRATEsfc" not in ds:
                continue
            da = ds["PRATEsfc"].isel(lat=CONUS_LAT_SLICE, lon=CONUS_LON_SLICE)
            if lat is None:
                lat = da["lat"].values.astype(np.float32)
                lon = da["lon"].values.astype(np.float32)
                shape = (len(lat), len(lon))
                total = {
                    "sum": np.zeros(shape, dtype=np.float64),
                    "count": np.zeros(shape, dtype=np.int64),
                }
            arr = (da.values.astype(np.float32) * SECONDS_PER_DAY)

        if year not in by_year:
            by_year[year] = {
                "sum": np.zeros_like(total["sum"]),
                "count": np.zeros_like(total["count"]),
            }
            file_counts[year] = 0
        _accumulate_map(total, arr)
        _accumulate_map(by_year[year], arr)
        file_counts[year] += 1

    if total is None or lat is None or lon is None:
        raise RuntimeError("Found files, but none contained PRATEsfc")

    years = sorted(by_year)
    maps = {"all": _finish(total)}
    maps.update({year: _finish(by_year[year]) for year in years})
    return maps, lat, lon, file_counts


def _lon_for_plot(lon):
    return lon - 360.0 if float(np.nanmean(lon)) > 180.0 else lon


def _draw_panel(ax, field, lat, lon, title, cmap, vmin, vmax, label_text,
                show_xaxis=True):
    lon_plot = _lon_for_plot(lon)
    extent = [
        float(lon_plot[0]) - 0.5,
        float(lon_plot[-1]) + 0.5,
        float(lat[0]) - 0.5,
        float(lat[-1]) + 0.5,
    ]
    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    _plain_map_axes(ax, lon, lat, pad=0.0)
    if not show_xaxis:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(title, fontsize=10, pad=3)
    ax.legend(
        [Line2D([], [], linestyle="none")],
        [label_text],
        loc="lower left",
        fontsize=7.5,
        handlelength=0,
        handletextpad=0,
        framealpha=0.95,
        borderpad=0.4,
    ).set_zorder(7)
    return im


def render(maps, lat, lon, file_counts):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    years = sorted(file_counts)
    panel_keys = ["all", *years]
    panel_titles = [
        f"All years ({sum(file_counts.values())} members)",
        *[f"{year} ({file_counts[year]} members)" for year in years],
    ]

    vals = np.concatenate([maps[key][np.isfinite(maps[key])] for key in panel_keys])
    vmax = float(np.nanpercentile(vals, 99.0))
    vmax = max(vmax, 1.0)

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    land = load_land_mask(lat, lon)

    fig = plt.figure(figsize=(13.6, 7.0), facecolor="white")
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.045],
        left=0.045,
        right=0.945,
        bottom=0.075,
        top=0.89,
        wspace=0.055,
        hspace=0.24,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    im = None
    for idx, (ax, key, title) in enumerate(zip(axes, panel_keys, panel_titles)):
        label = domain_scores_label("mean", maps[key], lat, land, fmt="{:.2f}") + " mm/day"
        im = _draw_panel(
            ax, maps[key], lat, lon, title, cmap, 0.0, vmax, label,
            show_xaxis=idx >= 2,
        )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean precipitation rate (mm/day)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "ACE2 short precipitation inference: one-month lead PRATEsfc",
        fontsize=13,
        y=0.955,
    )
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "input_dir": str(RUN_DIR),
        "output_png": str(OUT_PNG),
        "variable": "PRATEsfc",
        "conversion": "kg/m**2/s * 86400 = mm/day",
        "years": years,
        "members_per_year": {k: int(v) for k, v in file_counts.items()},
        "conus_mean_mm_day": {
            key: round(float(cos_lat_mean(maps[key], lat)), 4) for key in panel_keys
        },
        "colorbar_vmax_mm_day": round(vmax, 4),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PNG}", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


def main():
    maps, lat, lon, file_counts = load_precip_means()
    render(maps, lat, lon, file_counts)


if __name__ == "__main__":
    main()
