#!/usr/bin/env python3
"""2x2 precipitation metrics panel from the short ACE2 PRATEsfc run."""
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
from seasonal_jja_skill import load_land_mask, domain_scores_label  # noqa: E402

RUN_DIR = PROJECT_ROOT / "outputs/lag_may/runs_precip_short"
FIG_DIR = RUN_DIR / "figures"
OUT_PNG = FIG_DIR / "precip_short_metrics_panel.png"
OUT_JSON = FIG_DIR / "precip_short_metrics_panel_summary.json"
SECONDS_PER_DAY = 86400.0
WET_THRESHOLD_MM_DAY = 1.0


def _lon_for_plot(lon):
    return lon - 360.0 if float(np.nanmean(lon)) > 180.0 else lon


def load_metrics():
    files = sorted(RUN_DIR.glob("*/member_*/autoregressive_predictions.nc"))
    if not files:
        raise FileNotFoundError(f"No autoregressive_predictions.nc files under {RUN_DIR}")

    chunks = []
    run_means = []
    lat = lon = None

    for path in files:
        with xr.open_dataset(path) as ds:
            if "PRATEsfc" not in ds:
                continue
            da = ds["PRATEsfc"].isel(sample=0, lat=CONUS_LAT_SLICE, lon=CONUS_LON_SLICE)
            if lat is None:
                lat = da["lat"].values.astype(np.float32)
                lon = da["lon"].values.astype(np.float32)
            arr = (da.values.astype(np.float32) * SECONDS_PER_DAY)
        chunks.append(arr)
        run_means.append(np.nanmean(arr, axis=0))

    if not chunks:
        raise RuntimeError("Prediction files exist, but none contained PRATEsfc")

    data = np.concatenate(chunks, axis=0)
    mean_rate = np.nanmean(data, axis=0).astype(np.float32)
    p95_rate = np.nanpercentile(data, 95, axis=0).astype(np.float32)
    wet_freq = (np.nanmean(data >= WET_THRESHOLD_MM_DAY, axis=0) * 100.0).astype(np.float32)
    spread = np.nanstd(np.stack(run_means, axis=0), axis=0).astype(np.float32)
    return {
        "mean": mean_rate,
        "p95": p95_rate,
        "wet_freq": wet_freq,
        "spread": spread,
    }, lat, lon, len(chunks)


def finite_vmax(field, pct=99.0, floor=1.0):
    vals = field[np.isfinite(field)]
    if vals.size == 0:
        return floor
    return max(float(np.nanpercentile(vals, pct)), floor)


def draw_panel(ax, cax, fig, field, lat, lon, title, cmap, vmin, vmax,
               cbar_label, corner_label, show_xaxis):
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
        [corner_label],
        loc="lower left",
        fontsize=7.5,
        handlelength=0,
        handletextpad=0,
        framealpha=0.95,
        borderpad=0.4,
    ).set_zorder(7)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def render(metrics, lat, lon, n_files):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    land = load_land_mask(lat, lon)

    rate_cmap = plt.get_cmap("YlGnBu").copy()
    rate_cmap.set_bad("white")
    freq_cmap = plt.get_cmap("PuBuGn").copy()
    freq_cmap.set_bad("white")
    spread_cmap = plt.get_cmap("magma").copy()
    spread_cmap.set_bad("white")

    panels = [
        {
            "key": "mean",
            "title": "Mean rate",
            "cmap": rate_cmap,
            "vmin": 0.0,
            "vmax": finite_vmax(metrics["mean"], floor=1.0),
            "label": "mm/day",
            "legend": domain_scores_label("domain", metrics["mean"], lat, land, fmt="{:.2f}") + " mm/day",
        },
        {
            "key": "p95",
            "title": "Heavy-rate percentile",
            "cmap": rate_cmap,
            "vmin": 0.0,
            "vmax": finite_vmax(metrics["p95"], floor=2.0),
            "label": "mm/day",
            "legend": domain_scores_label("domain", metrics["p95"], lat, land, fmt="{:.2f}") + " mm/day",
        },
        {
            "key": "wet_freq",
            "title": "Wet-step frequency",
            "cmap": freq_cmap,
            "vmin": 0.0,
            "vmax": finite_vmax(metrics["wet_freq"], floor=10.0),
            "label": f"% steps >= {WET_THRESHOLD_MM_DAY:g} mm/day",
            "legend": domain_scores_label("domain", metrics["wet_freq"], lat, land, fmt="{:.1f}") + "%",
        },
        {
            "key": "spread",
            "title": "Run-to-run spread",
            "cmap": spread_cmap,
            "vmin": 0.0,
            "vmax": finite_vmax(metrics["spread"], floor=0.2),
            "label": "mm/day",
            "legend": domain_scores_label("domain", metrics["spread"], lat, land, fmt="{:.2f}") + " mm/day",
        },
    ]

    fig = plt.figure(figsize=(14.4, 7.0), facecolor="white")
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 0.035, 1.0, 0.035],
        left=0.045,
        right=0.965,
        bottom=0.075,
        top=0.89,
        wspace=0.075,
        hspace=0.24,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3]),
             fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 3])]

    for idx, (ax, cax, spec) in enumerate(zip(axes, caxes, panels)):
        draw_panel(
            ax, cax, fig, metrics[spec["key"]], lat, lon,
            spec["title"], spec["cmap"], spec["vmin"], spec["vmax"],
            spec["label"], spec["legend"], show_xaxis=idx >= 2,
        )

    fig.suptitle("ACE2 short precipitation inference metrics", fontsize=13, y=0.955)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "input_dir": str(RUN_DIR),
        "n_prediction_files": int(n_files),
        "variable": "PRATEsfc",
        "conversion": "kg/m**2/s * 86400 = mm/day",
        "wet_threshold_mm_day": WET_THRESHOLD_MM_DAY,
        "metrics": {
            name: {
                "domain_mean": float(np.nanmean(field)),
                "finite_min": float(np.nanmin(field)),
                "finite_max": float(np.nanmax(field)),
            }
            for name, field in metrics.items()
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PNG}", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


def main():
    metrics, lat, lon, n_files = load_metrics()
    render(metrics, lat, lon, n_files)


if __name__ == "__main__":
    main()
