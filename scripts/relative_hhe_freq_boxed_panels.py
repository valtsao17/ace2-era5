#!/usr/bin/env python3
"""Relative-HHE climatological frequency panels with the rank-correlation box."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import _plain_map_axes, CONUS_LAT_SLICE, CONUS_LON_SLICE  # noqa: E402

FREQ_NC = PROJECT_ROOT / "outputs/lag_may/relative_hhe_jja_sliding7d/jja_seasonal_freqs.nc"
SUMMARY_JSON = PROJECT_ROOT / "outputs/lag_may/relative_hhe_sst_rankcorr_box/relative_hhe_sst_rankcorr_box_summary.json"
FIG_DIR = PROJECT_ROOT / "outputs/lag_may/relative_hhe_sst_rankcorr_box/figures"
OUT_PNG = FIG_DIR / "fig2_relative_hhe_freq_panels.png"
CMAP = LinearSegmentedColormap.from_list(
    "wred", ["#ffffff", "#fee0d2", "#fc9272", "#ef3b2c", "#a50f15"]
)


def box_from_summary() -> tuple[float, float, float, float] | None:
    if not SUMMARY_JSON.exists():
        return None
    summary = json.loads(SUMMARY_JSON.read_text())
    raw = summary.get("relative_hhe_rankcorr_box")
    if not raw:
        return None
    lat_s, lat_n = [float(x) for x in raw["latN"]]
    lon_w, lon_e = [float(x) for x in raw["lonW"]]
    return (lat_s, lat_n, 360.0 - lon_e, 360.0 - lon_w)


def _panel(ax, field_pct, lat, lon, title, vmax, box=None):
    lon_plot = lon - 360.0 if float(lon.mean()) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    ax.set_facecolor("#eef1f4")
    im = ax.imshow(field_pct, origin="lower", extent=extent, aspect="equal",
                   vmin=0.0, vmax=vmax, cmap=CMAP, interpolation="nearest", zorder=1)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    if box is not None:
        la0, la1, lo0, lo1 = box
        x0 = lo0 - 360.0 if lo0 > 180 else lo0
        x1 = lo1 - 360.0 if lo1 > 180 else lo1
        ax.add_patch(Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="#00b050", linewidth=2.6, zorder=5))
    ax.set_title(title, fontsize=11)
    return im


def main() -> None:
    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        era5 = ds["era5_freq"].values.astype(np.float32)
        ace2 = ds["ace2_freq"].values.astype(np.float32)
        years = [int(y) for y in ds["year"].values]
        ace2_hi_inputs = ds.attrs.get("ace2_hi_inputs", "unknown")

    e = np.nanmean(era5, axis=0)[CONUS_LAT_SLICE, CONUS_LON_SLICE] * 100.0
    a = np.nanmean(ace2, axis=0)[CONUS_LAT_SLICE, CONUS_LON_SLICE] * 100.0
    latc = lat[CONUS_LAT_SLICE]
    lonc = lon[CONUS_LON_SLICE]
    box = box_from_summary()
    if box is not None:
        la0, la1, lo0, lo1 = box
        score = float(np.nanmean(e[np.ix_((latc >= la0) & (latc <= la1), (lonc >= lo0) & (lonc <= lo1))]))
        print(f"Relative-HHE frequency box lat {la0:.1f}-{la1:.1f}N "
              f"lon {360.0-lo1:.1f}-{360.0-lo0:.1f}W mean={score:.2f}%", flush=True)

    vmax = float(np.nanpercentile(np.concatenate([e.ravel(), a.ravel()]), 99.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    _panel(axes[0], e, latc, lonc, "ERA5", vmax, box=box)
    im = _panel(axes[1], a, latc, lonc, "ACE2", vmax, box=None)
    fig.suptitle(
        "Relative humid heat extreme (JJA HI > 90th-pct, 15-day window), seasonal frequency\n"
        f"{years[0]}-{years[-1]} | ACE2 HI inputs: {ace2_hi_inputs}",
        fontsize=13,
        y=1.02,
    )
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.12, shrink=0.5, aspect=40)
    cbar.set_label("% of JJA days above relative HI threshold")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
