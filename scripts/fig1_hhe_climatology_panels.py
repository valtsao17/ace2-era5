#!/usr/bin/env python3
"""Figure 1 style climatology panels for HHE daily ingredients.

Panels:
  a-c ERA5 JJA climatology of daily RHmin, Tmax, and HI.
  d-f model/ACE2 climatology of daily RHmin, Tmax, and HI.

The model RHmin/Tmax panels use raw hindcast daily values. The model HI panel
follows the paper caption: HI is computed from bias-corrected model RHmin and
Tmax using the additive bias fields from hhe_ace2_biascorr.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from heat_index_era5 import heat_index  # noqa: E402
from seasonal_jja_skill import load_land_mask  # noqa: E402
from sst_teleconnection_jja_sliding7d import _draw_conus_borders  # noqa: E402

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
ERA5_MONTHLY = HHE_DIR / "monthly_cache"
ACE2_CACHE = HHE_DIR / "ace2_daily_cache"
BIAS_NC = HHE_DIR / "bias_fields.nc"
OUT_NC = HHE_DIR / "fig1_hhe_climatology_panels.nc"
OUT_PNG = HHE_DIR / "fig1_hhe_climatology_panels.png"
MONTHS = (6, 7, 8)

EXTENT = (-170.0, -55.0, 23.0, 72.0)  # lon_min, lon_max, lat_min, lat_max


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def _init_acc(shape: tuple[int, int], names: tuple[str, ...]):
    return {name: np.zeros(shape, dtype=np.float64) for name in names}, {
        name: np.zeros(shape, dtype=np.float64) for name in names
    }


def _add(sum_map: np.ndarray, count_map: np.ndarray, arr: np.ndarray):
    valid = np.isfinite(arr)
    sum_map += np.nansum(np.where(valid, arr, 0.0), axis=0)
    count_map += valid.sum(axis=0)


def _final(sum_map: np.ndarray, count_map: np.ndarray) -> np.ndarray:
    out = sum_map / np.where(count_map > 0, count_map, np.nan)
    return out.astype(np.float32)


def era5_climatology(years: list[int]):
    sums = counts = lat = lon = None
    for year in years:
        for month in MONTHS:
            path = ERA5_MONTHLY / f"hi_daily_y{year:04d}_m{month:02d}.nc"
            if not path.exists():
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                if lat is None:
                    lat = ds["lat"].values.astype(np.float32)
                    lon = ds["lon"].values.astype(np.float32)
                    sums, counts = _init_acc((lat.size, lon.size), ("rhmin", "tmax", "hi"))
                _add(sums["rhmin"], counts["rhmin"], ds["rhmin_pct"].values.astype(np.float32))
                _add(sums["tmax"], counts["tmax"], ds["tmax_C"].values.astype(np.float32) * 9.0 / 5.0 + 32.0)
                _add(sums["hi"], counts["hi"], ds["hi_F"].values.astype(np.float32))
        print(f"  ERA5 climatology {year} done", flush=True)
    return {
        "rhmin": _final(sums["rhmin"], counts["rhmin"]),
        "tmax": _final(sums["tmax"], counts["tmax"]),
        "hi": _final(sums["hi"], counts["hi"]),
    }, lat, lon


def model_climatology(years: list[int]):
    if not BIAS_NC.exists():
        raise FileNotFoundError(f"Missing {BIAS_NC}; run hhe_ace2_biascorr.py --force-bias first.")
    with xr.open_dataset(BIAS_NC) as ds:
        bias_t = ds["bias_Tmax"].values.astype(np.float32)
        bias_r = ds["bias_RHmin"].values.astype(np.float32)

    sums = counts = lat = lon = None
    for year in years:
        paths = sorted(glob.glob(str(ACE2_CACHE / f"daily_y{year}_mem*.nc")))
        if not paths:
            raise FileNotFoundError(f"No ACE2 daily cache files for {year} under {ACE2_CACHE}")
        used = 0
        for path in paths:
            with xr.open_dataset(path) as ds:
                tmax_c = ds["tmax_C"].values.astype(np.float32)
                rhmin = ds["rhmin_pct"].values.astype(np.float32)
                if lat is None:
                    lat = ds["lat"].values.astype(np.float32)
                    lon = ds["lon"].values.astype(np.float32)
                    sums, counts = _init_acc((lat.size, lon.size), ("rhmin", "tmax", "hi"))
                _add(sums["rhmin"], counts["rhmin"], rhmin)
                _add(sums["tmax"], counts["tmax"], tmax_c * 9.0 / 5.0 + 32.0)

                tmax_corr = tmax_c - bias_t
                rhmin_corr = np.clip(rhmin - bias_r, 0.0, 100.0)
                hi_corr = heat_index(tmax_corr * 9.0 / 5.0 + 32.0, rhmin_corr)
                _add(sums["hi"], counts["hi"], hi_corr)
                used += 1
        print(f"  model climatology {year}: {used} members", flush=True)
    return {
        "rhmin": _final(sums["rhmin"], counts["rhmin"]),
        "tmax": _final(sums["tmax"], counts["tmax"]),
        "hi": _final(sums["hi"], counts["hi"]),
    }, lat, lon


def _subset_for_extent(field, lat, lon_360, land_mask):
    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    lon_min, lon_max, lat_min, lat_max = EXTENT
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon_180 >= lon_min) & (lon_180 <= lon_max)
    sub = field[np.ix_(lat_sel, lon_sel)]
    if land_mask is not None:
        sub = np.where(land_mask[np.ix_(lat_sel, lon_sel)], sub, np.nan)
    return sub, lat[lat_sel], lon_180[lon_sel]


def _cmap_rh():
    cmap = ListedColormap(
        [
            "#4b2316",
            "#6f4527",
            "#9a6f3a",
            "#c99b4a",
            "#f4e8ba",
            "#b5efb4",
            "#34c41e",
            "#2b9b1c",
            "#237b17",
            "#14550e",
        ],
        "rh_brown_green",
    )
    cmap.set_bad("white")
    return cmap


def _cmap_temp():
    cmap = LinearSegmentedColormap.from_list(
        "temp_blue_red",
        ["#2a00ff", "#3b63f1", "#5ec6f0", "#76f2ed", "#eaff33", "#f3a027", "#ff331c", "#b90d0d"],
        N=256,
    )
    cmap.set_bad("white")
    return cmap


def _plot_panel(ax, field, lat, lon_360, land_mask, title, cmap, norm):
    sub, lat_sub, lon_sub = _subset_for_extent(field, lat, lon_360, land_mask)
    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
    mesh = ax.pcolormesh(lon2d, lat2d, sub, cmap=cmap, norm=norm, shading="nearest", zorder=1)
    lon_min, lon_max, lat_min, lat_max = EXTENT
    ax.set_facecolor("white")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    _draw_conus_borders(ax, (lon_min, lon_max), (lat_min, lat_max))
    ax.set_xticks([-160, -120, -80])
    ax.set_xticklabels(["160W", "120W", "80W"], fontsize=8)
    ax.set_yticks([30, 40, 50, 60, 70])
    ax.set_yticklabels(["30N", "40N", "50N", "60N", "70N"], fontsize=8)
    ax.grid(True, color="k", linestyle=(0, (1, 3)), linewidth=0.7, alpha=0.8, zorder=2)
    ax.set_title(title, fontsize=14, weight="bold", pad=5)
    return mesh


def plot_figure(era5, model, lat, lon, model_label: str, out_png: Path):
    land = load_land_mask(lat, lon)
    rh_levels = np.arange(0, 101, 10)
    temp_levels = np.arange(40, 121, 10)
    rh_cmap = _cmap_rh()
    temp_cmap = _cmap_temp()
    rh_norm = BoundaryNorm(rh_levels, rh_cmap.N)
    temp_norm = BoundaryNorm(temp_levels, temp_cmap.N)

    fig = plt.figure(figsize=(14.5, 7.0))
    gs = fig.add_gridspec(
        2, 3,
        left=0.045, right=0.985, top=0.94, bottom=0.20,
        wspace=0.10, hspace=0.08,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    titles = [
        ("(a) RHmin, ERA5", "(b) Tmax, ERA5", "(c) HI, ERA5"),
        (f"(d) RHmin, {model_label}", f"(e) Tmax, {model_label}", f"(f) HI, {model_label}"),
    ]
    meshes = []
    meshes.append(_plot_panel(axes[0][0], era5["rhmin"], lat, lon, land, titles[0][0], rh_cmap, rh_norm))
    meshes.append(_plot_panel(axes[0][1], era5["tmax"], lat, lon, land, titles[0][1], temp_cmap, temp_norm))
    meshes.append(_plot_panel(axes[0][2], era5["hi"], lat, lon, land, titles[0][2], temp_cmap, temp_norm))
    meshes.append(_plot_panel(axes[1][0], model["rhmin"], lat, lon, land, titles[1][0], rh_cmap, rh_norm))
    meshes.append(_plot_panel(axes[1][1], model["tmax"], lat, lon, land, titles[1][1], temp_cmap, temp_norm))
    meshes.append(_plot_panel(axes[1][2], model["hi"], lat, lon, land, titles[1][2], temp_cmap, temp_norm))

    for ax in axes[0]:
        ax.set_xticklabels([])
    for row in axes:
        for ax in row[1:]:
            ax.set_yticklabels([])

    fig.canvas.draw()
    bottom_positions = [ax.get_position() for ax in axes[1]]
    cbar_y = min(pos.y0 for pos in bottom_positions) - 0.075
    caxes = [
        fig.add_axes([pos.x0 + 0.02 * pos.width, cbar_y, pos.width * 0.96, 0.026])
        for pos in bottom_positions
    ]
    cb0 = fig.colorbar(meshes[0], cax=caxes[0], orientation="horizontal", ticks=np.arange(0, 101, 20))
    cb1 = fig.colorbar(meshes[1], cax=caxes[1], orientation="horizontal", ticks=np.arange(40, 121, 20))
    cb2 = fig.colorbar(meshes[2], cax=caxes[2], orientation="horizontal", ticks=np.arange(40, 121, 20))
    cb0.set_label("RHmin (%)", fontsize=12)
    cb1.set_label("Tmax (degF)", fontsize=12)
    cb2.set_label("HI (degF)", fontsize=12)
    for cb in (cb0, cb1, cb2):
        cb.ax.tick_params(labelsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def write_nc(era5, model, lat, lon, era5_years, model_years, model_label, out_nc):
    ds = xr.Dataset(
        {
            "era5_rhmin_pct": (("lat", "lon"), era5["rhmin"]),
            "era5_tmax_F": (("lat", "lon"), era5["tmax"]),
            "era5_hi_F": (("lat", "lon"), era5["hi"]),
            "model_rhmin_pct_raw": (("lat", "lon"), model["rhmin"]),
            "model_tmax_F_raw": (("lat", "lon"), model["tmax"]),
            "model_hi_F_bias_corrected_inputs": (("lat", "lon"), model["hi"]),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "era5_years": f"{era5_years[0]}-{era5_years[-1]}",
            "model_years": f"{model_years[0]}-{model_years[-1]}",
            "model_label": model_label,
            "note": "Model RHmin/Tmax panels are raw; model HI uses bias-corrected Tmax/RHmin inputs.",
        },
    )
    tmp = out_nc.with_suffix(out_nc.suffix + ".tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, out_nc)
    print(f"wrote {out_nc}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--era5-years", default="1995:2022")
    p.add_argument("--model-years", default=None, help="Defaults to --era5-years")
    p.add_argument("--model-label", default="ACE2")
    p.add_argument("--out-png", type=Path, default=OUT_PNG)
    p.add_argument("--out-nc", type=Path, default=OUT_NC)
    args = p.parse_args()

    era5_years = parse_years(args.era5_years)
    model_years = parse_years(args.model_years or args.era5_years)

    print("=== Figure 1 climatology: ERA5 ===", flush=True)
    era5, lat, lon = era5_climatology(era5_years)
    print("=== Figure 1 climatology: model ===", flush=True)
    model, mlat, mlon = model_climatology(model_years)
    if not (np.allclose(lat, mlat) and np.allclose(lon, mlon)):
        raise ValueError("ERA5 and model daily caches are not on the same grid")

    write_nc(era5, model, lat, lon, era5_years, model_years, args.model_label, args.out_nc)
    plot_figure(era5, model, lat, lon, args.model_label, args.out_png)


if __name__ == "__main__":
    main()
