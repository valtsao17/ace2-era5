#!/usr/bin/env python3
"""Raw heat-extreme frequency regressed onto raw strongest-box SST indices.

Companion to sst_hhe_strongestbox_regression.py. For each raw heat-extreme
month (June, July, August), use the ERA5-side green SST box from
sst_raw_fig2box_summary.json as the scalar predictor:

    predictor: standardized, detrended box-mean same-month SST
    response : gridded monthly raw heat-extreme frequency (% of month days)

The same SST predictor is used for ERA5 and ACE2 response panels within a row.

Outputs -> outputs/lag_may/sst_teleconnection_hhe_lagged/figures/
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sst_teleconnection_hhe_lagged import OUT_DIR, FIG_DIR
from sst_raw_fig2box_corr import build_monthly_sst, SST_VAR
from seasonal_jja_skill import load_land_mask
from sst_teleconnection_jja_sliding7d import (
    bbox_mean, detrend_1d, detrend_along_year, regression_map,
    _draw_conus_borders, _thin_mask,
)

RAW_NPZ = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall/monthly_raw_rc_fields.npz"
SUMMARY = OUT_DIR / "sst_raw_fig2box_summary.json"
MONTHS = {"jun": (6, "June"), "jul": (7, "July"), "aug": (8, "August")}
CBAR_LABEL = "Delta raw freq (% month days per +1 sigma SST)"


def sst_index_box(summary, tag):
    """Return the month-specific ERA5 strongest-corr SST box in 0-360 lon."""
    j = summary[tag]["highest_corr_box_era5"]
    if j is None:
        raise ValueError(f"No ERA5 SST correlation box found for {tag}")
    la0, la1 = j["latN"]
    lo_w0, lo_w1 = j["lonW"]
    lo0, lo1 = 360.0 - lo_w1, 360.0 - lo_w0
    return (float(la0), float(la1), float(lo0), float(lo1)), j


def sub_lon(lon):
    return np.where(lon > 180.0, lon - 360.0, lon)


def _coord_edges(coord):
    coord = np.asarray(coord, dtype=np.float32)
    if coord.size < 2:
        pad = 0.5
    else:
        pad = 0.5 * float(np.nanmedian(np.abs(np.diff(coord))))
    return float(coord[0] - pad), float(coord[-1] + pad)


def _lon_tick_values(xlim):
    start = int(np.ceil(xlim[0] / 20.0) * 20)
    end = int(np.floor(xlim[1] / 20.0) * 20)
    return list(range(start, end + 1, 20))


def _lat_tick_values(ylim):
    start = int(np.ceil(ylim[0] / 10.0) * 10)
    end = int(np.floor(ylim[1] / 10.0) * 10)
    return list(range(start, end + 1, 10))


def _render_raw_regression_ax(ax, slope, pval, lon_180, lat, vmax, title):
    """Draw one raw monthly regression panel on the native raw grid extent."""
    xlim = _coord_edges(lon_180)
    ylim = _coord_edges(lat)
    LON2D, LAT2D = np.meshgrid(lon_180, lat)

    ax.set_facecolor("white")
    mesh = ax.pcolormesh(
        LON2D, LAT2D, slope,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading="nearest", zorder=1,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    _draw_conus_borders(ax, xlim, ylim)

    sig = np.isfinite(pval) & (pval < 0.05)
    sig = _thin_mask(sig)
    ax.scatter(LON2D[sig], LAT2D[sig], s=2.5, c="k", alpha=0.55, zorder=5, linewidths=0)

    xticks = _lon_tick_values(xlim)
    yticks = _lat_tick_values(ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x)}°W" for x in xticks], fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=8)
    ax.set_title(title, fontsize=10)
    return mesh


def compute_month(tag, raw, lat, lon, land_mask, sst, slat, slon, summary):
    month_num, month_name = MONTHS[tag]
    box, box_info = sst_index_box(summary, tag)
    la0, la1, lo0, lo1 = box
    print(f"{month_name} SST index box: lat {la0:.1f}-{la1:.1f}N  "
          f"lon {360-lo1:.1f}-{360-lo0:.1f}W  mean r={box_info['mean_r']}",
          flush=True)

    sst_idx = bbox_mean(np.asarray(sst[SST_VAR[tag]], np.float32), slat, slon, box)
    sst_idx_dt = detrend_1d(sst_idx)
    sd = float(np.nanstd(sst_idx_dt, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"Bad SST-index standard deviation for {tag}: {sd}")
    sst_idx_std = sst_idx_dt / sd
    print(f"  SST index: mean={np.nanmean(sst_idx):.3f} K  detrended std={sd:.3f} K",
          flush=True)

    era5 = raw[f"fe{month_num}"].astype(np.float32) * 100.0
    ace2 = raw[f"fa{month_num}"].astype(np.float32) * 100.0
    era5 = np.where(land_mask[np.newaxis, :, :], era5, np.nan).astype(np.float32)
    ace2 = np.where(land_mask[np.newaxis, :, :], ace2, np.nan).astype(np.float32)
    era5_dt = detrend_along_year(era5)
    ace2_dt = detrend_along_year(ace2)

    print(f"  regressing gridded ERA5 raw {month_name} freq ...", flush=True)
    slope_e, pval_e = regression_map(era5_dt, sst_idx_std)
    print(f"  regressing gridded ACE2 raw {month_name} freq ...", flush=True)
    slope_a, pval_a = regression_map(ace2_dt, sst_idx_std)

    return {
        "tag": tag,
        "month": month_name,
        "sst_box": box,
        "sst_box_info": box_info,
        "sst_idx_mean": float(np.nanmean(sst_idx)),
        "sst_idx_dt_std": sd,
        "slope_era5": slope_e,
        "pval_era5": pval_e,
        "slope_ace2": slope_a,
        "pval_ace2": pval_a,
    }


def render_pair(result, lat, lon_180, vmax):
    tag = result["tag"]
    month = result["month"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.03, hspace=0.02)
    _render_raw_regression_ax(
        axes[0], result["slope_era5"], result["pval_era5"], lon_180, lat, vmax,
        f"a  ERA5 reanalysis - raw {month}",
    )
    mesh = _render_raw_regression_ax(
        axes[1], result["slope_ace2"], result["pval_ace2"], lon_180, lat, vmax,
        f"b  ACE2 hindcast - raw {month}",
    )
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical", label=CBAR_LABEL)
    fig.suptitle(
        f"RAW {month} heat-extreme frequency regressed onto standardized "
        f"{month} SST index  |  land only, stipple p<0.05",
        fontsize=12,
        y=0.99,
    )
    out = FIG_DIR / f"sst_raw_strongestbox_regression_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def render_combined(results, lat, lon_180, vmax):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11.3), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.08, wspace=0.03, hspace=0.06)
    mesh = None
    for r, result in enumerate(results):
        month = result["month"]
        mesh = _render_raw_regression_ax(
            axes[r, 0], result["slope_era5"], result["pval_era5"], lon_180, lat, vmax,
            f"{month}  ERA5 reanalysis",
        )
        mesh = _render_raw_regression_ax(
            axes[r, 1], result["slope_ace2"], result["pval_ace2"], lon_180, lat, vmax,
            f"{month}  ACE2 hindcast",
        )
    fig.colorbar(mesh, ax=axes, shrink=0.72, orientation="vertical", label=CBAR_LABEL)
    fig.suptitle(
        "RAW heat-extreme frequency regressed onto month-specific standardized "
        "same-month SST indices  |  land only, stipple p<0.05",
        fontsize=12,
    )
    out = FIG_DIR / "sst_raw_strongestbox_regression.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    if not RAW_NPZ.exists():
        raise FileNotFoundError(f"Missing {RAW_NPZ}. Run monthly_raw_freq_rankcorr.py first.")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing {SUMMARY}. Run sst_raw_fig2box_corr.py first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(RAW_NPZ)
    lat = raw["lat"].astype(np.float32)
    lon = raw["lon"].astype(np.float32)
    lon_180 = sub_lon(lon)
    land_mask = load_land_mask(lat, lon)
    if land_mask is None:
        raise RuntimeError("Could not build land mask from forcing files")
    summary = json.loads(SUMMARY.read_text())

    sst, slat, slon = build_monthly_sst()
    results = [
        compute_month(tag, raw, lat, lon, land_mask, sst, slat, slon, summary)
        for tag in ("jun", "jul", "aug")
    ]

    finite = []
    for result in results:
        finite.append(result["slope_era5"][np.isfinite(result["slope_era5"])])
        finite.append(result["slope_ace2"][np.isfinite(result["slope_ace2"])])
    finite = np.concatenate(finite) if finite else np.array([], dtype=np.float32)
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-6)

    for result in results:
        render_pair(result, lat, lon_180, vmax)
    render_combined(results, lat, lon_180, vmax)

    ds_vars = {}
    out_summary = {}
    for result in results:
        tag = result["tag"]
        ds_vars[f"slope_era5_{tag}"] = (("lat", "lon"), result["slope_era5"])
        ds_vars[f"pval_era5_{tag}"] = (("lat", "lon"), result["pval_era5"])
        ds_vars[f"slope_ace2_{tag}"] = (("lat", "lon"), result["slope_ace2"])
        ds_vars[f"pval_ace2_{tag}"] = (("lat", "lon"), result["pval_ace2"])
        box = result["sst_box"]
        out_summary[tag] = {
            "sst_index_box_latN": [round(box[0], 2), round(box[1], 2)],
            "sst_index_box_lonW": [round(360.0 - box[3], 2), round(360.0 - box[2], 2)],
            "source_mean_r": result["sst_box_info"]["mean_r"],
            "sst_idx_mean_K": round(result["sst_idx_mean"], 4),
            "sst_idx_detrended_std_K": round(result["sst_idx_dt_std"], 4),
        }

    xr.Dataset(
        ds_vars,
        coords={"lat": lat, "lon": lon},
        attrs={
            "sst": "ERA5 forcing surface_temperature, same-month mean",
            "raw_response": "Monthly raw Tmax > full-JJA 90th percentile frequency, percent of month days",
            "mask": "Response regression computed over land cells only",
            "units": "percent of month days per 1 std-dev of detrended selected same-month SST index",
            "direction": "gridded raw heat-extreme frequency (response) on scalar SST index (predictor)",
        },
    ).to_netcdf(OUT_DIR / "sst_raw_strongestbox_regression.nc")
    (OUT_DIR / "sst_raw_strongestbox_regression_summary.json").write_text(
        json.dumps(out_summary, indent=2)
    )
    print(f"wrote {OUT_DIR / 'sst_raw_strongestbox_regression.nc'}", flush=True)
    print(f"wrote {OUT_DIR / 'sst_raw_strongestbox_regression_summary.json'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
