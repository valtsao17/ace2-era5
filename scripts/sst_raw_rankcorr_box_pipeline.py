#!/usr/bin/env python3
"""Seasonal RAW heat-extreme SST pipeline using a rank-correlation index box.

This is the JJA seasonal version of the RAW SST figures, but the RAW index box
is selected from ACE2-vs-ERA5 Kendall tau skill rather than from ERA5 extreme
frequency. The RAW/DRY heat-extreme fields are the existing seasonal JJA
90th-percentile TMP2m frequencies from seasonal_jja_skill.py:

    outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc
    outputs/lag_may/seasonal_jja_sliding7d/skill_jja_seasonal.nc

Outputs:
  outputs/lag_may/sst_raw_rankcorr_box/figures/
    raw_jja_rankcorr_box.png
    sst_raw_jja_rankcorrbox_corr.png
    sst_raw_jja_strongestbox_regression.png
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
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    CONUS_LAT_SLICE, CONUS_LON_SLICE, _TAU_CMAP,
)
from monthly_raw_freq_rankcorr import _map  # noqa: E402
from seasonal_jja_skill import (  # noqa: E402
    YEARS, domain_scores_label, load_land_mask,
)
from sst_teleconnection_hhe_lagged import build_lagged_sst  # noqa: E402
from sst_teleconnection_jja_sliding7d import (  # noqa: E402
    bbox_mean, detrend_1d, detrend_along_year, pearson_corr_map,
    regression_map, _draw_conus_borders, _render_corr_ax, _thin_mask,
)

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
FREQ_NC = SLIDING_DIR / "jja_seasonal_freqs.nc"
SKILL_NC = SLIDING_DIR / "skill_jja_seasonal.nc"

OUT_DIR = PROJECT_ROOT / "outputs/lag_may/sst_raw_rankcorr_box"
FIG_DIR = OUT_DIR / "figures"

# Same RAW/HHE figure-2 hotspot box geometry already used elsewhere.
RAW_BOX_LAT_SPAN = 8.0
RAW_BOX_LON_SPAN = 12.0
MIN_LAND_CELLS = 8
MIN_ALLOWED_LAND_FRACTION = 0.70
ALLOWED_RAW_BOX_COUNTRIES = ("United States of America", "Canada")

# Same SST predictor-box geometry/domain as the existing fig2box SST figures.
SST_BOX_LAT_SPAN = 10.0
SST_BOX_LON_SPAN = 15.0
SST_SEARCH = dict(lat_s=-25.0, lat_n=70.0, lon_w=190.0, lon_e=320.0)


def _box_json(box):
    la0, la1, lo0, lo1 = box
    return {
        "latN": [round(float(la0), 2), round(float(la1), 2)],
        "lonW": [round(float(360.0 - lo1), 2), round(float(360.0 - lo0), 2)],
    }


def _country_mask(lat, lon_360, country_names):
    """Boolean mask for grid-cell centers inside Natural Earth countries."""
    wanted = {name.lower() for name in country_names}
    shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries",
    )
    geoms = []
    for rec in shpreader.Reader(shp).records():
        attrs = rec.attributes
        names = {
            str(attrs.get("ADMIN", "")).lower(),
            str(attrs.get("NAME", "")).lower(),
            str(attrs.get("NAME_LONG", "")).lower(),
            str(attrs.get("SOVEREIGNT", "")).lower(),
        }
        if names & wanted:
            geoms.append(rec.geometry)
    if not geoms:
        raise RuntimeError(f"No Natural Earth geometry found for {country_names}")

    lon_180 = np.where(lon_360 > 180.0, lon_360 - 360.0, lon_360)
    out = np.zeros((len(lat), len(lon_360)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon_180):
            pt = Point(float(lo), float(la))
            out[i, j] = any(g.contains(pt) or g.touches(pt) for g in geoms)
    return out


def find_rankcorr_box(tau, lat, lon, land, allowed_land):
    """Select an 8x12 deg RAW index box by maximum mean tau over land cells.

    The box may include a little water, but it must be mostly land, and all
    land pixels inside the box must be in the allowed countries. Only finite
    allowed-country land pixels contribute to the score. Significance is
    deliberately not used for selection.
    """
    best = None
    for la0 in lat:
        la1 = float(la0) + RAW_BOX_LAT_SPAN
        if la1 > float(lat[-1]):
            continue
        sla = (lat >= la0) & (lat <= la1)
        for lo0 in lon:
            lo1 = float(lo0) + RAW_BOX_LON_SPAN
            if lo1 > float(lon[-1]):
                continue
            slo = (lon >= lo0) & (lon <= lo1)
            sub_tau = tau[np.ix_(sla, slo)]
            sub_land = land[np.ix_(sla, slo)]
            sub_allowed_land = allowed_land[np.ix_(sla, slo)]
            if np.any(sub_land & ~sub_allowed_land):
                continue
            allowed_land_fraction = float(sub_allowed_land.mean())
            if allowed_land_fraction < MIN_ALLOWED_LAND_FRACTION:
                continue
            valid = np.isfinite(sub_tau) & sub_allowed_land
            n_land = int(valid.sum())
            if n_land < MIN_LAND_CELLS:
                continue
            score = float(np.nanmean(sub_tau[valid]))
            if best is None or score > best["mean_tau"]:
                best = {
                    "bbox": (float(la0), float(la1), float(lo0), float(lo1)),
                    "mean_tau": score,
                    "n_land_cells": n_land,
                    "allowed_land_fraction": allowed_land_fraction,
                }
    if best is None:
        raise RuntimeError("No valid land rank-correlation box found.")
    return best


def find_positive_sst_corr_box(corr, lat, lon):
    """Find strongest positive mean-correlation SST box over finite ocean cells."""
    w = SST_SEARCH
    cand_lat = lat[(lat >= w["lat_s"]) & (lat <= w["lat_n"])]
    cand_lon = lon[(lon >= w["lon_w"]) & (lon <= w["lon_e"])]
    best = None
    for la0 in cand_lat:
        la1 = float(la0) + SST_BOX_LAT_SPAN
        if la1 > w["lat_n"]:
            continue
        sla = (lat >= la0) & (lat <= la1)
        for lo0 in cand_lon:
            lo1 = float(lo0) + SST_BOX_LON_SPAN
            if lo1 > w["lon_e"]:
                continue
            slo = (lon >= lo0) & (lon <= lo1)
            sub = corr[np.ix_(sla, slo)]
            finite = np.isfinite(sub)
            if finite.mean() < 0.6:
                continue
            mean_r = float(np.nanmean(sub))
            if mean_r <= 0.0:
                continue
            if best is None or mean_r > best["mean_r"]:
                best = {
                    "bbox": (float(la0), float(la1), float(lo0), float(lo1)),
                    "mean_r": mean_r,
                    "n_cells": int(finite.sum()),
                }
    if best is None:
        raise RuntimeError("No positive finite-ocean SST correlation box found.")
    return best


def draw_sst_box(ax, box, label):
    la0, la1, lo0, lo1 = box
    x0 = lo0 - 360.0 if lo0 > 180.0 else lo0
    x1 = lo1 - 360.0 if lo1 > 180.0 else lo1
    ax.add_patch(plt.Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="#00b050", lw=2.4, zorder=7))
    ax.text(x0, la0 - 1.5, label, color="#00b050", fontsize=9, weight="bold",
            va="top", ha="left", zorder=8)


def render_rankcorr_box(tau_c, pval_c, lat_c, lon_c, land_c, box_info):
    tlim = 1.0
    tau_ns = np.isfinite(tau_c) & ~(np.isfinite(pval_c) & (pval_c < 0.05))

    fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
    _map(
        ax, tau_c, lat_c, lon_c,
        "Kendall τ — RAW JJA  (ACE2 vs ERA5)",
        _TAU_CMAP, -tlim, tlim, "τ",
        sig=tau_ns,
        mean_lbl=domain_scores_label("mean τ", tau_c, lat_c, land_c),
        box=box_info["bbox"],
    )
    fig.suptitle(
        "RAW JJA heat-extreme rank correlation  |  green box = max land mean τ "
        f"(>= {MIN_ALLOWED_LAND_FRACTION:.0%} US/Canada land; "
        "no significance filter; stipple = not significant)",
        fontsize=10,
    )
    out = FIG_DIR / "raw_jja_rankcorr_box.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def _coord_edges(coord):
    coord = np.asarray(coord, dtype=np.float32)
    pad = 0.5 if coord.size < 2 else 0.5 * float(np.nanmedian(np.abs(np.diff(coord))))
    return float(coord[0] - pad), float(coord[-1] + pad)


def _lon_tick_values(xlim):
    start = int(np.ceil(xlim[0] / 20.0) * 20)
    end = int(np.floor(xlim[1] / 20.0) * 20)
    return list(range(start, end + 1, 20))


def _lat_tick_values(ylim):
    start = int(np.ceil(ylim[0] / 10.0) * 10)
    end = int(np.floor(ylim[1] / 10.0) * 10)
    return list(range(start, end + 1, 10))


def render_raw_regression_ax(ax, slope, pval, lon_180, lat, vmax, title):
    xlim = _coord_edges(lon_180)
    ylim = _coord_edges(lat)
    lon2d, lat2d = np.meshgrid(lon_180, lat)

    ax.set_facecolor("white")
    mesh = ax.pcolormesh(
        lon2d, lat2d, slope,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading="nearest", zorder=1,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    _draw_conus_borders(ax, xlim, ylim)

    sig = np.isfinite(pval) & (pval < 0.05)
    sig = _thin_mask(sig)
    ax.scatter(lon2d[sig], lat2d[sig], s=2.5, c="k", alpha=0.55,
               zorder=5, linewidths=0)

    xticks = _lon_tick_values(xlim)
    yticks = _lat_tick_values(ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x)}°W" for x in xticks], fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=8)
    ax.set_title(title, fontsize=10)
    return mesh


def render_sst_corr(corr_e, pval_e, corr_a, pval_a, slat, slon, sst_box):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
    mesh = _render_corr_ax(
        axes[0], corr_e, pval_e, slat, slon,
        "RAW JJA index: ERA5   (JJA SST x ERA5 raw freq)",
    )
    draw_sst_box(axes[0], sst_box["bbox"], f"r={sst_box['mean_r']:.2f}")
    mesh = _render_corr_ax(
        axes[1], corr_a, pval_a, slat, slon,
        "RAW JJA index: ACE2   (JJA SST x ACE2 raw freq)",
    )
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Pearson r  (detrended JJA SST x RAW JJA heat-extreme index)")
    fig.suptitle(
        "JJA SST vs RAW JJA heat-extreme frequency over rank-correlation box  |  "
        "ocean pixels only, stipple p<0.05, ERA5 green box = strongest positive mean correlation",
        fontsize=12,
    )
    out = FIG_DIR / "sst_raw_jja_rankcorrbox_corr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def render_regression(slope_e, pval_e, slope_a, pval_a, lat, lon, sst_box):
    lat_c = lat[CONUS_LAT_SLICE]
    lon_c = lon[CONUS_LON_SLICE]
    lon_180 = np.where(lon_c > 180.0, lon_c - 360.0, lon_c)

    def sub(a):
        return a[CONUS_LAT_SLICE, CONUS_LON_SLICE]

    se, pe = sub(slope_e), sub(pval_e)
    sa, pa = sub(slope_a), sub(pval_a)
    finite = np.concatenate([se[np.isfinite(se)], sa[np.isfinite(sa)]])
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.03, hspace=0.02)
    render_raw_regression_ax(
        axes[0], se, pe, lon_180, lat_c, vmax,
        "a  ERA5 reanalysis - RAW JJA",
    )
    mesh = render_raw_regression_ax(
        axes[1], sa, pa, lon_180, lat_c, vmax,
        "b  ACE2 hindcast - RAW JJA",
    )
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Delta raw freq (% JJA days per +1 sigma SST)")
    box_j = _box_json(sst_box)
    fig.suptitle(
        f"RAW JJA heat-extreme frequency regressed onto standardized JJA SST index "
        f"({box_j['latN'][0]:.0f}-{box_j['latN'][1]:.0f}N, "
        f"{box_j['lonW'][0]:.0f}-{box_j['lonW'][1]:.0f}W)  |  land only, stipple p<0.05",
        fontsize=12,
        y=0.99,
    )
    out = FIG_DIR / "sst_raw_jja_strongestbox_regression.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    if not FREQ_NC.exists():
        raise FileNotFoundError(f"Missing {FREQ_NC}. Run scripts/seasonal_jja_skill.py first.")
    if not SKILL_NC.exists():
        raise FileNotFoundError(f"Missing {SKILL_NC}. Run scripts/seasonal_jja_skill.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        era5 = ds["era5_freq"].values.astype(np.float32)
        ace2 = ds["ace2_freq"].values.astype(np.float32)
    with xr.open_dataset(SKILL_NC) as ds:
        tau = ds["kendall_tau"].values.astype(np.float32)
        tau_p = ds["tau_p_value"].values.astype(np.float32)

    land = load_land_mask(lat, lon)
    if land is None:
        raise RuntimeError("Could not build land mask from forcing files.")
    allowed_land = _country_mask(lat, lon, ALLOWED_RAW_BOX_COUNTRIES) & land
    print(
        "RAW rank-corr box allowed land countries: "
        f"{', '.join(ALLOWED_RAW_BOX_COUNTRIES)}",
        flush=True,
    )

    lat_c = lat[CONUS_LAT_SLICE]
    lon_c = lon[CONUS_LON_SLICE]
    tau_c = tau[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    tau_p_c = tau_p[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    land_c = land[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    allowed_land_c = allowed_land[CONUS_LAT_SLICE, CONUS_LON_SLICE]

    raw_box = find_rankcorr_box(tau_c, lat_c, lon_c, land_c, allowed_land_c)
    la0, la1, lo0, lo1 = raw_box["bbox"]
    print(f"RAW JJA rank-corr index box: lat {la0:.1f}-{la1:.1f}N  "
          f"lon {360.0-lo1:.1f}-{360.0-lo0:.1f}W  "
          f"land mean tau={raw_box['mean_tau']:.3f}  "
          f"n_land={raw_box['n_land_cells']}  "
          f"land_frac={raw_box['allowed_land_fraction']:.2f}", flush=True)
    render_rankcorr_box(tau_c, tau_p_c, lat_c, lon_c, land_c, raw_box)

    raw_bbox = raw_box["bbox"]
    raw_index_mask = allowed_land
    era5_idx = bbox_mean(np.where(raw_index_mask[np.newaxis, :, :], era5, np.nan),
                         lat, lon, raw_bbox)
    ace2_idx = bbox_mean(np.where(raw_index_mask[np.newaxis, :, :], ace2, np.nan),
                         lat, lon, raw_bbox)
    print(f"RAW JJA index climatology: ERA5={np.nanmean(era5_idx) * 100.0:.2f}%  "
          f"ACE2={np.nanmean(ace2_idx) * 100.0:.2f}%", flush=True)

    sst, slat, slon = build_lagged_sst()
    sst_jja = np.asarray(sst["lag0_JJA"], np.float32)
    sst_dt = detrend_along_year(sst_jja)

    print("Correlating JJA SST x ERA5 RAW JJA index ...", flush=True)
    corr_e, pval_e = pearson_corr_map(sst_dt, era5_idx)
    print("Correlating JJA SST x ACE2 RAW JJA index ...", flush=True)
    corr_a, pval_a = pearson_corr_map(sst_dt, ace2_idx)

    sst_box = find_positive_sst_corr_box(corr_e, slat, slon)
    sb = sst_box["bbox"]
    print(f"ERA5 strongest SST box: lat {sb[0]:.1f}-{sb[1]:.1f}N  "
          f"lon {360.0-sb[3]:.1f}-{360.0-sb[2]:.1f}W  "
          f"mean r={sst_box['mean_r']:.3f}  n_cells={sst_box['n_cells']}",
          flush=True)
    render_sst_corr(corr_e, pval_e, corr_a, pval_a, slat, slon, sst_box)

    sst_idx = bbox_mean(sst_jja, slat, slon, sb)
    sst_idx_dt = detrend_1d(sst_idx)
    sd = float(np.nanstd(sst_idx_dt, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"Bad SST-index standard deviation: {sd}")
    sst_idx_std = sst_idx_dt / sd
    print(f"SST index: mean={np.nanmean(sst_idx):.3f} K  detrended std={sd:.3f} K",
          flush=True)

    era5_land = np.where(land[np.newaxis, :, :], era5 * 100.0, np.nan).astype(np.float32)
    ace2_land = np.where(land[np.newaxis, :, :], ace2 * 100.0, np.nan).astype(np.float32)
    era5_dt = detrend_along_year(era5_land)
    ace2_dt = detrend_along_year(ace2_land)

    print("Regressing gridded ERA5 RAW JJA freq on standardized SST index ...", flush=True)
    slope_e, pval_re = regression_map(era5_dt, sst_idx_std)
    print("Regressing gridded ACE2 RAW JJA freq on standardized SST index ...", flush=True)
    slope_a, pval_ra = regression_map(ace2_dt, sst_idx_std)
    render_regression(slope_e, pval_re, slope_a, pval_ra, lat, lon, sb)

    xr.Dataset(
        {
            "corr_era5": (("lat", "lon"), corr_e),
            "pval_era5": (("lat", "lon"), pval_e),
            "corr_ace2": (("lat", "lon"), corr_a),
            "pval_ace2": (("lat", "lon"), pval_a),
        },
        coords={"lat": slat, "lon": slon},
        attrs={
            "raw_index_box": json.dumps(_box_json(raw_bbox)),
            "sst": "ERA5 forcing surface_temperature, JJA mean, detrended, ocean/ice masked",
            "raw_index": "JJA raw TMP2m heat-extreme seasonal frequency, +/-7d no-LOYO p90, averaged over US/Canada land cells inside rank-correlation box",
        },
    ).to_netcdf(OUT_DIR / "sst_raw_jja_rankcorrbox_corr.nc")

    xr.Dataset(
        {
            "slope_era5": (("lat", "lon"), slope_e),
            "pval_era5": (("lat", "lon"), pval_re),
            "slope_ace2": (("lat", "lon"), slope_a),
            "pval_ace2": (("lat", "lon"), pval_ra),
        },
        coords={"lat": lat, "lon": lon},
        attrs={
            "sst_index_box": json.dumps(_box_json(sb)),
            "units": "percent of JJA days per 1 std-dev of detrended box-mean JJA SST",
            "mask": "Response regression computed over land cells only",
        },
    ).to_netcdf(OUT_DIR / "sst_raw_jja_strongestbox_regression.nc")

    summary = {
        "years": [YEARS[0], YEARS[-1]],
        "raw_definition": "JJA raw TMP2m heat-extreme seasonal frequency; dataset-specific +/-7d no-LOYO 90th percentile thresholds",
        "raw_index_average": "cos-lat mean over US/Canada land cells inside the selected rank-correlation box",
        "rankcorr_box_selection": (
            "8x12 degree box maximizing mean Kendall tau over finite land cells; "
            "no significance filter; every land cell in the candidate box must be in the US or Canada; "
            f"candidate box must be at least {MIN_ALLOWED_LAND_FRACTION:.0%} US/Canada land"
        ),
        "allowed_raw_box_countries": list(ALLOWED_RAW_BOX_COUNTRIES),
        "minimum_allowed_land_fraction": MIN_ALLOWED_LAND_FRACTION,
        "raw_rankcorr_box": {
            **_box_json(raw_bbox),
            "land_mean_tau": round(float(raw_box["mean_tau"]), 4),
            "n_land_cells": int(raw_box["n_land_cells"]),
            "allowed_land_fraction": round(float(raw_box["allowed_land_fraction"]), 4),
            "era5_idx_clim_pct": round(float(np.nanmean(era5_idx) * 100.0), 4),
            "ace2_idx_clim_pct": round(float(np.nanmean(ace2_idx) * 100.0), 4),
        },
        "highest_corr_box_era5": {
            **_box_json(sb),
            "mean_r": round(float(sst_box["mean_r"]), 4),
            "n_cells": int(sst_box["n_cells"]),
            "sst_idx_mean_K": round(float(np.nanmean(sst_idx)), 4),
            "sst_idx_detrended_std_K": round(sd, 4),
        },
    }
    (OUT_DIR / "sst_raw_rankcorr_box_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'sst_raw_rankcorr_box_summary.json'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
