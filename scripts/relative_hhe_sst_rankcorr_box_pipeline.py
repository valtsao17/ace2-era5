#!/usr/bin/env python3
"""Relative-HHE SST pipeline using a rank-correlation index box.

This mirrors sst_raw_rankcorr_box_pipeline.py, but the target field is the JJA
relative humid-heat extreme frequency from relative_hhe_jja_skill.py:

  outputs/lag_may/relative_hhe_jja_sliding7d/jja_seasonal_freqs.nc
  outputs/lag_may/relative_hhe_jja_sliding7d/skill_jja_seasonal.nc

ACE2 SST is loaded from true inference outputs:
  outputs/lag_may/runs/{year}/member_XX/autoregressive_predictions.nc
"""
from __future__ import annotations

import json
import os
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

from cluster_skill_analysis_sliding7d import CONUS_LAT_SLICE, CONUS_LON_SLICE, _TAU_CMAP  # noqa: E402
from monthly_raw_freq_rankcorr import _map  # noqa: E402
from seasonal_jja_skill import domain_scores_label, load_land_mask  # noqa: E402
from sst_teleconnection_hhe_lagged import build_lagged_sst, YEARS as FORCING_SST_YEARS  # noqa: E402
from sst_teleconnection_jja_sliding7d import (  # noqa: E402
    bbox_mean, detrend_along_year, pearson_corr_map,
    regression_map, _draw_conus_borders, _render_corr_ax, _thin_mask,
)
from extreme_sst_pipeline import yearly_cube_from_ace2_runs  # noqa: E402

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/relative_hhe_jja_sliding7d"
FREQ_NC = SLIDING_DIR / "jja_seasonal_freqs.nc"
SKILL_NC = SLIDING_DIR / "skill_jja_seasonal.nc"
OUT_DIR = PROJECT_ROOT / "outputs/lag_may/relative_hhe_sst_rankcorr_box"
FIG_DIR = OUT_DIR / "figures"

RUNS_ROOT = Path(os.environ.get("RELHHE_MODEL_SST_RUNS_ROOT", PROJECT_ROOT / "outputs/lag_may/runs"))
MODEL_SST_VAR = os.environ.get("RELHHE_MODEL_SST_VAR", "surface_temperature")
SEASON_MONTHS = (6, 7, 8)

RAW_BOX_LAT_SPANS = (8.0, 10.0, 12.0, 15.0)
RAW_BOX_LON_SPANS = (10.0, 12.0, 15.0, 18.0)
SIGNIFICANCE_ALPHA = 0.05
MIN_LAND_CELLS = 12
MIN_SIGNIFICANT_LAND_CELLS = 8
MIN_SIGNIFICANT_LAND_FRACTION = 0.70
MIN_ALLOWED_LAND_FRACTION = 0.70
RAW_BOX_EXTEND_SOUTH_CELLS = 5
RAW_BOX_EXTEND_EAST_CELLS = 2
ALLOWED_RAW_BOX_COUNTRIES = ("United States of America", "Canada")

SST_BOX_LAT_SPAN = 10.0
SST_BOX_LON_SPAN = 15.0
SST_SEARCH = dict(lat_s=-25.0, lat_n=70.0, lon_w=190.0, lon_e=320.0)


def _box_json(box):
    la0, la1, lo0, lo1 = box
    return {
        "latN": [round(float(la0), 2), round(float(la1), 2)],
        "lonW": [round(float(360.0 - lo1), 2), round(float(360.0 - lo0), 2)],
    }


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def available_run_years(runs_root: Path, candidate_years: list[int]) -> list[int]:
    out = []
    for year in candidate_years:
        year_dir = runs_root / str(year)
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("member_*/autoregressive_predictions.nc")):
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                with xr.open_dataset(path, decode_times=False) as ds:
                    has_var = MODEL_SST_VAR in ds
            except Exception:
                has_var = False
            if has_var:
                out.append(int(year))
                break
    return out


def _subset_years(arr: np.ndarray, available_years: list[int], wanted_years: list[int]) -> np.ndarray:
    lookup = {int(y): i for i, y in enumerate(available_years)}
    missing = [int(y) for y in wanted_years if int(y) not in lookup]
    if missing:
        raise ValueError(f"Missing years: {missing[:8]}")
    return arr[[lookup[int(y)] for y in wanted_years]]


def _same_grid(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray, label: str):
    if not (np.allclose(lat_a, lat_b) and np.allclose(lon_a, lon_b)):
        raise ValueError(
            f"{label} grid mismatch: lat {lat_a.shape} vs {lat_b.shape}, "
            f"lon {lon_a.shape} vs {lon_b.shape}"
        )


def _standardize_index(x: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    x_dt = np.asarray(x, dtype=np.float32)
    # detrend_1d is equivalent to detrend_along_year for a single index, but the
    # latter expects a leading year axis and works fine for 1D input.
    from sst_teleconnection_jja_sliding7d import detrend_1d
    x_dt = detrend_1d(x_dt)
    sd = float(np.nanstd(x_dt, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError(f"Bad {label} standard deviation: {sd}")
    return x_dt / sd, sd


def expand_box_on_grid(
    box: tuple[float, float, float, float],
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    south_cells: int = RAW_BOX_EXTEND_SOUTH_CELLS,
    east_cells: int = RAW_BOX_EXTEND_EAST_CELLS,
) -> tuple[float, float, float, float]:
    la0, la1, lo0, lo1 = box
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    i0 = int(np.argmin(np.abs(lat - la0)))
    j1 = int(np.argmin(np.abs(lon - lo1)))
    i0_new = max(0, i0 - int(south_cells))
    j1_new = min(len(lon) - 1, j1 + int(east_cells))
    return float(lat[i0_new]), float(la1), float(lo0), float(lon[j1_new])


def _country_mask(lat, lon_360, country_names):
    wanted = {name.lower() for name in country_names}
    shp = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
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


def find_rankcorr_box(tau, pval, lat, lon, land, allowed_land):
    best = None
    for lat_span in RAW_BOX_LAT_SPANS:
        for lon_span in RAW_BOX_LON_SPANS:
            for la0 in lat:
                la1 = float(la0) + lat_span
                if la1 > float(lat[-1]):
                    continue
                sla = (lat >= la0) & (lat <= la1)
                for lo0 in lon:
                    lo1 = float(lo0) + lon_span
                    if lo1 > float(lon[-1]):
                        continue
                    slo = (lon >= lo0) & (lon <= lo1)
                    sub_tau = tau[np.ix_(sla, slo)]
                    sub_pval = pval[np.ix_(sla, slo)]
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
                    sig = valid & np.isfinite(sub_pval) & (sub_pval < SIGNIFICANCE_ALPHA)
                    n_sig = int(sig.sum())
                    sig_fraction = float(n_sig / n_land) if n_land else 0.0
                    if n_sig < MIN_SIGNIFICANT_LAND_CELLS:
                        continue
                    if sig_fraction < MIN_SIGNIFICANT_LAND_FRACTION:
                        continue
                    mean_tau = float(np.nanmean(sub_tau[valid]))
                    mean_tau_sig = float(np.nanmean(sub_tau[sig]))
                    score = mean_tau_sig * sig_fraction
                    if best is None or score > best["score"]:
                        best = {
                            "bbox": (float(la0), float(la1), float(lo0), float(lo1)),
                            "score": score,
                            "mean_tau": mean_tau,
                            "mean_tau_significant": mean_tau_sig,
                            "n_land_cells": n_land,
                            "n_significant_land_cells": n_sig,
                            "significant_land_fraction": sig_fraction,
                            "allowed_land_fraction": allowed_land_fraction,
                            "lat_span": float(lat_span),
                            "lon_span": float(lon_span),
                        }
    if best is not None:
        return best
    raise RuntimeError("No valid land rank-correlation box found.")


def find_positive_sst_corr_box(corr, lat, lon):
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
    tau_ns = np.isfinite(tau_c) & ~(np.isfinite(pval_c) & (pval_c < 0.05))
    fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
    _map(
        ax, tau_c, lat_c, lon_c,
        "Kendall tau - relative HHE JJA (ACE2 vs ERA5)",
        _TAU_CMAP, -1.0, 1.0, "tau",
        sig=tau_ns,
        mean_lbl=domain_scores_label("mean tau", tau_c, lat_c, land_c),
        box=box_info["bbox"],
    )
    fig.suptitle(
        "Relative-HHE JJA rank correlation | green box = compact significant-skill box, expanded "
        f"S+{RAW_BOX_EXTEND_SOUTH_CELLS}/E+{RAW_BOX_EXTEND_EAST_CELLS} grid cells "
        f"(>= {MIN_ALLOWED_LAND_FRACTION:.0%} US/Canada land; stipple = not significant)",
        fontsize=10,
    )
    out = FIG_DIR / "relhhe_jja_rankcorr_box.png"
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


def render_regression_ax(ax, slope, pval, lon_180, lat, vmax, title):
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
    sig = _thin_mask(np.isfinite(pval) & (pval < 0.05))
    ax.scatter(lon2d[sig], lat2d[sig], s=2.5, c="k", alpha=0.55, zorder=5, linewidths=0)
    xticks = _lon_tick_values(xlim)
    yticks = _lat_tick_values(ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x)}W" for x in xticks], fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}N" for y in yticks], fontsize=8)
    ax.set_title(title, fontsize=10)
    return mesh


def render_sst_corr(corr_e, pval_e, corr_a, pval_a, slat, slon, sst_box):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
    mesh = _render_corr_ax(
        axes[0], corr_e, pval_e, slat, slon,
        "Relative HHE index: ERA5 (forcing SST x ERA5 freq)",
    )
    draw_sst_box(axes[0], sst_box["bbox"], f"r={sst_box['mean_r']:.2f}")
    mesh = _render_corr_ax(
        axes[1], corr_a, pval_a, slat, slon,
        "Relative HHE index: ACE2 (inference SST x ACE2 freq)",
    )
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Pearson r (detrended JJA SST x relative-HHE index)")
    fig.suptitle(
        "JJA SST vs relative HHE frequency over rank-correlation box | "
        "ERA5 uses forcing SST, ACE2 uses inference-output SST; stipple p<0.05",
        fontsize=12,
    )
    out = FIG_DIR / "sst_relhhe_jja_rankcorrbox_corr.png"
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
    render_regression_ax(axes[0], se, pe, lon_180, lat_c, vmax, "a  ERA5 reanalysis - relative HHE")
    mesh = render_regression_ax(axes[1], sa, pa, lon_180, lat_c, vmax, "b  ACE2 hindcast - relative HHE")
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Delta relative-HHE freq (% JJA days per +1 sigma SST)")
    box_j = _box_json(sst_box)
    fig.suptitle(
        f"Relative HHE frequency regressed onto standardized JJA SST index "
        f"({box_j['latN'][0]:.0f}-{box_j['latN'][1]:.0f}N, "
        f"{box_j['lonW'][0]:.0f}-{box_j['lonW'][1]:.0f}W) | "
        "ERA5 forcing SST, ACE2 inference SST; stipple p<0.05",
        fontsize=12,
        y=0.99,
    )
    out = FIG_DIR / "sst_relhhe_jja_strongestbox_regression.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    if not FREQ_NC.exists():
        raise FileNotFoundError(f"Missing {FREQ_NC}. Run scripts/relative_hhe_jja_skill.py first.")
    if not SKILL_NC.exists():
        raise FileNotFoundError(f"Missing {SKILL_NC}. Run scripts/relative_hhe_jja_skill.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(FREQ_NC) as ds:
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        target_years_all = [int(y) for y in ds["year"].values]
        requested_years = os.environ.get("RELHHE_MODEL_SST_YEARS")
        analysis_years = parse_years(requested_years) if requested_years else available_run_years(RUNS_ROOT, target_years_all)
        if not analysis_years:
            raise RuntimeError(f"No inference-output years found under {RUNS_ROOT}.")
        missing_targets = [y for y in analysis_years if y not in target_years_all]
        if missing_targets:
            raise ValueError(f"Requested years missing from {FREQ_NC}: {missing_targets[:8]}")
        target_idx = [target_years_all.index(y) for y in analysis_years]
        era5 = ds["era5_freq"].isel(year=target_idx).values.astype(np.float32)
        ace2 = ds["ace2_freq"].isel(year=target_idx).values.astype(np.float32)
        target_definition = ds.attrs.get("definition", "relative humid-heat extreme")
        ace2_hi_inputs = ds.attrs.get("ace2_hi_inputs", "unknown")
    print(
        f"Relative-HHE SST/rankcorr years: {analysis_years[0]}-{analysis_years[-1]} "
        f"({len(analysis_years)} years), model SST from {RUNS_ROOT}",
        flush=True,
    )

    with xr.open_dataset(SKILL_NC) as ds:
        tau = ds["kendall_tau"].values.astype(np.float32)
        tau_p = ds["tau_p_value"].values.astype(np.float32)

    land = load_land_mask(lat, lon)
    if land is None:
        raise RuntimeError("Could not build land mask from forcing files.")
    allowed_land = _country_mask(lat, lon, ALLOWED_RAW_BOX_COUNTRIES) & land

    lat_c = lat[CONUS_LAT_SLICE]
    lon_c = lon[CONUS_LON_SLICE]
    tau_c = tau[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    tau_p_c = tau_p[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    land_c = land[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    allowed_land_c = allowed_land[CONUS_LAT_SLICE, CONUS_LON_SLICE]

    hhe_box = find_rankcorr_box(tau_c, tau_p_c, lat_c, lon_c, land_c, allowed_land_c)
    selected_bbox = hhe_box["bbox"]
    hhe_box["selected_bbox"] = selected_bbox
    hhe_box["bbox"] = expand_box_on_grid(selected_bbox, lat_c, lon_c)
    la0, la1, lo0, lo1 = hhe_box["bbox"]
    print(f"Relative-HHE rank-corr index box: lat {la0:.1f}-{la1:.1f}N "
          f"lon {360.0-lo1:.1f}-{360.0-lo0:.1f}W "
          f"mean tau={hhe_box['mean_tau']:.3f} sig mean tau={hhe_box['mean_tau_significant']:.3f}",
          flush=True)
    render_rankcorr_box(tau_c, tau_p_c, lat_c, lon_c, land_c, hhe_box)

    hhe_bbox = hhe_box["bbox"]
    hhe_index_mask = allowed_land
    era5_idx = bbox_mean(np.where(hhe_index_mask[np.newaxis, :, :], era5, np.nan), lat, lon, hhe_bbox)
    ace2_idx = bbox_mean(np.where(hhe_index_mask[np.newaxis, :, :], ace2, np.nan), lat, lon, hhe_bbox)
    print(f"Relative-HHE index climatology: ERA5={np.nanmean(era5_idx) * 100.0:.2f}% "
          f"ACE2={np.nanmean(ace2_idx) * 100.0:.2f}%", flush=True)

    era5_sst_all, slat, slon = build_lagged_sst()
    era5_sst_jja = _subset_years(
        np.asarray(era5_sst_all["lag0_JJA"], np.float32),
        list(FORCING_SST_YEARS),
        analysis_years,
    )
    era5_sst_dt = detrend_along_year(era5_sst_jja)

    print(f"Loading ACE2 inference SST from {RUNS_ROOT} ({MODEL_SST_VAR}) ...", flush=True)
    ace2_sst_jja, model_slat, model_slon, model_sst_years, model_sst_var = yearly_cube_from_ace2_runs(
        RUNS_ROOT,
        MODEL_SST_VAR,
        analysis_years,
        season_months=SEASON_MONTHS,
        allow_partial=True,
        mask_ocean=True,
    )
    if model_sst_years != analysis_years:
        raise ValueError(f"Model SST years mismatch: {model_sst_years} vs {analysis_years}")
    _same_grid(slat, slon, model_slat, model_slon, "ERA5 forcing SST vs ACE2 inference SST")
    ace2_sst_dt = detrend_along_year(ace2_sst_jja)

    print("Correlating ERA5 forcing JJA SST x ERA5 relative-HHE index ...", flush=True)
    corr_e, pval_e = pearson_corr_map(era5_sst_dt, era5_idx)
    print("Correlating ACE2 inference JJA SST x ACE2 relative-HHE index ...", flush=True)
    corr_a, pval_a = pearson_corr_map(ace2_sst_dt, ace2_idx)

    sst_box = find_positive_sst_corr_box(corr_e, slat, slon)
    sb = sst_box["bbox"]
    print(f"ERA5 strongest SST box: lat {sb[0]:.1f}-{sb[1]:.1f}N "
          f"lon {360.0-sb[3]:.1f}-{360.0-sb[2]:.1f}W mean r={sst_box['mean_r']:.3f}",
          flush=True)
    render_sst_corr(corr_e, pval_e, corr_a, pval_a, slat, slon, sst_box)

    era5_sst_idx = bbox_mean(era5_sst_jja, slat, slon, sb)
    ace2_sst_idx = bbox_mean(ace2_sst_jja, model_slat, model_slon, sb)
    era5_sst_idx_std, era5_sd = _standardize_index(era5_sst_idx, "ERA5 SST index")
    ace2_sst_idx_std, ace2_sd = _standardize_index(ace2_sst_idx, "ACE2 inference SST index")
    print(f"ERA5 SST index detrended std={era5_sd:.3f} K", flush=True)
    print(f"ACE2 inference SST index detrended std={ace2_sd:.3f} K", flush=True)

    era5_land = np.where(land[np.newaxis, :, :], era5 * 100.0, np.nan).astype(np.float32)
    ace2_land = np.where(land[np.newaxis, :, :], ace2 * 100.0, np.nan).astype(np.float32)
    slope_e, pval_re = regression_map(detrend_along_year(era5_land), era5_sst_idx_std)
    slope_a, pval_ra = regression_map(detrend_along_year(ace2_land), ace2_sst_idx_std)
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
            "relative_hhe_index_box": json.dumps(_box_json(hhe_bbox)),
            "sst_era5": "ERA5 forcing surface_temperature, JJA mean, detrended, ocean/ice masked",
            "sst_ace2": f"ACE2 inference-output {model_sst_var}, JJA mean across members, detrended, ocean masked",
            "model_sst_runs_root": str(RUNS_ROOT),
            "target_definition": str(target_definition),
            "ace2_hi_inputs": str(ace2_hi_inputs),
        },
    ).to_netcdf(OUT_DIR / "sst_relhhe_jja_rankcorrbox_corr.nc")

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
            "sst_era5": "ERA5 panel uses standardized ERA5 forcing surface_temperature box index",
            "sst_ace2": f"ACE2 panel uses standardized ACE2 inference-output {model_sst_var} box index",
            "model_sst_runs_root": str(RUNS_ROOT),
            "target_definition": str(target_definition),
            "ace2_hi_inputs": str(ace2_hi_inputs),
        },
    ).to_netcdf(OUT_DIR / "sst_relhhe_jja_strongestbox_regression.nc")

    summary = {
        "years": [analysis_years[0], analysis_years[-1]],
        "n_years": len(analysis_years),
        "analysis_years": analysis_years,
        "target_definition": str(target_definition),
        "ace2_hi_inputs": str(ace2_hi_inputs),
        "relative_hhe_index_average": "cos-lat mean over US/Canada land cells inside the selected rank-correlation box",
        "selected_relative_hhe_rankcorr_box_before_expansion": _box_json(hhe_box["selected_bbox"]),
        "relative_hhe_rankcorr_box": {
            **_box_json(hhe_bbox),
            "selection_score": round(float(hhe_box["score"]), 4),
            "land_mean_tau": round(float(hhe_box["mean_tau"]), 4),
            "significant_land_mean_tau": round(float(hhe_box["mean_tau_significant"]), 4),
            "n_land_cells": int(hhe_box["n_land_cells"]),
            "n_significant_land_cells": int(hhe_box["n_significant_land_cells"]),
            "significant_land_fraction": round(float(hhe_box["significant_land_fraction"]), 4),
            "era5_idx_clim_pct": round(float(np.nanmean(era5_idx) * 100.0), 4),
            "ace2_idx_clim_pct": round(float(np.nanmean(ace2_idx) * 100.0), 4),
        },
        "highest_corr_box_era5": {
            **_box_json(sb),
            "mean_r": round(float(sst_box["mean_r"]), 4),
            "n_cells": int(sst_box["n_cells"]),
            "era5_sst_idx_detrended_std_K": round(era5_sd, 4),
            "ace2_inference_sst_idx_detrended_std_K": round(ace2_sd, 4),
            "ace2_inference_sst_variable": model_sst_var,
            "ace2_inference_sst_runs_root": str(RUNS_ROOT),
        },
    }
    (OUT_DIR / "relative_hhe_sst_rankcorr_box_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR / 'relative_hhe_sst_rankcorr_box_summary.json'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
