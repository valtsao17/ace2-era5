#!/usr/bin/env python3
"""ACE2-vs-ERA5 precipitation skill panel for the short PRATEsfc run.

The short precip run (outputs/lag_may/runs_precip_short, 1980-82 x 25 members,
6-hourly, 124 steps = 31 days = the month of May each year) is verified against
observed ERA5 total_precipitation pulled from the public ARCO-ERA5 zarr (the same
source heat_index_era5.py uses for temperature). 3 years is far too short for an
interannual Kendall tau, so this scores the *climatological* skill instead:

    Row 1  mean precip rate (mm/day)      ERA5 | ACE2 | bias (ACE2 - ERA5)
    Row 2  wet-day frequency (% >=1mm)    ERA5 | ACE2 | difference

Definitions (both internally consistent, pooled over all years/members):
  ERA5  daily total (mm) = sum of the 24 hourly total_precipitation (m) x 1000.
  ACE2  daily mean rate (mm/day) = mean of the 4 6-hourly PRATEsfc x 86400.
  mean rate       = mean of daily values over (years[, members], days).
  wet-day freq    = % of days whose daily value >= 1 mm/day.
  pattern corr r  = cos-lat-weighted spatial correlation of the two mean fields
                    (printed in the bias-panel title), domain / land / sea.

ERA5 is regridded (nearest) onto the ACE2 prediction grid and cached per year to
  outputs/lag_may/runs_precip_short/era5_precip_cache/era5_precip_daily_{Y}.nc
so re-runs / --replot are instant.

Output -> outputs/lag_may/runs_precip_short/figures/precip_short_skill_vs_era5.png
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cluster_skill_analysis_sliding7d import (  # noqa: E402
    CONUS_LAT_SLICE, CONUS_LON_SLICE, _plain_map_axes,
)
from seasonal_jja_skill import cos_lat_mean, load_land_mask, domain_scores_label  # noqa: E402
from hiro_ace_pipeline.io import subset_bbox, regrid_nearest_to_template, write_atomic  # noqa: E402

RUN_DIR   = PROJECT_ROOT / "outputs/lag_may/runs_precip_short"
FIG_DIR   = RUN_DIR / "figures"
ERA5_DIR  = RUN_DIR / "era5_precip_cache"
OUT_PNG   = FIG_DIR / "precip_short_skill_vs_era5.png"
OUT_JSON  = FIG_DIR / "precip_short_skill_vs_era5_summary.json"

ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BBOX      = (-90.0, 90.0, 0.0, 360.0)
YEARS     = [1980, 1981, 1982]
MONTH     = 5                       # the short run covers May
N_DAYS    = 31
SEC_PER_DAY = 86400.0
WET_MM    = 1.0
LA, LO    = CONUS_LAT_SLICE, CONUS_LON_SLICE


def template_grid():
    f = sorted(RUN_DIR.glob("*/member_*/autoregressive_predictions.nc"))[0]
    with xr.open_dataset(f) as ds:
        da = ds["PRATEsfc"].isel(sample=0, time=0)
        return xr.DataArray(np.zeros((da.lat.size, da.lon.size), np.float32),
                            coords={"lat": da.lat.values, "lon": da.lon.values},
                            dims=("lat", "lon"))


# ── ERA5 observed daily precip totals (mm) ───────────────────────────────────

def era5_daily_precip(year, template, force=False):
    """ERA5 May daily total precip (mm) on the ACE2 grid -> (31, lat, lon)."""
    path = ERA5_DIR / f"era5_precip_daily_{year}.nc"
    if path.exists() and not force:
        with xr.open_dataset(path) as ds:
            return ds["precip_mm"].values.astype(np.float32)

    start = f"{year}-{MONTH:02d}-01"
    end   = f"{year}-{MONTH:02d}-{N_DAYS:02d}T23:00:00"
    print(f"  [{year}] opening zarr + selecting May hourly tp ...", flush=True)
    ds = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={}, storage_options={"token": "anon"})
    tp = subset_bbox(ds["total_precipitation"].sel(time=slice(start, end)), BBOX)
    print(f"  [{year}] regridding nearest -> ACE2 grid + loading ...", flush=True)
    tp = regrid_nearest_to_template(tp, template).load()
    ds.close()

    nh, nlat, nlon = tp.shape
    assert nh == N_DAYS * 24, f"expected {N_DAYS*24} hours, got {nh} for {year}"
    daily_mm = (tp.values.reshape(N_DAYS, 24, nlat, nlon).sum(axis=1) * 1000.0).astype(np.float32)
    out = xr.Dataset(
        {"precip_mm": (("day", "lat", "lon"), daily_mm)},
        coords={"day": np.arange(1, N_DAYS + 1),
                "lat": template.lat.values, "lon": template.lon.values},
        attrs={"long_name": f"ERA5 daily total precip (mm), {year}-{MONTH:02d}, ACE2 grid"},
    )
    write_atomic(out, path)
    print(f"  wrote {path}", flush=True)
    return daily_mm


# ── ACE2 daily-mean precip rate (mm/day) ─────────────────────────────────────

def ace2_daily_precip(year):
    """ACE2 May daily-mean precip rate (mm/day) -> (n_members, 31, lat, lon)."""
    files = sorted((RUN_DIR / str(year)).glob("member_*/autoregressive_predictions.nc"))
    out = []
    for f in files:
        with xr.open_dataset(f) as ds:
            arr = ds["PRATEsfc"].isel(sample=0).values.astype(np.float32) * SEC_PER_DAY
        nt, nlat, nlon = arr.shape
        nd = nt // 4
        out.append(arr[:nd * 4].reshape(nd, 4, nlat, nlon).mean(axis=1))
    return np.stack(out, axis=0)


def pattern_corr(a, b, lat, mask=None):
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(a)
    good = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        good = good & mask
    if good.sum() < 3:
        return np.nan
    aw, bw, ww = a[good], b[good], w[good]
    am = np.average(aw, weights=ww); bm = np.average(bw, weights=ww)
    cov = np.average((aw - am) * (bw - bm), weights=ww)
    va = np.average((aw - am) ** 2, weights=ww); vb = np.average((bw - bm) ** 2, weights=ww)
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


def corr_label(a, b, lat, land):
    rall = pattern_corr(a, b, lat)
    rlnd = pattern_corr(a, b, lat, land)
    rsea = pattern_corr(a, b, lat, ~land)
    return f"pattern r = {rall:.3f}  (land {rlnd:.3f}, sea {rsea:.3f})"


# ── compute pooled fields ────────────────────────────────────────────────────

def compute(force=False):
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    template = template_grid()
    lat = template.lat.values.astype(np.float32)
    lon = template.lon.values.astype(np.float32)

    era5_days, ace2_days = [], []
    for Y in YEARS:
        print(f"ERA5 {Y} ...", flush=True)
        era5_days.append(era5_daily_precip(Y, template, force=force))      # (31, lat, lon)
        print(f"ACE2 {Y} ...", flush=True)
        ace2_days.append(ace2_daily_precip(Y))                            # (mem, 31, lat, lon)

    era5 = np.concatenate(era5_days, axis=0)                              # (3*31, lat, lon)
    ace2 = np.concatenate(ace2_days, axis=1)                             # (mem, 3*31, lat, lon)

    fields = {
        "era5_mean":     np.nanmean(era5, axis=0),
        "ace2_mean":     np.nanmean(ace2, axis=(0, 1)),
        "era5_wetfreq":  np.nanmean(era5 >= WET_MM, axis=0) * 100.0,
        "ace2_wetfreq":  np.nanmean(ace2 >= WET_MM, axis=(0, 1)) * 100.0,
    }
    fields = {k: v.astype(np.float32) for k, v in fields.items()}
    fields["mean_bias"]    = fields["ace2_mean"] - fields["era5_mean"]
    fields["wetfreq_diff"] = fields["ace2_wetfreq"] - fields["era5_wetfreq"]
    return fields, lat, lon


# ── rendering ────────────────────────────────────────────────────────────────

def _panel(ax, cax, fig, field, lat, lon, title, cmap, vmin, vmax, clabel, legend,
           show_x):
    lon_plot = lon - 360.0 if float(np.nanmean(lon)) > 180 else lon
    extent = [float(lon_plot[0]) - 0.5, float(lon_plot[-1]) + 0.5,
              float(lat[0]) - 0.5, float(lat[-1]) + 0.5]
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    _plain_map_axes(ax, lon, lat, pad=0.0)
    if not show_x:
        ax.set_xlabel(""); ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(title, fontsize=10, pad=3)
    if legend is not None:
        ax.legend([Line2D([], [], linestyle="none")], [legend], loc="lower left",
                  fontsize=7.5, handlelength=0, handletextpad=0, framealpha=0.95,
                  borderpad=0.4).set_zorder(7)
    cb = fig.colorbar(im, cax=cax); cb.set_label(clabel, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def sym(field):
    v = field[np.isfinite(field)]
    return float(np.nanpercentile(np.abs(v), 98)) if v.size else 1.0


def render(fields, lat, lon):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cf = {k: v[LA, LO] for k, v in fields.items()}
    latc, lonc = lat[LA], lon[LO]
    land = load_land_mask(latc, lonc)

    rate = plt.get_cmap("YlGnBu").copy();  rate.set_bad("white")
    freq = plt.get_cmap("PuBuGn").copy();  freq.set_bad("white")
    div  = plt.get_cmap("RdBu_r").copy();  div.set_bad("white")

    rmax = max(sym(cf["era5_mean"]), sym(cf["ace2_mean"]))
    fmax = max(sym(cf["era5_wetfreq"]), sym(cf["ace2_wetfreq"]))
    bmax = sym(cf["mean_bias"]); dmax = sym(cf["wetfreq_diff"])

    r_mean = corr_label(cf["ace2_mean"], cf["era5_mean"], latc, land)
    r_wet  = corr_label(cf["ace2_wetfreq"], cf["era5_wetfreq"], latc, land)

    specs = [
        ("era5_mean",    "ERA5 mean rate",        rate, 0.0, rmax, "mm/day",
         domain_scores_label("domain", cf["era5_mean"], latc, land, "{:.2f}") + " mm/day"),
        ("ace2_mean",    "ACE2 mean rate",        rate, 0.0, rmax, "mm/day",
         domain_scores_label("domain", cf["ace2_mean"], latc, land, "{:.2f}") + " mm/day"),
        ("mean_bias",    f"Bias (ACE2 - ERA5)  |  {r_mean}", div, -bmax, bmax, "mm/day",
         domain_scores_label("domain", cf["mean_bias"], latc, land, "{:+.2f}") + " mm/day"),
        ("era5_wetfreq", "ERA5 wet-day freq",     freq, 0.0, fmax, "% days >=1mm",
         domain_scores_label("domain", cf["era5_wetfreq"], latc, land, "{:.1f}") + "%"),
        ("ace2_wetfreq", "ACE2 wet-day freq",     freq, 0.0, fmax, "% days >=1mm",
         domain_scores_label("domain", cf["ace2_wetfreq"], latc, land, "{:.1f}") + "%"),
        ("wetfreq_diff", f"Difference (ACE2 - ERA5)  |  {r_wet}", div, -dmax, dmax, "% pts",
         domain_scores_label("domain", cf["wetfreq_diff"], latc, land, "{:+.1f}") + "%"),
    ]

    fig = plt.figure(figsize=(18.5, 7.4), facecolor="white")
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 0.035, 1, 0.035, 1, 0.035],
                          left=0.035, right=0.975, bottom=0.07, top=0.88,
                          wspace=0.08, hspace=0.22)
    for idx, (key, title, cmap, vmin, vmax, clab, leg) in enumerate(specs):
        r, c = divmod(idx, 3)
        ax  = fig.add_subplot(gs[r, c * 2])
        cax = fig.add_subplot(gs[r, c * 2 + 1])
        _panel(ax, cax, fig, cf[key], latc, lonc, title, cmap, vmin, vmax, clab, leg,
               show_x=(r == 1))

    fig.suptitle("ACE2 vs ERA5 short precipitation skill — May 1980-82, 25 members, "
                 "climatological (stipple-free; 3 yrs too short for interannual τ)",
                 fontsize=13, y=0.965)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "run_dir": str(RUN_DIR), "era5_source": ERA5_ZARR, "years": YEARS,
        "month": MONTH, "wet_threshold_mm_day": WET_MM,
        "domain_means": {k: float(cos_lat_mean(cf[k], latc)) for k in cf},
        "pattern_corr_mean_rate": r_mean, "pattern_corr_wetfreq": r_wet,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PNG}", flush=True)
    print(f"  mean rate  ERA5={cos_lat_mean(cf['era5_mean'], latc):.2f}  "
          f"ACE2={cos_lat_mean(cf['ace2_mean'], latc):.2f} mm/day", flush=True)
    print(f"  wet-day    ERA5={cos_lat_mean(cf['era5_wetfreq'], latc):.1f}  "
          f"ACE2={cos_lat_mean(cf['ace2_wetfreq'], latc):.1f} %", flush=True)
    print(f"  {r_mean}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch ERA5 even if cached")
    ap.add_argument("--replot", action="store_true", help="reuse ERA5 cache, just recompute+render")
    args = ap.parse_args()
    fields, lat, lon = compute(force=args.force)
    render(fields, lat, lon)


if __name__ == "__main__":
    main()
