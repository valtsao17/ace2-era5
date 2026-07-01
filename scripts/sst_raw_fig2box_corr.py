#!/usr/bin/env python3
"""SST correlation maps for monthly RAW heat-extreme hotspot boxes.

Companion to sst_hhe_fig2box_corr.py. For each raw heat-extreme month
(June, July, August), the index box is the same ERA5-defined highest-
concentration 8x12 degree box drawn on raw_frequency_{jun,jul,aug}.png.

For each month:
  left  : same-month SST x ERA5 raw-extreme frequency index
  right : same-month SST x ACE2 raw-extreme frequency index

The ERA5 and ACE2 indices both use the same month-specific ERA5 hotspot box.
Stippling marks p < 0.05. The ERA5-side green SST box marks the strongest
positive mean-correlation region; significance is not required for box
placement.

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

from extreme_freq_boxed_panels import find_box as find_hotspot_box
from sst_hhe_fig2box_corr import CORR_BOX_LAT, CORR_BOX_LON, SEARCH
from sst_teleconnection_hhe_lagged import OUT_DIR, FIG_DIR, FORCING_DIR, YEARS
from sst_teleconnection_jja_sliding7d import (
    bbox_mean, detrend_along_year, pearson_corr_map, _render_corr_ax,
)

RAW_NPZ = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall/monthly_raw_rc_fields.npz"
MONTHS = {6: "June", 7: "July", 8: "August"}
TAG = {6: "jun", 7: "jul", 8: "aug"}
SST_VAR = {"jun": "sst_jun", "jul": "sst_jul", "aug": "sst_aug"}
SST_WINDOWS = {"sst_jun": ("06-01", "06-30"),
               "sst_jul": ("07-01", "07-31"),
               "sst_aug": ("08-01", "08-31")}


def build_monthly_sst():
    """ERA5 forcing SST month means for Jun/Jul/Aug, ocean/ice masked."""
    cache = OUT_DIR / "monthly_sst.nc"
    if cache.exists():
        print(f"Reusing {cache}", flush=True)
        ds = xr.open_dataset(cache)
        lat, lon = ds["lat"].values, ds["lon"].values
        out = {name: ds[name].values.astype(np.float32) for name in SST_WINDOWS}
        ds.close()
        return out, lat, lon

    acc = {name: [] for name in SST_WINDOWS}
    ice_acc = {name: [] for name in SST_WINDOWS}
    lat = lon = None
    ocn_static = None
    for year in YEARS:
        with xr.open_dataset(FORCING_DIR / f"forcing_{year}.nc") as d:
            if lat is None:
                lat = d["latitude"].values
                lon = d["longitude"].values
            ocn = d["ocean_fraction"]
            ocn = (ocn.isel(time=0) if "time" in ocn.dims else ocn).values > 0.5
            if ocn_static is None:
                ocn_static = ocn
            st = d["surface_temperature"]
            ice = d["sea_ice_fraction"]
            for name, (start, end) in SST_WINDOWS.items():
                win = st.sel(time=slice(f"{year}-{start}", f"{year}-{end}")).mean("time").values
                icew = ice.sel(time=slice(f"{year}-{start}", f"{year}-{end}")).mean("time").values
                acc[name].append(np.where(ocn_static, win, np.nan).astype(np.float32))
                ice_acc[name].append(icew.astype(np.float32))
        print(f"  monthly SST {year} done", flush=True)

    out = {}
    for name in SST_WINDOWS:
        arr = np.stack(acc[name], axis=0)
        clim_ice = np.nanmean(np.stack(ice_acc[name], axis=0), axis=0)
        arr[:, clim_ice > 0.15] = np.nan
        out[name] = arr.astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {name: (("year", "lat", "lon"), arr) for name, arr in out.items()},
        coords={"year": YEARS, "lat": lat, "lon": lon},
    ).to_netcdf(cache)
    print(f"wrote {cache}", flush=True)
    return out, lat, lon


def _draw_corr_box(ax, box, label):
    if box is None:
        return
    la0, la1, lo0, lo1 = box
    x0 = lo0 - 360.0 if lo0 > 180 else lo0
    x1 = lo1 - 360.0 if lo1 > 180 else lo1
    ax.add_patch(plt.Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="#00b050", lw=2.4, zorder=7))
    ax.text(x0, la0 - 1.5, label, color="#00b050", fontsize=9, weight="bold",
            va="top", ha="left", zorder=8)


def _box_json(box):
    if box is None:
        return None
    la0, la1, lo0, lo1 = box
    return {
        "latN": [round(float(la0), 1), round(float(la1), 1)],
        "lonW": [round(float(360.0 - lo1), 1), round(float(360.0 - lo0), 1)],
    }


def _corr_box_json(box, score, n_sig):
    out = _box_json(box)
    if out is None:
        return None
    out["mean_r"] = round(float(score), 3)
    out["n_cells"] = int(n_sig)
    return out


def find_positive_corr_box(corr, lat, lon):
    """Find the strongest positive mean-correlation SST box, ignoring p-values."""
    w = SEARCH
    cand_lat = lat[(lat >= w["lat_s"]) & (lat <= w["lat_n"])]
    cand_lon = lon[(lon >= w["lon_w"]) & (lon <= w["lon_e"])]
    best = None
    for la0 in cand_lat:
        la1 = la0 + CORR_BOX_LAT
        if la1 > w["lat_n"]:
            continue
        sla = (lat >= la0) & (lat <= la1)
        for lo0 in cand_lon:
            lo1 = lo0 + CORR_BOX_LON
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
            n_cells = int(finite.sum())
            if best is None or mean_r > best[0]:
                best = (mean_r, n_cells, (float(la0), float(la1), float(lo0), float(lo1)))
    if best is None:
        return None, None, 0
    return best[2], best[0], best[1]


def render_month(month, fe, fa, lat, lon, sst_field, slat, slon, summaries, ds_vars):
    tag = TAG[month]
    name = MONTHS[month]

    era5_clim_pct = np.nanmean(fe, axis=0) * 100.0
    index_box, index_score = find_hotspot_box(era5_clim_pct, lat, lon)
    la0, la1, lo0, lo1 = index_box
    print(f"{name} raw index box: lat {la0:.1f}-{la1:.1f}N  "
          f"lon {360-lo1:.0f}-{360-lo0:.0f}degW  "
          f"ERA5 mean={index_score:.2f}%", flush=True)

    era5_idx = bbox_mean(fe, lat, lon, index_box)
    ace2_idx = bbox_mean(fa, lat, lon, index_box)
    print(f"  raw-extreme index clim: ERA5={np.nanmean(era5_idx) * 100.0:.2f}%  "
          f"ACE2={np.nanmean(ace2_idx) * 100.0:.2f}%", flush=True)

    sst_dt = detrend_along_year(sst_field)

    print(f"  correlating {name} SST x ERA5 {name} raw index ...", flush=True)
    corr_e, pval_e = pearson_corr_map(sst_dt, era5_idx)
    print(f"  correlating {name} SST x ACE2 {name} raw index ...", flush=True)
    corr_a, pval_a = pearson_corr_map(sst_dt, ace2_idx)

    box_e, se, ne = find_positive_corr_box(corr_e, slat, slon)
    if box_e is not None:
        print(f"  strongest +corr box ERA5: lon {360-box_e[3]:.0f}-{360-box_e[2]:.0f}degW "
              f"lat {box_e[0]:.0f}-{box_e[1]:.0f}N  n_cells={ne}  mean r={se:.2f}",
              flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
    mesh = _render_corr_ax(
        axes[0], corr_e, pval_e, slat, slon,
        f"RAW {name} index: ERA5   ({name} SST x ERA5 raw freq)",
    )
    _draw_corr_box(axes[0], box_e, f"r={se:.2f}" if se is not None else "")
    mesh = _render_corr_ax(
        axes[1], corr_a, pval_a, slat, slon,
        f"RAW {name} index: ACE2   ({name} SST x ACE2 raw freq)",
    )
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical",
                 label="Pearson r  (detrended same-month SST x raw-extreme freq index)")
    fig.suptitle(
        f"{name} SST vs RAW {name} heat-extreme frequency over ERA5 hotspot box  |  "
        "stipple p<0.05, ERA5 green box = strongest positive mean correlation",
        fontsize=12,
    )
    out = FIG_DIR / f"sst_raw_fig2box_corr_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    ds_vars[f"corr_era5_{tag}"] = (("lat", "lon"), corr_e)
    ds_vars[f"pval_era5_{tag}"] = (("lat", "lon"), pval_e)
    ds_vars[f"corr_ace2_{tag}"] = (("lat", "lon"), corr_a)
    ds_vars[f"pval_ace2_{tag}"] = (("lat", "lon"), pval_a)
    summaries[tag] = {
        "raw_index_box": {
            **_box_json(index_box),
            "era5_box_mean_pct": round(float(index_score), 3),
        },
        "era5_idx_clim_pct": round(float(np.nanmean(era5_idx) * 100.0), 4),
        "ace2_idx_clim_pct": round(float(np.nanmean(ace2_idx) * 100.0), 4),
        "highest_corr_box_era5": _corr_box_json(box_e, se, ne),
        "highest_corr_box_ace2": None,
    }


def main():
    if not RAW_NPZ.exists():
        raise FileNotFoundError(f"Missing {RAW_NPZ}. Run monthly_raw_freq_rankcorr.py first.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(RAW_NPZ)
    lat = raw["lat"].astype(np.float32)
    lon = raw["lon"].astype(np.float32)

    sst, slat, slon = build_monthly_sst()

    summaries = {}
    ds_vars = {}
    for month in (6, 7, 8):
        render_month(
            month,
            raw[f"fe{month}"].astype(np.float32),
            raw[f"fa{month}"].astype(np.float32),
            lat,
            lon,
            sst[SST_VAR[TAG[month]]],
            slat,
            slon,
            summaries,
            ds_vars,
        )

    xr.Dataset(
        ds_vars,
        coords={"lat": slat, "lon": slon},
        attrs={
            "sst": "ERA5 forcing surface_temperature, same-month mean, detrended",
            "raw_index": "Monthly raw Tmax > full-JJA 90th percentile frequency",
            "index_box": "Month-specific 8x12 degree ERA5 raw-frequency hotspot box",
        },
    ).to_netcdf(OUT_DIR / "sst_raw_fig2box_corr.nc")
    (OUT_DIR / "sst_raw_fig2box_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"wrote {OUT_DIR / 'sst_raw_fig2box_corr.nc'}", flush=True)
    print(f"wrote {OUT_DIR / 'sst_raw_fig2box_summary.json'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
