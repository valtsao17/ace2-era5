#!/usr/bin/env python3
"""Lagged SST–HHE teleconnection on the TRUE humid-heat extreme, with a
data-driven southeastern-US index box.

Two tasks in one pass:

  (1) LAGGED SST  -> assess predictability lead time.  Pre-JJA SST is averaged
      over cumulative windows ending 31 May and correlated with the JJA HHE
      index:
          lag0   contemporaneous  Jun 1 - Aug 31  (the usual teleconnection)
          lag30  May 1  - May 31
          lag60  Apr 1  - May 31
          lag90  Mar 1  - May 31
      SST is ERA5 surface_temperature from the ACE2 forcing files
      (data/lag_data/forcing_data_ace2era5/forcing_<yr>.nc) — already on the
      1deg HHE grid, ocean-masked (ocean_fraction>0.5, sea-ice excluded).

  (2) DATA-DRIVEN SE-US BOX  -> instead of a hand-drawn SEUS box, slide a box
      over a southeastern-US search window and pick the position that maximizes
      the grid-cell rank correlation (Kendall tau) between ACE2 and ERA5 true
      HHE frequency — i.e. the SE-US region where ACE2 most faithfully tracks
      ERA5's interannual HHE variability.  The HHE index is the cos-lat mean
      over that box.

True HHE freq  : outputs/lag_may/heat_index_era5/jja_hi_freq_{era5,ace2}.nc
Outputs        : outputs/lag_may/sst_teleconnection_hhe_lagged/
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# reuse the proven helpers from the contemporaneous SST script
from sst_teleconnection_jja_sliding7d import (   # noqa: E402
    bbox_mean, detrend_1d, detrend_along_year, pearson_corr_map,
    _render_corr_ax, _draw_coast_and_borders, TNA_BBOX,
)

HHE_DIR     = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
FORCING_DIR = PROJECT_ROOT / "data/lag_data/forcing_data_ace2era5"
OUT_DIR     = PROJECT_ROOT / "outputs/lag_may/sst_teleconnection_hhe_lagged"
FIG_DIR     = OUT_DIR / "figures"

YEARS = list(range(1980, 2017))

# cumulative pre-JJA windows (month-day start, end) + contemporaneous reference
LAGS = {
    "lag0_JJA":   ("06-01", "08-31"),   # contemporaneous summer SST
    "lag30_May":  ("05-01", "05-31"),   # 1 month lead
    "lag60_AprMay": ("04-01", "05-31"), # 2 month lead (cumulative)
    "lag90_MarMay": ("03-01", "05-31"), # 3 month lead (cumulative)
}

# southeastern-US search window for the data-driven box (0-360 lon)
SEUS_WINDOW = dict(lat_s=24.0, lat_n=40.0, lon_w=255.0, lon_e=292.0)
BOX_LAT_SPANS = [6.0, 8.0, 10.0]
BOX_LON_SPANS = [8.0, 12.0, 16.0]
FREQ_MIN = 0.002          # box cells must actually see HHE (ERA5 clim mean)
MIN_CELLS = 8


# ────────────────────────────────────────────────────────────────────────────
# SST from forcing files
# ────────────────────────────────────────────────────────────────────────────
def build_lagged_sst():
    """ERA5 SST (forcing surface_temperature) window-means per lag, ocean-masked.
    Returns dict lag -> (37, lat, lon), plus lat, lon."""
    cache = OUT_DIR / "lagged_sst.nc"
    if cache.exists():
        print(f"Reusing {cache}", flush=True)
        ds = xr.open_dataset(cache)
        lat, lon = ds["lat"].values, ds["lon"].values
        out = {lag: ds[lag].values.astype(np.float32) for lag in LAGS}
        ds.close()
        return out, lat, lon

    acc = {lag: [] for lag in LAGS}
    ice_acc = {lag: [] for lag in LAGS}
    lat = lon = None
    ocn_static = None
    for y in YEARS:
        f = FORCING_DIR / f"forcing_{y}.nc"
        d = xr.open_dataset(f)
        if lat is None:
            lat = d["latitude"].values; lon = d["longitude"].values
        ocn = d["ocean_fraction"]
        ocn = (ocn.isel(time=0) if "time" in ocn.dims else ocn).values > 0.5
        if ocn_static is None:
            ocn_static = ocn               # geographic, time-invariant
        st = d["surface_temperature"]
        ice = d["sea_ice_fraction"]
        for lag, (a, b) in LAGS.items():
            win = st.sel(time=slice(f"{y}-{a}", f"{y}-{b}")).mean("time").values
            icew = ice.sel(time=slice(f"{y}-{a}", f"{y}-{b}")).mean("time").values
            acc[lag].append(np.where(ocn_static, win, np.nan).astype(np.float32))
            ice_acc[lag].append(icew.astype(np.float32))
        d.close()
        print(f"  SST windows {y} done", flush=True)

    # STATIC ice mask: drop cells that are climatologically sea-ice in the window
    # (uniform across years -> every cell series is all-finite or all-NaN, so the
    # downstream linear detrend never sees a partial-NaN column).
    out = {}
    for lag in LAGS:
        arr = np.stack(acc[lag], 0)
        clim_ice = np.nanmean(np.stack(ice_acc[lag], 0), axis=0)
        bad = clim_ice > 0.15
        arr[:, bad] = np.nan
        out[lag] = arr
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xr.Dataset({lag: (("year", "lat", "lon"), out[lag]) for lag in LAGS},
               coords={"year": YEARS, "lat": lat, "lon": lon}).to_netcdf(cache)
    print(f"wrote {cache}", flush=True)
    return out, lat, lon


# ────────────────────────────────────────────────────────────────────────────
# grid-cell Kendall tau (ACE2 vs ERA5 HHE) + data-driven SE-US box
# ────────────────────────────────────────────────────────────────────────────
def kendall_skill_map(ace2, era5):
    """Per-cell Kendall tau between two (37, lat, lon) HHE-freq fields."""
    ny, nlat, nlon = ace2.shape
    tau = np.full((nlat, nlon), np.nan, np.float32)
    for i in range(nlat):
        for j in range(nlon):
            a = ace2[:, i, j]; e = era5[:, i, j]
            m = np.isfinite(a) & np.isfinite(e)
            if m.sum() < 10 or np.all(a[m] == a[m][0]) or np.all(e[m] == e[m][0]):
                continue
            t, _ = stats.kendalltau(a[m], e[m])
            tau[i, j] = t
    return tau


def find_box(tau, era5_clim, lat, lon):
    """Slide candidate boxes over the SE-US window; pick the one maximizing mean
    Kendall tau over land/HHE cells. Returns bbox + diagnostics."""
    w = SEUS_WINDOW
    in_lat = (lat >= w["lat_s"]) & (lat <= w["lat_n"])
    in_lon = (lon >= w["lon_w"]) & (lon <= w["lon_e"])
    cand_lat = lat[in_lat]; cand_lon = lon[in_lon]

    best = None
    for ls in BOX_LAT_SPANS:
        for lo in BOX_LON_SPANS:
            for la0 in cand_lat:
                for lo0 in cand_lon:
                    la1, lo1 = la0 + ls, lo0 + lo
                    if la1 > w["lat_n"] or lo1 > w["lon_e"]:
                        continue
                    sel_la = (lat >= la0) & (lat <= la1)
                    sel_lo = (lon >= lo0) & (lon <= lo1)
                    t = tau[np.ix_(sel_la, sel_lo)]
                    c = era5_clim[np.ix_(sel_la, sel_lo)]
                    valid = np.isfinite(t) & np.isfinite(c) & (c > FREQ_MIN)
                    if valid.sum() < MIN_CELLS:
                        continue
                    score = float(np.nanmean(t[valid]))
                    if best is None or score > best["mean_tau"]:
                        best = dict(bbox=(float(la0), float(la1), float(lo0), float(lo1)),
                                    mean_tau=score, n_cells=int(valid.sum()),
                                    lat_span=ls, lon_span=lo)
    return best


# ────────────────────────────────────────────────────────────────────────────
# figures
# ────────────────────────────────────────────────────────────────────────────
def plot_box_on_skill(tau, lat, lon, box, out_png):
    """CONUS Kendall-tau (ACE2 vs ERA5 HHE) map with the chosen index box."""
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180)
    lon_s = lon180[order]; tau_s = tau[:, order]
    LON, LAT = np.meshgrid(lon_s, lat)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(LON, LAT, tau_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                         shading="nearest", zorder=1)
    ax.set_xlim(-128, -65); ax.set_ylim(22, 50); ax.set_aspect(1.0)
    _draw_coast_and_borders(ax, (-128, -65), (22, 50))

    la0, la1, lo0, lo1 = box["bbox"]
    x0 = lo0 - 360 if lo0 > 180 else lo0
    x1 = lo1 - 360 if lo1 > 180 else lo1
    ax.add_patch(plt.Rectangle((x0, la0), x1 - x0, la1 - la0, fill=False,
                               edgecolor="lime", lw=2.5, zorder=6))
    ax.text(x0, la1 + 0.4, f"data-driven SE-US box  (mean tau={box['mean_tau']:.2f})",
            color="green", fontsize=9, weight="bold")
    fig.colorbar(mesh, ax=ax, shrink=0.85, label="Kendall tau  (ACE2 vs ERA5 HHE freq)")
    ax.set_title("SE-US HHE index box from maximum ACE2–ERA5 rank correlation",
                 fontsize=11)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def plot_lag_panels(corr_by_lag, pval_by_lag, lat, lon, index_name, out_png):
    """One row of SST-correlation maps, one panel per lag (predictability lead)."""
    lags = list(LAGS.keys())
    fig, axes = plt.subplots(1, len(lags), figsize=(5.2 * len(lags), 5))
    for ax, lag in zip(axes, lags):
        mesh = _render_corr_ax(ax, corr_by_lag[lag], pval_by_lag[lag], lat, lon,
                               lag.replace("_", "  "))
    fig.colorbar(mesh, ax=axes, shrink=0.8, orientation="vertical", label="Pearson r")
    fig.suptitle(f"Pre-JJA SST  ×  {index_name} JJA HHE index  |  predictability lead time"
                 f"  |  stipple p<0.05", fontsize=12, y=1.02)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


def plot_lead_curve(tna_curves, out_png):
    """TNA-box SST × HHE-index correlation vs lead time, ERA5 & ACE2."""
    leads = [0, 30, 60, 90]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (vals, col) in tna_curves.items():
        ax.plot(leads, vals, "-o", color=col, lw=2, label=name)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(leads)
    ax.set_xlabel("SST lead time (days before 1 Jun)")
    ax.set_ylabel("Pearson r  (TNA SST × SE-US HHE index)")
    ax.set_title("Predictability lead time: TNA spring SST → summer HHE", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}", flush=True)


# ────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- true HHE freq ---
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_era5.nc") as d:
        lat = d["lat"].values; lon = d["lon"].values
        era5 = d["era5_hi_freq"].values.astype(np.float32)
    with xr.open_dataset(HHE_DIR / "jja_hi_freq_ace2.nc") as d:
        ace2 = d["ace2_hi_freq"].values.astype(np.float32)
    era5_clim = np.nanmean(era5, axis=0)

    # --- (2) data-driven SE-US box from ACE2-vs-ERA5 rank correlation ---
    print("Computing ACE2–ERA5 HHE Kendall-tau skill map ...", flush=True)
    tau = kendall_skill_map(ace2, era5)
    box = find_box(tau, era5_clim, lat, lon)
    if box is None:
        raise RuntimeError("No valid SE-US box found — relax FREQ_MIN/MIN_CELLS")
    la0, la1, lo0, lo1 = box["bbox"]
    print(f"  data-driven box: lat {la0:.1f}-{la1:.1f}  lon {lo0:.1f}-{lo1:.1f}  "
          f"mean_tau={box['mean_tau']:.3f}  n={box['n_cells']}", flush=True)
    plot_box_on_skill(tau, lat, lon, box, FIG_DIR / "seus_box_from_rankcorr.png")

    box_bbox = (la0, la1, lo0, lo1)
    era5_idx = bbox_mean(era5, lat, lon, box_bbox)
    ace2_idx = bbox_mean(ace2, lat, lon, box_bbox)
    print(f"  ERA5 SE-US HHE index mean={np.nanmean(era5_idx):.4f}  "
          f"ACE2 mean={np.nanmean(ace2_idx):.4f}", flush=True)

    # --- (1) lagged SST ---
    sst, slat, slon = build_lagged_sst()

    corr_e = {}; pval_e = {}; corr_a = {}; pval_a = {}
    tna_e = []; tna_a = []
    for lag in LAGS:
        sst_dt = detrend_along_year(sst[lag])
        corr_e[lag], pval_e[lag] = pearson_corr_map(sst_dt, era5_idx)
        corr_a[lag], pval_a[lag] = pearson_corr_map(sst_dt, ace2_idx)
        # TNA-box SST index correlation with each HHE index (lead-time curve)
        tna = detrend_1d(bbox_mean(sst[lag], slat, slon, TNA_BBOX))
        re = stats.pearsonr(tna, detrend_1d(era5_idx))[0]
        ra = stats.pearsonr(tna, detrend_1d(ace2_idx))[0]
        tna_e.append(re); tna_a.append(ra)
        print(f"  {lag:14s}  TNA-SST×ERA5idx r={re:+.3f}   TNA-SST×ACE2idx r={ra:+.3f}",
              flush=True)

    # save corr fields
    xr.Dataset({**{f"corr_era5_{k}": (("lat", "lon"), corr_e[k]) for k in LAGS},
                **{f"corr_ace2_{k}": (("lat", "lon"), corr_a[k]) for k in LAGS}},
               coords={"lat": slat, "lon": slon}).to_netcdf(OUT_DIR / "lagged_sst_corr.nc")

    plot_lag_panels(corr_e, pval_e, slat, slon, "ERA5", FIG_DIR / "sst_lag_corr_era5.png")
    plot_lag_panels(corr_a, pval_a, slat, slon, "ACE2", FIG_DIR / "sst_lag_corr_ace2.png")
    plot_lead_curve({"ERA5 HHE index": (tna_e, "#1f5fd0"),
                     "ACE2 HHE index": (tna_a, "#e8202a")},
                    FIG_DIR / "predictability_lead_curve.png")

    # summary json
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "hhe_definition": "NOAA HI>=105F from daily Tmax + RHmin (true HHE)",
        "data_driven_seus_box": {"lat_s": la0, "lat_n": la1, "lon_w": lo0, "lon_e": lo1,
                                 "mean_kendall_tau": box["mean_tau"], "n_cells": box["n_cells"]},
        "lags": {k: list(v) for k, v in LAGS.items()},
        "tna_sst_x_era5_idx_by_lead": {l: round(float(r), 4) for l, r in zip(LAGS, tna_e)},
        "tna_sst_x_ace2_idx_by_lead": {l: round(float(r), 4) for l, r in zip(LAGS, tna_a)},
        "era5_idx_clim": round(float(np.nanmean(era5_idx)), 5),
        "ace2_idx_clim": round(float(np.nanmean(ace2_idx)), 5),
    }, indent=2))
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
