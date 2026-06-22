#!/usr/bin/env python3
"""Grid-point modified-Kendall z skill maps (parallel to the Kendall-τ maps).

Recomputes the per-grid-point JJA HHE seasonal-frequency skill using the
modified-Kendall z statistic (Zheng & Lo 2006, top-k weighted) instead of
scipy Kendall τ, from the same 37-year frequency arrays in
seasonal_jja_sliding7d/jja_seasonal_freqs.nc.

Writes (to outputs/lag_may/seasonal_jja_sliding7d_modkendall/):
  skill_jja_seasonal.nc          — variable `kendall_tau` holds the z map, so
                                    the cluster script's grid-point reference
                                    loader picks it up unchanged.
  tau_jja_seasonal_global_modkendall.png
  tau_jja_seasonal_conus_modkendall.png

Caveat: raw seasonal frequencies are integer counts with many ties; the modified
Kendall's analytic null assumes tie-free permutations, so grid-point z is biased
high relative to the (tie-free) cluster-aggregated z. Interpret the map as a
top-weighted agreement score, not a bounded correlation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mod_kendall_metric import mk_z, DEFAULT_K
from cluster_skill_analysis_sliding7d import (
    _single_map_figure, _TAU_CMAP, CONUS_LAT_SLICE, CONUS_LON_SLICE,
)

SLIDING_DIR = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d"
OUT_DIR     = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d_modkendall"


def _z_one_row(pred_row, obs_row, k):
    nlon = pred_row.shape[1]
    out = np.full(nlon, np.nan, dtype=np.float32)
    for j in range(nlon):
        out[j] = mk_z(pred_row[:, j], obs_row[:, j], k)
    return out


def mk_z_map(pred, obs, k):
    """Per-grid-point modified-Kendall z. pred/obs: (n_years, nlat, nlon)."""
    n_lat = pred.shape[1]
    rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_z_one_row)(pred[:, i, :], obs[:, i, :], k) for i in range(n_lat)
    )
    return np.stack(rows, axis=0)


def cos_lat_mean(field, lat):
    w = np.cos(np.deg2rad(lat))[:, None]
    v = np.isfinite(field)
    return float(np.nansum(field * w * v) / np.nansum(w * v))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--metric-k", type=int, default=DEFAULT_K)
    args = p.parse_args()
    k = args.metric_k
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    freq_nc = SLIDING_DIR / "jja_seasonal_freqs.nc"
    print(f"Loading {freq_nc}", flush=True)
    ds = xr.open_dataset(freq_nc)
    lat = ds["lat"].values
    lon = ds["lon"].values
    pred = ds["ace2_freq"].values.astype(np.float32)
    obs  = ds["era5_freq"].values.astype(np.float32)
    ds.close()

    print(f"Computing grid-point modified-Kendall z (k={k}) ...", flush=True)
    z_map = mk_z_map(pred, obs, k)
    zref = cos_lat_mean(z_map, lat)
    print(f"  global cos-lat mean z = {zref:.4f}", flush=True)

    # Save with variable name `kendall_tau` so the cluster script's grid-point
    # reference loader (ds["kendall_tau"]) consumes it without changes.
    da = xr.DataArray(z_map, dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
                      attrs={"long_name": f"modified-Kendall z (k={k})", "truncation_k": k})
    out_nc = OUT_DIR / "skill_jja_seasonal.nc"
    xr.Dataset({"kendall_tau": da, "mod_kendall_z": da}).to_netcdf(out_nc)
    print(f"wrote {out_nc}", flush=True)

    # symmetric, data-driven color scale (z is unbounded)
    finite = z_map[np.isfinite(z_map)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 4.0
    label = f"mod-Kendall z (k={k})"

    _single_map_figure(z_map, lat, lon,
                       f"Grid-point modified-Kendall z  (k={k})  |  ACE2 vs ERA5 JJA HHE freq",
                       -vmax, vmax, _TAU_CMAP, label,
                       OUT_DIR / "tau_jja_seasonal_global_modkendall.png")

    zc = z_map[CONUS_LAT_SLICE, CONUS_LON_SLICE]
    latc, lonc = lat[CONUS_LAT_SLICE], lon[CONUS_LON_SLICE]
    fc = zc[np.isfinite(zc)]
    vmaxc = float(np.nanpercentile(np.abs(fc), 98)) if fc.size else vmax
    _single_map_figure(zc, latc, lonc,
                       f"CONUS grid-point modified-Kendall z  (k={k})",
                       -vmaxc, vmaxc, _TAU_CMAP, label,
                       OUT_DIR / "tau_jja_seasonal_conus_modkendall.png")

    print("done.", flush=True)


if __name__ == "__main__":
    main()
