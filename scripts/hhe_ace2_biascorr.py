#!/usr/bin/env python3
"""ACE2 HHE frequency with paper-style additive bias correction (Jia et al. 2024 §8).

The absolute HI>=105F threshold makes the HHE count exquisitely sensitive to
climatological mean errors in Tmax/RHmin, so the paper bias-corrects the two
inputs *separately and additively* before the nonlinear heat index — it never
corrects HI or the HHE frequency directly. We reproduce that here:

  bias_v[grid] = mu_model_v[grid] - mu_obs_v[grid]            (v = Tmax or RHmin)
  v_corrected  = v_raw - bias_v                               (additive, per cell)

with mu_* the JJA climatological means over ALL days, ALL years, and (model)
ALL members.  We have a single forecast lead (the May-init experiment = lead 1),
so there is one bias field.  Strict-paper mode: full-period climatology
(verification years included), consistent with the no-LOYO percentile choice.

Pipeline (idempotent / resumable):
  A. build a compact daily JJA Tmax(°C)+RHmin(%) cache from the raw 6-hourly
     members  -> ace2_daily_cache/daily_y{YYYY}_mem{ii}.nc   (one raw read)
  B. ERA5 obs climatology  mu_obs   from heat_index_era5 monthly_cache
  C. ACE2 model climatology mu_model from the cache -> bias_fields.nc
  D. corrected & raw JJA HHE frequency -> jja_hi_freq_ace2.nc (+ _raw.nc)

RH model formula = paper Eq.3 (set in hhe_ace2.rh_from_q).  RHmin is clipped to
[0,100] after correction (physically-constrained mode; clip count reported).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from heat_index_era5 import heat_index, HI_THRESH
from hhe_ace2 import (
    lag_times, assign_times, rh_from_q, NEEDED, N_MEMBERS, YEARS, RUNS_ROOT,
)

OUT_DIR    = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
CACHE_DIR  = OUT_DIR / "ace2_daily_cache"
ERA5_MONTHLY = OUT_DIR / "monthly_cache"
BIAS_NC    = OUT_DIR / "bias_fields.nc"
OUT_NC     = OUT_DIR / "jja_hi_freq_ace2.nc"
RAW_NC     = OUT_DIR / "jja_hi_freq_ace2_raw.nc"

CONUS_LAT = slice(105, 163)   # matches cluster_skill CONUS_LAT_SLICE
CONUS_LON = slice(200, 305)   # matches cluster_skill CONUS_LON_SLICE


def _atomic_write(ds: xr.Dataset, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- A: daily cache
def member_daily(pred_path: Path, init_time):
    """Daily JJA Tmax(°C) and RHmin(%) for one member, or None if vars missing."""
    ds = xr.open_dataset(pred_path, decode_times=False)
    if not set(NEEDED).issubset(ds.data_vars):
        ds.close()
        return None
    ds = assign_times(ds, init_time)
    t = ds["TMP2m"].isel(sample=0)
    q = ds["Q2m"].isel(sample=0)
    p = ds["PRESsfc"].isel(sample=0)
    rh = xr.apply_ufunc(rh_from_q, t, q, p, dask="parallelized", output_dtypes=[np.float32])
    tmax_C = (t.resample(time="1D").max() - 273.15)
    rhmin  = rh.resample(time="1D").min()
    ds.close()
    jja = tmax_C.time.dt.month.isin([6, 7, 8])
    return (tmax_C.sel(time=jja).astype(np.float32),
            rhmin.sel(time=jja).astype(np.float32))


def build_cache(years, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    built = skipped = missing = 0
    for year in years:
        times = lag_times(year)
        for idx in range(N_MEMBERS):
            cpath = CACHE_DIR / f"daily_y{year}_mem{idx:02d}.nc"
            if cpath.exists() and not force:
                skipped += 1
                continue
            pred = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
            if not pred.exists() or pred.stat().st_size == 0:
                missing += 1
                continue
            res = member_daily(pred, times[idx])
            if res is None:
                missing += 1
                continue
            tmax_C, rhmin = res
            _atomic_write(xr.Dataset({"tmax_C": tmax_C, "rhmin_pct": rhmin}), cpath)
            built += 1
        print(f"  cache {year}: built so far={built} skipped={skipped} missing={missing}", flush=True)
    print(f"cache done: built={built} skipped(existing)={skipped} missing/incomplete={missing}", flush=True)


# ---------------------------------------------------------- B/C: climatologies
def _accumulate(paths, tvar, rvar):
    """cos-grid running mean of daily tvar/rvar over all days in the given files."""
    sum_t = sum_r = cnt = None
    for p in paths:
        with xr.open_dataset(p) as d:
            t = d[tvar].values.astype(np.float64)   # (ndays, lat, lon)
            r = d[rvar].values.astype(np.float64)
        if sum_t is None:
            sh = t.shape[1:]
            sum_t = np.zeros(sh); sum_r = np.zeros(sh); cnt = np.zeros(sh)
        sum_t += np.nansum(t, axis=0)
        sum_r += np.nansum(r, axis=0)
        cnt   += np.isfinite(t).sum(axis=0)
    cnt = np.where(cnt > 0, cnt, np.nan)
    return sum_t / cnt, sum_r / cnt


def obs_clim():
    paths = sorted(glob.glob(str(ERA5_MONTHLY / "hi_daily_y*_m*.nc")))
    assert paths, "no ERA5 monthly_cache files"
    print(f"  ERA5 obs clim from {len(paths)} monthly files ...", flush=True)
    return _accumulate(paths, "tmax_C", "rhmin_pct")


def model_clim():
    paths = sorted(glob.glob(str(CACHE_DIR / "daily_y*_mem*.nc")))
    assert paths, "no ACE2 daily cache files — run cache build first"
    print(f"  ACE2 model clim from {len(paths)} member files ...", flush=True)
    return _accumulate(paths, "tmax_C", "rhmin_pct")


def compute_bias():
    mu_obs_t, mu_obs_r = obs_clim()
    mu_mod_t, mu_mod_r = model_clim()
    bias_t = mu_mod_t - mu_obs_t
    bias_r = mu_mod_r - mu_obs_r
    with xr.open_dataset(sorted(glob.glob(str(CACHE_DIR / "*.nc")))[0]) as d:
        lat, lon = d.lat.values, d.lon.values

    def _cm(a):  # CONUS land-ish mean for a quick sanity print
        return float(np.nanmean(a[CONUS_LAT, CONUS_LON]))
    print(f"  bias CONUS-mean: Tmax={_cm(bias_t):+.2f} °C   RHmin={_cm(bias_r):+.2f} pts", flush=True)

    ds = xr.Dataset(
        {"bias_Tmax": (("lat", "lon"), bias_t.astype(np.float32)),
         "bias_RHmin": (("lat", "lon"), bias_r.astype(np.float32)),
         "mu_model_Tmax": (("lat", "lon"), mu_mod_t.astype(np.float32)),
         "mu_obs_Tmax": (("lat", "lon"), mu_obs_t.astype(np.float32)),
         "mu_model_RHmin": (("lat", "lon"), mu_mod_r.astype(np.float32)),
         "mu_obs_RHmin": (("lat", "lon"), mu_obs_r.astype(np.float32))},
        coords={"lat": lat, "lon": lon},
        attrs={"method": "additive JJA climatological-mean bias correction "
                         "(Jia et al. 2024 §8); bias = mu_model - mu_obs; "
                         "single lead (May-init); full-period clim 1980-2016",
               "tmax_units": "degC", "rhmin_units": "percent_points"},
    )
    _atomic_write(ds, BIAS_NC)
    print(f"wrote {BIAS_NC}", flush=True)
    return bias_t.astype(np.float32), bias_r.astype(np.float32), lat, lon


# --------------------------------------------------------------- D: frequency
def compute_freq(years, bias_t, bias_r, lat, lon):
    freq_corr, freq_raw, kept, counts = [], [], [], []
    n_clipped = total = 0
    for year in years:
        mf_corr, mf_raw = [], []
        for idx in range(N_MEMBERS):
            cpath = CACHE_DIR / f"daily_y{year}_mem{idx:02d}.nc"
            if not cpath.exists():
                continue
            with xr.open_dataset(cpath) as d:
                tmax_C = d["tmax_C"].values.astype(np.float32)     # (ndays, lat, lon)
                rhmin  = d["rhmin_pct"].values.astype(np.float32)
            # raw (no bias correction)
            hi_raw = heat_index(tmax_C * 9.0 / 5.0 + 32.0, rhmin)
            mf_raw.append((hi_raw >= HI_THRESH).mean(axis=0))
            # corrected: subtract additive bias, then clip RH to [0,100]
            tmax_c = tmax_C - bias_t
            rhmin_c = rhmin - bias_r
            n_clipped += int(((rhmin_c < 0) | (rhmin_c > 100)).sum())
            total += rhmin_c.size
            rhmin_c = np.clip(rhmin_c, 0.0, 100.0)
            hi_c = heat_index(tmax_c * 9.0 / 5.0 + 32.0, rhmin_c)
            mf_corr.append((hi_c >= HI_THRESH).mean(axis=0))
        if not mf_corr:
            print(f"  {year}: 0 members — skipped", flush=True)
            continue
        fc = np.nanmean(np.stack(mf_corr, 0), axis=0)
        fr = np.nanmean(np.stack(mf_raw, 0), axis=0)
        freq_corr.append(fc); freq_raw.append(fr)
        kept.append(year); counts.append(len(mf_corr))
        print(f"  {year}: {len(mf_corr)} members  corr={np.nanmean(fc):.5f}  raw={np.nanmean(fr):.5f}",
              flush=True)
    print(f"  RH clip after correction: {n_clipped}/{total} ({100*n_clipped/max(total,1):.3f}%)",
          flush=True)

    counts_arr = np.array(counts, dtype=np.int16)
    for arr, path, var, lab in ((freq_corr, OUT_NC, "ace2_hi_freq", "bias-corrected"),
                                (freq_raw, RAW_NC, "ace2_hi_freq", "raw (uncorrected)")):
        ds = xr.Dataset(
            {var: (("year", "lat", "lon"), np.stack(arr, 0)),
             "n_members": (("year",), counts_arr)},
            coords={"year": kept, "lat": lat, "lon": lon},
            attrs={"long_name": f"ACE2 JJA freq of HI>={HI_THRESH}F days (HHE), {lab}",
                   "definition": "HI from daily Tmax & daily RHmin; RH=paper Eq.3; "
                                 + ("additive Tmax/RHmin bias correction (Jia 2024 §8)"
                                    if "corrected" in lab else "no bias correction")},
        )
        _atomic_write(ds, path)
        print(f"wrote {path}  ({len(kept)}/{len(YEARS)} years, {lab})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="all")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--force-bias", action="store_true")
    ap.add_argument("--skip-cache", action="store_true",
                    help="assume the daily cache is already built")
    args = ap.parse_args()
    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]

    print("=== A. build daily Tmax/RHmin cache ===", flush=True)
    if not args.skip_cache:
        build_cache(years, force=args.force_cache)

    print("=== B/C. climatologies + bias fields ===", flush=True)
    if BIAS_NC.exists() and not args.force_bias:
        with xr.open_dataset(BIAS_NC) as d:
            bias_t = d["bias_Tmax"].values; bias_r = d["bias_RHmin"].values
            lat, lon = d.lat.values, d.lon.values
        print(f"  using existing {BIAS_NC}", flush=True)
    else:
        bias_t, bias_r, lat, lon = compute_bias()

    print("=== D. corrected & raw HHE frequency ===", flush=True)
    compute_freq(years, bias_t, bias_r, lat, lon)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
