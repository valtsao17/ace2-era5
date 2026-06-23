#!/usr/bin/env python3
"""ACE2 Humid Heat Extreme (HHE) frequency from re-inference outputs.

Mirrors heat_index_era5.py (paper definition: a day with NOAA Heat Index
HI >= 105 °F is an HHE) but for the ACE2 May-experiment ensemble, using the
6-hourly TMP2m, Q2m and PRESsfc saved by the re-run inference.

Per member (one autoregressive_predictions.nc):
  - assign datetimes from the raw integer time coord (µs from init)
  - RH per 6-hourly step from specific humidity (Q2m), T (TMP2m), p (PRESsfc)
  - daily  Tmax  = resample('1D').max()  on TMP2m   (paper: Tmax)
  - daily  RHmin = resample('1D').min()  on RH       (paper: RHmin)
  - daily  HI    = NOAA Rothfusz(Tmax °F, RHmin %)
  - keep JJA days

Per year: stack 25 members on member 12's (May 1 00:00) day axis; HHE freq =
fraction of JJA days with HI >= 105 °F, averaged over members.

Outputs -> outputs/lag_may/heat_index_era5/jja_hi_freq_ace2.nc
             ace2_hi_freq (year, lat, lon)  — parallel to era5_hi_freq
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from heat_index_era5 import heat_index, HI_THRESH, MAGNUS_A, MAGNUS_B

RUNS_ROOT = PROJECT_ROOT / "outputs/lag_may/runs"
OUT_DIR   = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
OUT_NC    = OUT_DIR / "jja_hi_freq_ace2.nc"

YEARS     = list(range(1980, 2017))
N_MEMBERS = 25
EPS       = 0.622          # Rd/Rv
NEEDED    = ("TMP2m", "Q2m", "PRESsfc")


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 5, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def assign_times(ds: xr.Dataset, init_time: datetime) -> xr.Dataset:
    vals = ds["time"].values.astype(np.int64)
    return ds.assign_coords(time=pd.Timestamp(init_time) + pd.to_timedelta(vals, unit="us"))


def rh_from_q(t_K: np.ndarray, q: np.ndarray, p_Pa: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from specific humidity, temperature, pressure.

    Jia et al. (2024) Eq. 3 — the exact SPEAR/model formula (paper-strict):

        RH% = 0.263 * p[Pa] * q[kg/kg] * exp(-17.67 (T-273.16) / (T-29.65))

    This is e ~= p*q/0.622 over es(T) = 611.2 exp(17.67(T-273.16)/(T-29.65)),
    folded into the single constant 100/(0.622*611.2) ~= 0.263 (Bolton/SPEAR
    saturation-vapor-pressure form, distinct from the Magnus a,b used for ERA5).
    T in K, p in Pa, q in kg/kg; clipped to [0, 100].
    """
    exponent = 17.67 * (t_K - 273.16) / (t_K - 29.65)
    return np.clip(0.263 * p_Pa * q / np.exp(exponent), 0.0, 100.0)


def member_jja_hi(pred_path: Path, init_time: datetime) -> xr.DataArray | None:
    """Daily JJA Heat Index (°F) for one member, or None if vars are missing."""
    ds = xr.open_dataset(pred_path, decode_times=False)
    if not set(NEEDED).issubset(ds.data_vars):
        ds.close()
        return None
    ds = assign_times(ds, init_time)
    t = ds["TMP2m"].isel(sample=0)
    q = ds["Q2m"].isel(sample=0)
    p = ds["PRESsfc"].isel(sample=0)

    rh = xr.apply_ufunc(rh_from_q, t, q, p, dask="parallelized", output_dtypes=[np.float32])
    tmax_K = t.resample(time="1D").max()
    rhmin  = rh.resample(time="1D").min()
    tmax_F = (tmax_K - 273.15) * 9.0 / 5.0 + 32.0
    hi = xr.apply_ufunc(heat_index, tmax_F, rhmin, dask="parallelized",
                        output_dtypes=[np.float32])
    ds.close()
    jja = hi.time.dt.month.isin([6, 7, 8])
    return hi.sel(time=jja).astype(np.float32)


def year_freq(year: int):
    """HHE freq (lat, lon) for one year + count of members used, or (None, 0)."""
    times = lag_times(year)
    members, ref_idx = {}, None
    for idx in range(N_MEMBERS):
        pred = RUNS_ROOT / str(year) / f"member_{idx:02d}" / "autoregressive_predictions.nc"
        if not pred.exists() or pred.stat().st_size == 0:
            continue
        hi = member_jja_hi(pred, times[idx])
        if hi is None:
            continue
        members[idx] = hi
        if idx == 12:
            ref_idx = 12
    if not members:
        return None, 0, None, None

    ref = members[ref_idx] if ref_idx is not None else members[min(members)]
    ref_times = ref.time.values
    exc = np.stack(
        [m.reindex(time=ref_times, fill_value=np.nan).values >= HI_THRESH
         for m in members.values()],
        axis=0,
    ).astype(np.float32)                       # (n_used, n_jja_days, lat, lon)
    # fraction of JJA days over threshold, then average across members
    freq = np.nanmean(exc.mean(axis=1), axis=0)
    return freq, len(members), ref["lat"].values, ref["lon"].values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="all", help="'all' or comma-separated years")
    p.add_argument("--min-members", type=int, default=1,
                   help="Skip a year unless at least this many members are complete")
    args = p.parse_args()
    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]

    freqs, kept_years, lat, lon, counts = [], [], None, None, []
    for year in years:
        f, n, la, lo = year_freq(year)
        if f is None or n < args.min_members:
            print(f"  {year}: {n} members — skipped", flush=True)
            continue
        if lat is None:
            lat, lon = la, lo
        freqs.append(f); kept_years.append(year); counts.append(n)
        print(f"  {year}: {n} members  mean freq={np.nanmean(f):.5f}  max={np.nanmax(f):.5f}",
              flush=True)

    if not freqs:
        print("No complete years found.", flush=True)
        return

    arr = np.stack(freqs, axis=0)
    out = xr.Dataset(
        {"ace2_hi_freq": (("year", "lat", "lon"), arr),
         "n_members": (("year",), np.array(counts, dtype=np.int16))},
        coords={"year": kept_years, "lat": lat, "lon": lon},
        attrs={"long_name": f"ACE2 JJA frequency of HI >= {HI_THRESH}F days (HHE)",
               "definition": "HI from daily Tmax and daily RHmin; RH from Q2m,TMP2m,PRESsfc"},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(kept_years) == len(YEARS):
        out.to_netcdf(OUT_NC)
        print(f"wrote {OUT_NC}", flush=True)
    else:
        partial = OUT_DIR / "jja_hi_freq_ace2_partial.nc"
        out.to_netcdf(partial)
        print(f"wrote {partial}  ({len(kept_years)}/{len(YEARS)} years)", flush=True)


if __name__ == "__main__":
    main()
