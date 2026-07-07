#!/usr/bin/env python3
"""JJA relative humid-heat extreme skill map.

Definition:
  - daily heat index (HI), not Tmax
  - extreme = HI > grid-cell/day 90th percentile
  - threshold pool = +/-7 calendar days (=15-day window), full climatology
  - thresholds computed separately for ERA5 and ACE2, matching the dry-heat
    relative-threshold workflow.

ACE2 HI can be computed from raw daily Tmax/RHmin or from the Jia-style
bias-corrected daily inputs. The default is bias-corrected, so this remains a
humid-heat analogue of the paper's ACE2 HHE treatment while using a relative
threshold instead of the absolute 105F cutoff.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from heat_index_era5 import heat_index  # noqa: E402
from seasonal_jja_skill import (  # noqa: E402
    HOT_PCT, THRESH_WINDOW, N_MEMBERS, JJA_SEQ, _JJA_POS,
    compute_daywise_thresholds, exceedance_frequency,
    pearson_r_map, kendall_tau_map, cos_lat_mean,
    load_land_mask, plot_global,
)

HHE_DIR = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
ERA5_MONTHLY = HHE_DIR / "monthly_cache"
ACE2_CACHE = HHE_DIR / "ace2_daily_cache"
BIAS_NC = HHE_DIR / "bias_fields.nc"
DEFAULT_OUT = PROJECT_ROOT / f"outputs/lag_may/relative_hhe_jja_sliding{THRESH_WINDOW}d"


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        start, end = [int(x) for x in spec.split(":", 1)]
        return list(range(start, end + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def load_era5_hi_month(year: int, month: int) -> np.ndarray:
    path = ERA5_MONTHLY / f"hi_daily_y{year:04d}_m{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        return ds["hi_F"].values.astype(np.float32)


def load_all_era5_hi(years: list[int], nlat: int, nlon: int) -> np.ndarray:
    arr = np.full((len(years), 92, nlat, nlon), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(years, desc="ERA5 HI load")):
        for month in (6, 7, 8):
            data = load_era5_hi_month(year, month)
            for d in range(data.shape[0]):
                pos = _JJA_POS.get((month, d + 1))
                if pos is not None:
                    arr[y_idx, pos] = data[d]
    return arr


def load_bias_fields(model_hi_inputs: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_hi_inputs == "raw":
        return None, None
    if not BIAS_NC.exists():
        raise FileNotFoundError(f"Missing {BIAS_NC}; run hhe_ace2_biascorr.py --force-bias first.")
    with xr.open_dataset(BIAS_NC) as ds:
        return ds["bias_Tmax"].values.astype(np.float32), ds["bias_RHmin"].values.astype(np.float32)


def load_all_ace2_hi(
    years: list[int],
    nlat: int,
    nlon: int,
    model_hi_inputs: str,
) -> np.ndarray:
    bias_t, bias_r = load_bias_fields(model_hi_inputs)
    arr = np.full((len(years), N_MEMBERS, 92, nlat, nlon), np.nan, dtype=np.float32)
    for y_idx, year in enumerate(tqdm(years, desc=f"ACE2 HI load ({model_hi_inputs})")):
        used = 0
        for member in range(N_MEMBERS):
            path = ACE2_CACHE / f"daily_y{year}_mem{member:02d}.nc"
            if not path.exists():
                continue
            with xr.open_dataset(path) as ds:
                tmax_c = ds["tmax_C"].values.astype(np.float32)
                rhmin = ds["rhmin_pct"].values.astype(np.float32)
                times = pd.DatetimeIndex(ds["time"].values)
            if model_hi_inputs == "bias-corrected":
                tmax_c = tmax_c - bias_t
                rhmin = np.clip(rhmin - bias_r, 0.0, 100.0)
            hi = heat_index(tmax_c * 9.0 / 5.0 + 32.0, rhmin).astype(np.float32)
            for t_idx, ts in enumerate(times):
                pos = _JJA_POS.get((int(ts.month), int(ts.day)))
                if pos is not None:
                    arr[y_idx, member, pos] = hi[t_idx]
            used += 1
        print(f"  ACE2 HI {year}: {used}/{N_MEMBERS} members", flush=True)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="1980:2022")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-hi-inputs", choices=("bias-corrected", "raw"), default="bias-corrected")
    args = p.parse_args()

    years = parse_years(args.years)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    template = sorted(glob.glob(str(ERA5_MONTHLY / f"hi_daily_y{years[0]:04d}_m06.nc")))
    if not template:
        raise FileNotFoundError(f"Missing ERA5 HI monthly cache for {years[0]}; run heat_index_era5.py first.")
    with xr.open_dataset(template[0]) as ds:
        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
    nlat, nlon = len(lat), len(lon)
    print(f"Relative HHE years: {years[0]}-{years[-1]} ({len(years)} years)", flush=True)
    print(f"Grid: {nlat} lat x {nlon} lon", flush=True)

    era5_hi = load_all_era5_hi(years, nlat, nlon)
    ace2_hi = load_all_ace2_hi(years, nlat, nlon, args.model_hi_inputs)

    print(f"Computing ERA5 HI p{HOT_PCT:.0f} +/-{THRESH_WINDOW}d thresholds ...", flush=True)
    era5_thresh = compute_daywise_thresholds(era5_hi, window=THRESH_WINDOW, pct=HOT_PCT)
    print(f"Computing ACE2 HI p{HOT_PCT:.0f} +/-{THRESH_WINDOW}d thresholds ...", flush=True)
    ace2_thresh = compute_daywise_thresholds(ace2_hi, window=THRESH_WINDOW, pct=HOT_PCT)

    era5_freqs = []
    ace2_freqs = []
    for i, year in enumerate(years):
        ef = exceedance_frequency(era5_hi[i], era5_thresh[i], axes=0)
        af = exceedance_frequency(ace2_hi[i], ace2_thresh[i][np.newaxis], axes=(0, 1))
        era5_freqs.append(ef)
        ace2_freqs.append(af)
        print(f"  {year}: ERA5={np.nanmean(ef):.3f} ACE2={np.nanmean(af):.3f}", flush=True)
    era5_freq = np.stack(era5_freqs, axis=0).astype(np.float32)
    ace2_freq = np.stack(ace2_freqs, axis=0).astype(np.float32)

    print("Computing Pearson r ...", flush=True)
    r_map, r_pval = pearson_r_map(ace2_freq, era5_freq)
    print("Computing Kendall tau ...", flush=True)
    tau_map, tau_pval = kendall_tau_map(ace2_freq, era5_freq)

    freq_attrs = {
        "years": f"{years[0]}-{years[-1]}",
        "n_years": len(years),
        "definition": "relative humid-heat extreme: daily HI > grid-cell/day p90",
        "threshold_percentile": HOT_PCT,
        "threshold_window_days_each_side": THRESH_WINDOW,
        "threshold_window_total_days": 2 * THRESH_WINDOW + 1,
        "threshold_method": "full-climatology no-LOYO, computed separately for ERA5 and ACE2",
        "ace2_hi_inputs": args.model_hi_inputs,
    }
    xr.Dataset(
        {
            "ace2_freq": (("year", "lat", "lon"), ace2_freq),
            "era5_freq": (("year", "lat", "lon"), era5_freq),
        },
        coords={"year": years, "lat": lat, "lon": lon},
        attrs=freq_attrs,
    ).to_netcdf(out_dir / "jja_seasonal_freqs.nc")
    print(f"wrote {out_dir / 'jja_seasonal_freqs.nc'}", flush=True)

    xr.Dataset(
        {
            "pearson_r": (("lat", "lon"), r_map),
            "r_p_value": (("lat", "lon"), r_pval),
            "kendall_tau": (("lat", "lon"), tau_map),
            "tau_p_value": (("lat", "lon"), tau_pval),
        },
        coords={"lat": lat, "lon": lon},
        attrs=freq_attrs,
    ).to_netcdf(out_dir / "skill_jja_seasonal.nc")
    print(f"wrote {out_dir / 'skill_jja_seasonal.nc'}", flush=True)

    land = load_land_mask(lat, lon)
    yr_range = f"{years[0]}-{years[-1]} (n={len(years)} seasons)"
    thresh = f"HI p{HOT_PCT:.0f}, +/-{THRESH_WINDOW}d thresholds"
    plot_global(
        r_map, r_pval, lat, lon,
        f"ACE2-ERA5 | JJA relative HHE frequency skill | Pearson r\n{yr_range} | {thresh}",
        out_dir / "pearsonr_jja_seasonal_global.png",
        metric_label="Pearson r",
        land_mask=land,
    )
    plot_global(
        tau_map, tau_pval, lat, lon,
        f"ACE2-ERA5 | JJA relative HHE frequency skill | Kendall tau\n{yr_range} | {thresh}",
        out_dir / "tau_jja_seasonal_global.png",
        metric_label="Kendall tau",
        land_mask=land,
        signed=True,
    )
    print(f"Global mean |tau| = {cos_lat_mean(np.abs(tau_map), lat):.3f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
