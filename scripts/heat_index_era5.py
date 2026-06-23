#!/usr/bin/env python3
"""ERA5-only NOAA Heat Index (HI) climatology, JJA 1980-2016.

For each grid cell and each JJA calendar day (paper definition: daily Tmax +
daily RHmin, both from 6-hourly samples so ERA5 matches ACE2's cadence 1:1):
  1. Regrid hourly 2m T and Td to the 1-degree ACE2 template (nearest), then
     subsample to 6-hourly (00,06,12,18 UTC) — the 4 synoptic times ACE2 has.
  2. Daily Tmax  = max of the 4 6-hourly temperatures.
  3. RH (Magnus-Tetens, a=17.72, b=243.12, T/Td in degC) at each 6-hourly step:
       RH = 100 * exp(a*Td/(b+Td)) / exp(a*T/(b+T));  daily RHmin = min of the 4.
  4. NOAA HI (Rothfusz regression, T in degF, RH in 0-100), from Tmax & RHmin:
       HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH
            - 0.00683783*T^2 - 0.05481717*RH^2 + 0.00122874*T^2*RH
            + 0.00085282*T*RH^2 - 0.00000199*T^2*RH^2
  5. Day counts as an extreme if HI >= HI_THRESH (105 degF).

Seasonal frequency = fraction of JJA days with HI >= HI_THRESH, per year
per grid cell -> era5_hi_freq (n_years, lat, lon), same shape/grid as the
existing era5_freq in jja_seasonal_freqs.nc.

Outputs -> outputs/lag_may/heat_index_era5/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hiro_ace_pipeline.io import subset_bbox, regrid_nearest_to_template, to_celsius, write_atomic

ERA5_ZARR   = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
YEARS       = list(range(1980, 2017))
MONTHS      = [6, 7, 8]
BBOX        = (-90.0, 90.0, 0.0, 360.0)
HI_THRESH   = 105.0

MAGNUS_A = 17.72    # WMO/paper Magnus coefficients (Eqn 1), matches reference paper
MAGNUS_B = 243.12   # (was 17.625/243.04 Alduchov-Eskridge; switched to match paper)

OUT_DIR    = PROJECT_ROOT / "outputs/lag_may/heat_index_era5"
CACHE_DIR  = OUT_DIR / "monthly_cache"
TEMPLATE_NC = PROJECT_ROOT / "outputs/lag_may/seasonal_jja_sliding7d/jja_seasonal_freqs.nc"


def get_template() -> xr.DataArray:
    with xr.open_dataset(TEMPLATE_NC) as ds:
        return ds["era5_freq"].isel(year=0).drop_vars("year")


def relative_humidity(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    return 100.0 * np.exp(MAGNUS_A * td_c / (MAGNUS_B + td_c)) / np.exp(MAGNUS_A * t_c / (MAGNUS_B + t_c))


def heat_index(t_f: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """NOAA/NWS heat index (°F) with the standard conditional algorithm.

    The Rothfusz regression is only valid for hot conditions; applied to cold
    temperatures it diverges to nonsensical values (e.g. hundreds of °F at the
    poles). NWS therefore uses the simple Steadman formula and only switches to
    the Rothfusz regression (with low/high-RH adjustments) when the result is
    >= 80 °F. See https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    """
    T = np.asarray(t_f, dtype=np.float64)
    R = np.asarray(rh, dtype=np.float64)

    # Simple Steadman formula — used for cooler conditions.
    hi_simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)

    # Full Rothfusz regression.
    hi_roth = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
               - 0.00683783 * T**2 - 0.05481717 * R**2 + 0.00122874 * T**2 * R
               + 0.00085282 * T * R**2 - 0.00000199 * T**2 * R**2)

    # Low-RH adjustment (subtract) and high-RH adjustment (add).
    low = (R < 13.0) & (T > 80.0) & (T < 112.0)
    hi_roth = np.where(
        low,
        hi_roth - ((13.0 - R) / 4.0) * np.sqrt(np.maximum(17.0 - np.abs(T - 95.0), 0.0) / 17.0),
        hi_roth,
    )
    high = (R > 85.0) & (T > 80.0) & (T < 87.0)
    hi_roth = np.where(high, hi_roth + ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0), hi_roth)

    # Use Steadman unless the (Steadman, T) average reaches 80 °F.
    return np.where((hi_simple + T) / 2.0 >= 80.0, hi_roth, hi_simple).astype(np.float32)


def compute_month_hi(year: int, month: int, template: xr.DataArray, force: bool = False) -> Path:
    """Daily max-T, matched-Td, RH, HI for one year-month -> cached .nc."""
    path = CACHE_DIR / f"hi_daily_y{year:04d}_m{month:02d}.nc"
    if path.exists() and not force:
        return path

    start = pd.Timestamp(year=year, month=month, day=1)
    end   = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)

    print(f"  [{year}-{month:02d}] opening zarr ...", flush=True)
    ds = xr.open_dataset(ERA5_ZARR, engine="zarr", chunks={}, storage_options={"token": "anon"})
    t_da  = subset_bbox(ds["2m_temperature"].sel(time=slice(str(start), str(end))), BBOX)
    td_da = subset_bbox(ds["2m_dewpoint_temperature"].sel(time=slice(str(start), str(end))), BBOX)

    print(f"  [{year}-{month:02d}] regridding + loading T ...", flush=True)
    t_da  = to_celsius(regrid_nearest_to_template(t_da, template)).load()
    print(f"  [{year}-{month:02d}] regridding + loading Td ...", flush=True)
    td_da = to_celsius(regrid_nearest_to_template(td_da, template)).load()
    ds.close()
    print(f"  [{year}-{month:02d}] fetch done, computing HI ...", flush=True)

    n_hours, nlat, nlon = t_da.shape
    n_days = n_hours // 24
    assert n_hours == n_days * 24, f"unexpected hour count {n_hours} for {year}-{month:02d}"

    t_arr  = t_da.values.reshape(n_days, 24, nlat, nlon)
    td_arr = td_da.values.reshape(n_days, 24, nlat, nlon)

    # Subsample to 6-hourly (00,06,12,18 UTC) so ERA5 matches ACE2's 6-hourly
    # cadence exactly: BOTH now use daily Tmax + daily RHmin from 4 synoptic
    # times (paper definition), instead of ERA5's old RH-at-Tmax-hour from 24h.
    t6  = t_arr[:, ::6, :, :]                              # (n_days, 4, nlat, nlon)
    td6 = td_arr[:, ::6, :, :]
    t_max  = t6.max(axis=1)                                # daily Tmax  (paper)
    rh6    = relative_humidity(t6, td6)                    # RH at each 6-hourly step
    rh_min = np.clip(rh6.min(axis=1), 0.0, 100.0)          # daily RHmin (paper, matches ACE2)
    hi = heat_index(t_max * 9.0 / 5.0 + 32.0, rh_min)

    day_times = t_da.time.values.reshape(n_days, 24)[:, 0]
    out = xr.Dataset(
        {
            "tmax_C":    (("time", "lat", "lon"), t_max.astype(np.float32)),
            "rhmin_pct": (("time", "lat", "lon"), rh_min.astype(np.float32)),
            "hi_F":      (("time", "lat", "lon"), hi.astype(np.float32)),
        },
        coords={"time": day_times, "lat": template.lat.values, "lon": template.lon.values},
    )
    write_atomic(out, path)
    print(f"wrote {path}", flush=True)
    return path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--year", type=int, default=None, help="Run a single year (for testing)")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    template = get_template()
    years = [args.year] if args.year else YEARS

    freq = np.full((len(years), template.lat.size, template.lon.size), np.nan, dtype=np.float32)
    for yi, year in enumerate(tqdm(years, desc="years")):
        month_paths = [compute_month_hi(year, m, template, force=args.force) for m in MONTHS]
        hi_all = xr.concat([xr.open_dataset(p)["hi_F"] for p in month_paths], dim="time").sortby("time")
        freq[yi] = (hi_all.values >= HI_THRESH).mean(axis=0)

    if args.year:
        print(f"year={args.year}  mean HI>=105F freq over grid: {np.nanmean(freq[0]):.5f}  "
              f"max: {np.nanmax(freq[0]):.5f}", flush=True)
        return

    out = xr.Dataset(
        {"era5_hi_freq": (("year", "lat", "lon"), freq)},
        coords={"year": years, "lat": template.lat.values, "lon": template.lon.values},
        attrs={"long_name": f"ERA5 JJA frequency of HI >= {HI_THRESH}F days"},
    )
    write_atomic(out, OUT_DIR / "jja_hi_freq_era5.nc")
    print(f"wrote {OUT_DIR / 'jja_hi_freq_era5.nc'}", flush=True)


if __name__ == "__main__":
    main()
