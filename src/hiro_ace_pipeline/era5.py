from __future__ import annotations

from pathlib import Path
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from .io import pick_var, TEMP_CANDIDATES, subset_bbox, to_celsius, regrid_nearest_to_template, write_atomic, complete_netcdf


def open_era5(era5_zarr: str):
    ds = xr.open_dataset(era5_zarr, engine="zarr", chunks={})
    return ds, pick_var(ds, TEMP_CANDIDATES, "ERA5 2m temperature")


def cache_era5_month(
    era5_zarr: str,
    year: int,
    month: int,
    bbox: tuple[float, float, float, float],
    template: xr.DataArray,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    path = cache_dir / f"era5_daily_tmax_C_y{year:04d}_m{month:02d}_on_grid.nc"
    if complete_netcdf(path, ["era5_daily_tmax_C"]) and not force:
        return path

    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)

    ds, vname = open_era5(era5_zarr)
    da = subset_bbox(ds[vname].sel(time=slice(str(start), str(end))), bbox)
    da = regrid_nearest_to_template(da, template)
    daily = to_celsius(da).resample(time="1D").max()
    daily = daily.sel(time=daily.time.dt.month == month).rename("era5_daily_tmax_C").astype("float32").load()
    write_atomic(daily, path)
    ds.close()
    return path


def observed_target_date(
    era5_zarr: str,
    target_date: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    template: xr.DataArray,
    cache_dir: Path,
    force: bool = False,
) -> xr.DataArray:
    path = cache_dir / f"era5_observed_daily_tmax_C_{target_date:%Y%m%d}.nc"
    if complete_netcdf(path, ["era5_daily_tmax_C"]) and not force:
        return xr.open_dataset(path)["era5_daily_tmax_C"]

    start = pd.Timestamp(target_date).normalize()
    end = start + pd.Timedelta(hours=23)

    ds, vname = open_era5(era5_zarr)
    da = subset_bbox(ds[vname].sel(time=slice(str(start), str(end))), bbox)
    da = regrid_nearest_to_template(da, template)
    daily = to_celsius(da).resample(time="1D").max()
    daily = daily.sel(time=start).rename("era5_daily_tmax_C").astype("float32").load()
    write_atomic(daily, path)
    ds.close()
    return daily


def sliding_window_baseline_threshold(
    era5_zarr: str,
    target_date: pd.Timestamp,
    window_half_days: int,
    baseline_start_year: int,
    baseline_end_year: int,
    bbox: tuple[float, float, float, float],
    template: xr.DataArray,
    cache_dir: Path,
    processed_dir: Path,
    force: bool = False,
):
    target_md = f"{target_date.month:02d}{target_date.day:02d}"
    win_days = 2 * window_half_days + 1
    out = processed_dir / (
        f"ERA5_sliding{win_days}d_md{target_md}"
        f"_baseline_threshold_{baseline_start_year}_{baseline_end_year}.nc"
    )
    if complete_netcdf(out, ["monthly_tmax_baseline_C", "hot_anomaly_threshold_C"]) and not force:
        ds = xr.open_dataset(out)
        return ds["monthly_tmax_baseline_C"], ds["hot_anomaly_threshold_C"]

    arrays = []
    for year in tqdm(range(baseline_start_year, baseline_end_year + 1), desc=f"ERA5 sliding-window baseline {target_md}"):
        try:
            center = pd.Timestamp(year=year, month=target_date.month, day=target_date.day)
        except ValueError:
            center = pd.Timestamp(year=year, month=target_date.month, day=28)

        win_start = center - pd.Timedelta(days=window_half_days)
        win_end = center + pd.Timedelta(days=window_half_days)

        months_needed = set()
        for delta in range(-window_half_days, window_half_days + 1):
            d = center + pd.Timedelta(days=delta)
            months_needed.add((d.year, d.month))

        month_arrays = []
        for y, m in sorted(months_needed):
            p = cache_era5_month(era5_zarr, y, m, bbox, template, cache_dir, force=False)
            with xr.open_dataset(p) as ds_m:
                month_arrays.append(ds_m["era5_daily_tmax_C"].load())

        all_days = xr.concat(month_arrays, dim="time").sortby("time")
        window_data = all_days.sel(time=slice(str(win_start.date()), str(win_end.date())))
        arrays.append(window_data)

    daily = xr.concat(arrays, dim="time").sortby("time")
    baseline = daily.mean("time").rename("monthly_tmax_baseline_C").astype("float32")
    threshold = (daily - baseline).quantile(0.90, dim="time").rename("hot_anomaly_threshold_C").astype("float32")
    if "quantile" in threshold.coords:
        threshold = threshold.drop_vars("quantile")

    write_atomic(xr.Dataset({"monthly_tmax_baseline_C": baseline, "hot_anomaly_threshold_C": threshold}), out)
    return baseline, threshold


def monthly_baseline_threshold(
    era5_zarr: str,
    month: int,
    baseline_start_year: int,
    baseline_end_year: int,
    bbox: tuple[float, float, float, float],
    template: xr.DataArray,
    cache_dir: Path,
    processed_dir: Path,
    force: bool = False,
):
    out = processed_dir / f"ERA5_month{month:02d}_baseline_threshold_{baseline_start_year}_{baseline_end_year}.nc"
    if complete_netcdf(out, ["monthly_tmax_baseline_C", "hot_anomaly_threshold_C"]) and not force:
        ds = xr.open_dataset(out)
        return ds["monthly_tmax_baseline_C"], ds["hot_anomaly_threshold_C"]

    paths = []
    for year in tqdm(range(baseline_start_year, baseline_end_year + 1), desc=f"ERA5 cache month {month:02d}"):
        paths.append(cache_era5_month(era5_zarr, year, month, bbox, template, cache_dir, force=False))

    arrays = []
    for p in tqdm(paths, desc=f"load ERA5 month {month:02d}", leave=False):
        with xr.open_dataset(p) as ds:
            arrays.append(ds["era5_daily_tmax_C"].load())

    daily = xr.concat(arrays, dim="time").sortby("time")
    baseline = daily.mean("time").rename("monthly_tmax_baseline_C").astype("float32")
    threshold = (daily - baseline).quantile(0.90, dim="time").rename("hot_anomaly_threshold_C").astype("float32")
    if "quantile" in threshold.coords:
        threshold = threshold.drop_vars("quantile")

    write_atomic(xr.Dataset({"monthly_tmax_baseline_C": baseline, "hot_anomaly_threshold_C": threshold}), out)
    return baseline, threshold
