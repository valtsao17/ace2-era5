from __future__ import annotations

from pathlib import Path
import pandas as pd
import xarray as xr

from .io import pick_var, TEMP_CANDIDATES, assign_valid_time, subset_bbox, to_celsius, write_atomic, complete_netcdf


def load_prediction_daily_tmax(
    prediction_path: Path,
    init_time: pd.Timestamp,
    expected_members: int,
    expected_steps: int,
    bbox: tuple[float, float, float, float],
) -> xr.DataArray:
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    ds = xr.open_dataset(prediction_path, decode_times=False, chunks={"time": 40})
    vname = pick_var(ds, TEMP_CANDIDATES, "ACE2S TMP2m prediction")
    if vname != "TMP2m":
        ds = ds.rename({vname: "TMP2m"})

    if expected_members and ds.sizes.get("sample") != expected_members:
        raise ValueError(f"{prediction_path}: expected sample={expected_members}, found {ds.sizes.get('sample')}")
    if expected_steps and ds.sizes.get("time") < expected_steps:
        raise ValueError(f"{prediction_path}: expected at least time={expected_steps}, found {ds.sizes.get('time')}")

    da = assign_valid_time(ds["TMP2m"], init_time)
    da = subset_bbox(to_celsius(da), bbox)
    return da.resample(time="1D").max().rename("predicted_daily_tmax_C")


def compute_target_date_extremes(
    pred_daily: xr.DataArray,
    obs_daily: xr.DataArray,
    baseline: xr.DataArray,
    threshold: xr.DataArray,
    target_date: pd.Timestamp,
    out_path: Path,
    force: bool = False,
) -> xr.Dataset:
    required = ["predicted_hot_extreme", "observed_hot_extreme"]
    if complete_netcdf(out_path, required) and not force:
        return xr.open_dataset(out_path)

    pred_target = pred_daily.sel(time=target_date)
    # obs_daily is already the single target-date slice (scalar time coord)
    obs_target = obs_daily.sel(time=target_date) if "time" in obs_daily.dims else obs_daily

    pred_anom = pred_target - baseline
    obs_anom = obs_target - baseline

    pred_extreme = (pred_anom > threshold).rename("predicted_hot_extreme").astype("float32")
    obs_extreme = (obs_anom > threshold).rename("observed_hot_extreme").astype("float32")

    # These are target-date binary event probabilities/frequencies across ensemble members.
    pred_prob = pred_extreme.mean("sample").rename("predicted_hot_extreme_probability").astype("float32")
    ens_unc = pred_extreme.std("sample").rename("ensemble_uncertainty").astype("float32")
    error = (obs_extreme - pred_prob).rename("prediction_error_era5_minus_ace2s").astype("float32")

    pred_tmax_ensmean = pred_target.mean("sample").rename("predicted_daily_tmax_C_ensmean").astype("float32")
    pred_anom_ensmean = pred_anom.mean("sample").rename("predicted_tmax_anomaly_C_ensmean").astype("float32")
    pred_tmax_member0 = pred_target.isel(sample=0).drop_vars("sample", errors="ignore").rename("predicted_daily_tmax_C_member0").astype("float32")
    pred_anom_member0 = pred_anom.isel(sample=0).drop_vars("sample", errors="ignore").rename("predicted_tmax_anomaly_C_member0").astype("float32")
    obs_tmax = obs_target.rename("observed_daily_tmax_C").astype("float32")
    obs_anom_da = obs_anom.rename("observed_tmax_anomaly_C").astype("float32")

    ds = xr.Dataset(
        {
            "predicted_hot_extreme": pred_extreme,
            "observed_hot_extreme": obs_extreme,
            "predicted_hot_extreme_probability": pred_prob,
            "prediction_error_era5_minus_ace2s": error,
            "ensemble_uncertainty": ens_unc,
            "monthly_tmax_baseline_C": baseline.astype("float32"),
            "hot_anomaly_threshold_C": threshold.astype("float32"),
            "predicted_daily_tmax_C_ensmean": pred_tmax_ensmean,
            "predicted_tmax_anomaly_C_ensmean": pred_anom_ensmean,
            "predicted_daily_tmax_C_member0": pred_tmax_member0,
            "predicted_tmax_anomaly_C_member0": pred_anom_member0,
            "observed_daily_tmax_C": obs_tmax,
            "observed_tmax_anomaly_C": obs_anom_da,
        }
    )
    ds.attrs["target_date"] = str(pd.Timestamp(target_date).date())
    ds.attrs["extreme_definition"] = "target-date daily Tmax anomaly above monthly ERA5 90th percentile anomaly threshold"
    write_atomic(ds, out_path)
    return ds
