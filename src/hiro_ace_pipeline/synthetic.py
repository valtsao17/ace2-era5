from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from .planning import TargetDateCase
from .io import write_atomic


def make_synthetic_prediction(case: TargetDateCase, out_path: Path, members: int = 3) -> Path:
    ntime = case.required_6hour_steps
    lat = np.linspace(35.0, 37.0, 8)
    lon = np.linspace(250.0, 252.0, 8)
    time = np.arange(ntime)

    rng = np.random.default_rng(case.lead_days)
    trend = np.linspace(294, 299, ntime)[:, None, None]
    spatial = 1.5 * np.sin(np.linspace(0, np.pi, len(lat)))[None, :, None]
    noise = rng.normal(0, 0.8, size=(members, ntime, len(lat), len(lon)))
    lead_bias = 0.02 * case.lead_days

    arr = trend[None, :, :, :] + spatial[None, :, :, :] + noise + lead_bias
    da = xr.DataArray(
        arr.astype("float32"),
        dims=("sample", "time", "lat", "lon"),
        coords={"sample": np.arange(members), "time": time, "lat": lat, "lon": lon},
        name="TMP2m",
    )
    write_atomic(da, out_path)
    return out_path


def make_synthetic_era5_target_date(target_date: pd.Timestamp, template: xr.DataArray) -> xr.DataArray:
    lat = template["lat"]
    lon = template["lon"]
    yy = np.linspace(0, 1, len(lat))
    xx = np.linspace(0, 1, len(lon))
    arr = 25 + 3 * yy[:, None] + 1.5 * xx[None, :]
    return xr.DataArray(
        arr.astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="era5_target_tmax_C",
    ).expand_dims(time=[target_date])


def make_synthetic_baseline_threshold(template: xr.DataArray):
    field = template.isel(time=0, sample=0, drop=True) - 273.15
    baseline = (field * 0 + 22).rename("monthly_tmax_baseline_C")
    threshold = (field * 0 + 1.0).rename("hot_anomaly_threshold_C")
    return baseline, threshold
