#!/usr/bin/env python3
"""Combine 25 lag member outputs into a single prediction file.

Each member ran from a different lag init time to exactly 2014-04-01T00:00.
We take the last time step from each member's output (which is Apr 1) and
stack them as sample=25.

Output: outputs/lag_experiment/predictions/ace2_TMP2m_lag_target20140401.nc
  dims: sample=25, time=1, lat=180, lon=360
  time: [2014-04-01T00:00:00] as pandas datetime (decode_times=True gives proper datetime64)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

WORK_ROOT = Path("/home/jovyan/hiro_ace_clean_v4/outputs/lag_experiment")
RUNS_DIR = WORK_ROOT / "runs"
OUT_PATH = WORK_ROOT / "predictions/ace2_TMP2m_lag_target20140401.nc"
N_MEMBERS = 25


def member_complete(member_dir: Path) -> bool:
    pred = member_dir / "autoregressive_predictions.nc"
    if not pred.exists() or pred.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(pred, decode_times=False) as ds:
            return "TMP2m" in ds.data_vars and ds.sizes.get("time", 0) > 0
    except Exception:
        return False


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    slices = []
    for idx in range(N_MEMBERS):
        member_dir = RUNS_DIR / f"member_{idx:02d}"
        if not member_complete(member_dir):
            raise FileNotFoundError(f"Member {idx:02d} output missing or incomplete")

        pred_path = member_dir / "autoregressive_predictions.nc"
        ds = xr.open_dataset(pred_path, decode_times=False)

        # Each member ran exactly to Apr 1 — take the last time step.
        tmp2m = ds["TMP2m"].isel(time=-1, sample=0)  # (lat, lon)
        slices.append(tmp2m.values)
        ds.close()
        print(f"  member {idx:02d}: loaded last time step", flush=True)

    data = np.stack(slices, axis=0)  # (25, lat, lon)

    # Reconstruct lat/lon from the first member's output
    ds0 = xr.open_dataset(RUNS_DIR / "member_00/autoregressive_predictions.nc", decode_times=False)
    lat = ds0["lat"].values
    lon = ds0["lon"].values
    ds0.close()

    # Write combined file with time=[2014-04-01] as proper datetime
    target_time = pd.to_datetime(["2014-04-01T00:00:00"])
    combined = xr.DataArray(
        data[np.newaxis, :, :, :],   # (1, 25, lat, lon) -> reorder below
        dims=["time", "sample", "lat", "lon"],
        coords={
            "time": target_time,
            "sample": np.arange(N_MEMBERS),
            "lat": lat,
            "lon": lon,
        },
    ).transpose("sample", "time", "lat", "lon")

    ds_out = combined.to_dataset(name="TMP2m")
    ds_out.attrs["description"] = (
        "ACE2-ERA5 lag ensemble: 25 members centred on 2014-02-01T00:00, "
        "6 h spacing, target date 2014-04-01T00:00"
    )

    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds_out.to_netcdf(tmp)
    tmp.replace(OUT_PATH)

    print(f"\nwrote: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
