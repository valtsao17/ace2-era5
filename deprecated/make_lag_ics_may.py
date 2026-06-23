#!/usr/bin/env python3
"""Extract lag IC files for May 1 of each year 1980-2016 from the ERA5 zarr.

Each file: 25 members, 6 h spacing, centred on May 1 00:00.
  member  0: Apr 28 00:00 (-72 h)
  member 12: May  1 00:00 (centre)
  member 24: May  4 00:00 (+72 h)

ICs are pulled from the ACE2-ERA5 requester-pays zarr on GCS.
Set BILLING_PROJECT to a GCP project with billing enabled.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import zarr
import zarr.storage

BILLING_PROJECT = "ace2-ic-download"
ZARR_URL = (
    "gs://ai2cm-public-requester-pays/"
    "2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr"
)
ERA5_REF = datetime(1940, 1, 1, 12, 0, 0)
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parents[1] / "data/lag_data/initial_conditions")

YEARS = list(range(1980, 2017))
N_MEMBERS = 25

IC_VARS = [
    "PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m",
    *[f"air_temperature_{i}" for i in range(8)],
    *[f"specific_total_water_{i}" for i in range(8)],
    *[f"eastward_wind_{i}" for i in range(8)],
    *[f"northward_wind_{i}" for i in range(8)],
]


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 5, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def zarr_idx(dt: datetime) -> int:
    return max(0, int((dt - ERA5_REF).total_seconds() / 3600 // 6))


def make_ic(store, year: int, lat, lon, out_path: str):
    times = lag_times(year)
    t_indices = [zarr_idx(t) for t in times]
    time_vals = [int((t - ERA5_REF).total_seconds() / 3600) for t in times]

    ds = nc.Dataset(out_path, "w")
    ds.createDimension("time", N_MEMBERS)
    ds.createDimension("latitude", len(lat))
    ds.createDimension("longitude", len(lon))

    tv = ds.createVariable("time", "i8", ("time",))
    tv.units = "hours since 1940-01-01T12:00:00"
    tv.calendar = "proleptic_gregorian"
    tv[:] = time_vals

    lv = ds.createVariable("latitude", "f4", ("latitude",))
    lv.units = "index"
    lv.long_name = "y-index of cell center points"
    lv[:] = lat

    lo = ds.createVariable("longitude", "f4", ("longitude",))
    lo.units = "index"
    lo.long_name = "x-index of cell center points"
    lo[:] = lon

    for vname in IC_VARS:
        src = store[vname]
        attrs = dict(src.attrs)
        data = src[t_indices, :, :]
        v = ds.createVariable(vname, "f4", ("time", "latitude", "longitude"))
        v.units = attrs.get("units", "")
        v.long_name = attrs.get("long_name", vname)
        v[:] = data
        print(f"    {vname}", flush=True)

    ds.close()
    print(f"  wrote: {out_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="Directory to write IC files (default: data/lag_data/initial_conditions)")
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated years, e.g. '1980,1981'")
    p.add_argument("--billing-project", default=BILLING_PROJECT,
                   help="GCP billing project for requester-pays bucket")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]

    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())

    store = zarr.open(
        ZARR_URL,
        mode="r",
        storage_options={"project": args.billing_project, "requester_pays": True, "token": creds},
    )
    lat = store["latitude"][:]
    lon = store["longitude"][:]

    for year in years:
        out_path = str(out_dir / f"ic_lag_{year}0501_25m.nc")
        if Path(out_path).exists():
            print(f"  SKIP {year}: already exists")
            continue
        print(f"\n=== {year} May 1 lag IC ===", flush=True)
        make_ic(store, year, lat, lon, out_path)

    print("\nAll IC files done.", flush=True)


if __name__ == "__main__":
    main()
