#!/usr/bin/env python3
"""Revised ACE2 SST-causality experiment requested by Yutian/CZ.

Subcommands
-----------
composite
    Select top/bottom ACE2 HHE years and build March-August prescribed-SST
    high-minus-low composites.
climatology
    Build a 6-hourly 1980-2022 surface-temperature seasonal climatology on a
    chosen template year (normally 2000).
prepare
    Create a shared climatological-SST control and a perturbation input set.
    Perturbations may be persistent, evolving May-August, global, or restricted
    to one named ocean basin.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MONTH_NAMES = {3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug"}
BASINS = {
    "global": None,
    "tropical_pacific": (-20.0, 20.0, 120.0, 280.0),
    "north_pacific": (20.0, 65.0, 120.0, 260.0),
    "tropical_atlantic": (-20.0, 20.0, 290.0, 15.0),
    "north_atlantic": (20.0, 65.0, 280.0, 360.0),
}


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":", 1))
        return list(range(a, b + 1))
    if "-" in spec and "," not in spec:
        a, b = (int(x) for x in spec.split("-", 1))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def box_mask(lat: np.ndarray, lon: np.ndarray, box: tuple[float, float, float, float] | list[float]) -> np.ndarray:
    south, north, west, east = box
    lon = np.asarray(lon) % 360.0
    if west % 360.0 <= east % 360.0 and not (east == 360.0):
        x = (lon >= west % 360.0) & (lon <= east % 360.0)
    else:
        x = (lon >= west % 360.0) | (lon <= east % 360.0)
    return (lat[:, None] >= south) & (lat[:, None] <= north) & x[None, :]


def weighted_index(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(field)
    numerator = np.nansum(field * w[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * w[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def detrend_axis0(arr: np.ndarray) -> np.ndarray:
    """Remove a linear trend along the leading axis while retaining its mean."""
    a = np.asarray(arr, dtype=np.float64)
    x = np.arange(a.shape[0], dtype=np.float64)
    xc = x - x.mean()
    shape = (-1,) + (1,) * (a.ndim - 1)
    valid = np.isfinite(a)
    count = valid.sum(axis=0)
    mean = np.divide(np.nansum(a, axis=0), count, out=np.full(a.shape[1:], np.nan), where=count > 0)
    cov = np.nansum(np.where(valid, (a - mean) * xc.reshape(shape), 0.0), axis=0)
    varx = np.sum(np.where(valid, xc.reshape(shape) ** 2, 0.0), axis=0)
    slope = np.divide(cov, varx, out=np.zeros_like(cov), where=varx > 0)
    return np.where(valid, a - xc.reshape(shape) * slope, np.nan)


def time_keys(time: xr.DataArray) -> list[tuple[int, int, int]]:
    return list(zip(
        np.asarray(time.dt.month, dtype=int).tolist(),
        np.asarray(time.dt.day, dtype=int).tolist(),
        np.asarray(time.dt.hour, dtype=int).tolist(),
    ))


def forcing_land_ocean(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"], dtype=float)
        lon = np.asarray(ds["longitude"], dtype=float) % 360.0
        land = ds["land_fraction"]
        if "time" in land.dims:
            land = land.isel(time=0)
        return lat, lon, np.asarray(land) > 0.5


def render_composites(ds: xr.Dataset, path: Path) -> None:
    lon = np.asarray(ds["lon"], dtype=float)
    lat = np.asarray(ds["lat"], dtype=float)
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180)
    fields = [np.asarray(ds[f"sst_high_minus_low_{MONTH_NAMES[m]}"])[:, order] for m in MONTH_NAMES]
    finite = np.concatenate([np.abs(x[np.isfinite(x)]) for x in fields])
    vmax = max(float(np.nanpercentile(finite, 98)), 0.25) if finite.size else 1.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True)
    mesh = None
    for ax, month, field in zip(axes.flat, MONTH_NAMES, fields):
        mesh = ax.pcolormesh(lon180[order], lat, field, shading="nearest", cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
        ax.set_title(f"{month:02d}  {datetime(2001, month, 1):%B}", loc="left", fontweight="bold")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.colorbar(mesh, ax=axes, shrink=0.88, label="ACE2-selected high-minus-low SST (K)")
    fig.suptitle("Prescribed SST composites conditioned on ACE2 HHE top/bottom quintiles",
                 fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def build_composite(args: argparse.Namespace) -> None:
    requested = parse_years(args.years)
    with xr.open_dataset(args.ace2_hhe_nc) as ds:
        if args.ace2_hhe_var not in ds:
            raise KeyError(f"{args.ace2_hhe_var!r} not in {list(ds.data_vars)}")
        da = ds[args.ace2_hhe_var]
        available = {int(y) for y in da["year"].values}
        years = [y for y in requested if y in available]
        missing = [y for y in requested if y not in available]
        if missing and args.strict_years:
            raise ValueError(f"ACE2 HHE target is missing years: {missing}")
        target = np.asarray(da.sel(year=years).transpose("year", "lat", "lon"), dtype=np.float32)
        target_lat = np.asarray(da["lat"], dtype=float)
        target_lon = np.asarray(da["lon"], dtype=float) % 360.0
    if len(years) < 10:
        raise ValueError(f"Need at least 10 target years for quintiles; found {len(years)}")

    first_forcing = Path(args.forcing_dir) / f"forcing_{years[0]}.nc"
    lat, lon, land = forcing_land_ocean(first_forcing)
    if target.shape[1:] != land.shape or not (np.allclose(target_lat, lat) and np.allclose(target_lon, lon)):
        raise ValueError("ACE2 HHE and prescribed-SST forcing grids differ")
    region = box_mask(lat, lon, args.index_box) & land
    index = weighted_index(target, lat, region)
    index_dt = detrend_axis0(index)
    if not np.all(np.isfinite(index_dt)):
        raise ValueError("ACE2 regional HHE index contains missing values")
    n_tail = max(2, int(math.floor(len(years) * args.tail_fraction)))
    order = np.argsort(index_dt)
    low_i, high_i = order[:n_tail], order[-n_tail:]

    monthly_sst: dict[int, list[np.ndarray]] = {m: [] for m in MONTH_NAMES}
    monthly_ice: dict[int, list[np.ndarray]] = {m: [] for m in MONTH_NAMES}
    ocean_static = None
    for year in years:
        path = Path(args.forcing_dir) / f"forcing_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            if ocean_static is None:
                oc = ds["ocean_fraction"]
                ocean_static = np.asarray(oc.isel(time=0) if "time" in oc.dims else oc) > args.ocean_min
            for month in MONTH_NAMES:
                sel = ds.time.dt.month == month
                monthly_sst[month].append(np.asarray(ds["surface_temperature"].sel(time=sel).mean("time"), dtype=np.float32))
                monthly_ice[month].append(np.asarray(ds["sea_ice_fraction"].sel(time=sel).mean("time"), dtype=np.float32))
        print(f"loaded ACE2-prescribed SST {year}", flush=True)

    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "regional_ace2_hhe_index": (("year",), index.astype(np.float32)),
        "regional_ace2_hhe_index_detrended": (("year",), index_dt.astype(np.float32)),
        "group": (("year",), np.array([
            1 if i in set(high_i) else -1 if i in set(low_i) else 0 for i in range(len(years))
        ], dtype=np.int8)),
    }
    for month in MONTH_NAMES:
        cube = detrend_axis0(np.stack(monthly_sst[month]))
        ice_clim = np.nanmean(np.stack(monthly_ice[month]), axis=0)
        valid_ocean = ocean_static & (ice_clim <= args.max_sea_ice)
        high = np.nanmean(cube[high_i], axis=0)
        low = np.nanmean(cube[low_i], axis=0)
        diff = np.where(valid_ocean, high - low, np.nan).astype(np.float32)
        _, pval = stats.ttest_ind(cube[high_i], cube[low_i], axis=0, equal_var=False, nan_policy="omit")
        tag = MONTH_NAMES[month]
        variables[f"sst_high_minus_low_{tag}"] = (("lat", "lon"), diff)
        variables[f"sst_high_composite_{tag}"] = (("lat", "lon"), np.where(valid_ocean, high, np.nan).astype(np.float32))
        variables[f"sst_low_composite_{tag}"] = (("lat", "lon"), np.where(valid_ocean, low, np.nan).astype(np.float32))
        variables[f"welch_p_value_{tag}"] = (("lat", "lon"), np.where(valid_ocean, pval, np.nan).astype(np.float32))
        variables[f"valid_ocean_{tag}"] = (("lat", "lon"), valid_ocean.astype(np.int8))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        variables,
        coords={"year": years, "lat": lat, "lon": lon},
        attrs={
            "description": "Prescribed-SST composites conditioned on ACE2 ensemble-mean HHE years",
            "ace2_hhe_file": str(Path(args.ace2_hhe_nc).resolve()),
            "ace2_hhe_variable": args.ace2_hhe_var,
            "sst_source": "ACE2 prescribed surface_temperature forcing experienced by the hindcasts",
            "selection": "detrended land-only regional ACE2 HHE index; exact ranked tails",
            "tail_fraction": args.tail_fraction,
            "index_box": json.dumps(args.index_box),
            "high_years": json.dumps([years[i] for i in high_i]),
            "low_years": json.dumps([years[i] for i in low_i]),
        },
    )
    for month in MONTH_NAMES.values():
        out[f"sst_high_minus_low_{month}"].attrs.update(units="K", long_name="ACE2-selected high-HHE minus low-HHE prescribed SST")
    out.to_netcdf(out_path)
    figure = Path(args.figure) if args.figure else out_path.with_suffix(".png")
    render_composites(out, figure)
    summary = {
        "available_years": years,
        "missing_requested_years": missing,
        "high_years": [years[i] for i in high_i],
        "low_years": [years[i] for i in low_i],
        "tail_size": n_tail,
        "index_box": args.index_box,
        "output": str(out_path),
        "figure": str(figure),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_climatology(args: argparse.Namespace) -> None:
    years = parse_years(args.years)
    template_path = Path(args.forcing_dir) / f"forcing_{args.template_year}.nc"
    if not template_path.exists():
        raise FileNotFoundError(template_path)
    with xr.open_dataset(template_path) as template_ds:
        target_time = template_ds["time"].load()
        lat = np.asarray(template_ds["latitude"], dtype=np.float32)
        lon = np.asarray(template_ds["longitude"], dtype=np.float32)
    target_keys = time_keys(target_time)
    target_lookup = {key: i for i, key in enumerate(target_keys)}
    total = np.zeros((len(target_keys), len(lat), len(lon)), dtype=np.float32)
    count = np.zeros(len(target_keys), dtype=np.int16)

    for year in years:
        path = Path(args.forcing_dir) / f"forcing_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            keys = time_keys(ds["time"])
            indices = np.array([target_lookup[k] for k in keys if k in target_lookup], dtype=int)
            source_indices = np.array([i for i, k in enumerate(keys) if k in target_lookup], dtype=int)
            values = np.asarray(ds["surface_temperature"].isel(time=source_indices), dtype=np.float32)
        total[indices] += values
        count[indices] += 1
        print(f"climatology accumulated {year}", flush=True)
    if np.any(count == 0):
        missing = [target_keys[i] for i in np.where(count == 0)[0][:10]]
        raise ValueError(f"Climatology has unfilled calendar slots: {missing}")
    climatology = total / count[:, None, None]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = xr.Dataset(
        {
            "surface_temperature_climatology": (("time", "latitude", "longitude"), climatology),
            "sample_count": (("time",), count),
        },
        coords={"time": target_time.values, "latitude": lat, "longitude": lon},
        attrs={
            "description": "6-hourly calendar-slot surface-temperature climatology",
            "years": f"{years[0]}-{years[-1]}",
            "template_year": args.template_year,
            "leap_day_note": "29 February averages leap years only",
        },
    )
    encoding = {"surface_temperature_climatology": {"zlib": True, "complevel": 1, "chunksizes": (32, len(lat), len(lon))}}
    out.to_netcdf(out_path, encoding=encoding)
    print(f"wrote {out_path}", flush=True)


def nc_dates(ds: nc.Dataset) -> list:
    tv = ds.variables["time"]
    return list(nc.num2date(tv[:], tv.units, calendar=getattr(tv, "calendar", "standard")))


def date_key(dt) -> tuple[int, int, int]:
    return int(dt.month), int(dt.day), int(dt.hour)


def delta_for_date(
    dt,
    mode: str,
    precursor_month: int | None,
    patterns: dict[int, np.ndarray],
    start: datetime,
    end: datetime,
) -> np.ndarray | None:
    stamp = datetime(int(dt.year), int(dt.month), int(dt.day), int(dt.hour))
    if stamp < start or stamp > end:
        return None
    if mode == "persistent":
        return patterns[int(precursor_month)]
    # Evolving experiment: use May anomaly for late-April lag ICs, then the
    # matching May/June/July/August pattern through the season.
    month = int(dt.month)
    return patterns[month] if month in (5, 6, 7, 8) else patterns[5]


def complete_forcing(path: Path, expected_time: int) -> bool:
    try:
        with nc.Dataset(path) as ds:
            return "surface_temperature" in ds.variables and len(ds.dimensions["time"]) == expected_time
    except Exception:
        return False


def create_control_forcing(
    source: Path,
    destination: Path,
    climatology: np.ndarray,
    overwrite: bool,
) -> None:
    with nc.Dataset(source) as src:
        ntime = len(src.dimensions["time"])
    if complete_forcing(destination, ntime) and not overwrite:
        print(f"using existing shared control forcing {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)
    with nc.Dataset(destination, "r+") as ds:
        st = ds.variables["surface_temperature"]
        oc = ds.variables["ocean_fraction"]
        ice = ds.variables["sea_ice_fraction"]
        for t0 in range(0, ntime, 32):
            t1 = min(t0 + 32, ntime)
            original = np.asarray(st[t0:t1], dtype=np.float32)
            ocean = np.asarray(oc[t0:t1], dtype=np.float32) > 0.5
            ice_free = np.asarray(ice[t0:t1], dtype=np.float32) <= 0.15
            mask = ocean & ice_free
            original[mask] = climatology[t0:t1][mask]
            st[t0:t1] = original
        ds.setncattr("sst_experiment_role", "1980-2022 climatological-SST control")
    print(f"wrote {destination}", flush=True)


def create_perturbation_forcing(
    control: Path,
    destination: Path,
    patterns: dict[int, np.ndarray],
    basin_mask: np.ndarray,
    mode: str,
    precursor_month: int | None,
    start: datetime,
    end: datetime,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Perturbation forcing exists; use --overwrite to rebuild: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(control, destination)
    with nc.Dataset(destination, "r+") as ds:
        dates = nc_dates(ds)
        st = ds.variables["surface_temperature"]
        oc = ds.variables["ocean_fraction"]
        ice = ds.variables["sea_ice_fraction"]
        for i, dt in enumerate(dates):
            delta = delta_for_date(dt, mode, precursor_month, patterns, start, end)
            if delta is None:
                continue
            valid = basin_mask & np.isfinite(delta)
            ocean = np.asarray(oc[i], dtype=np.float32) > 0.5
            ice_free = np.asarray(ice[i], dtype=np.float32) <= 0.15
            mask = valid & ocean & ice_free
            values = np.asarray(st[i], dtype=np.float32)
            values[mask] += delta[mask]
            st[i] = values
        ds.setncattr("sst_experiment_role", f"perturbation: {mode}")
        ds.setncattr("sst_experiment_precursor_month", -1 if precursor_month is None else precursor_month)
    print(f"wrote {destination}", flush=True)


def create_ic_pair(
    source_ic: Path,
    control_ic: Path,
    perturb_ic: Path,
    control_forcing: Path,
    patterns: dict[int, np.ndarray],
    basin_mask: np.ndarray,
    mode: str,
    precursor_month: int | None,
    start: datetime,
    end: datetime,
    overwrite_perturbation: bool,
    overwrite_control: bool,
) -> None:
    if not source_ic.exists():
        raise FileNotFoundError(source_ic)
    with nc.Dataset(control_forcing) as f:
        f_dates = nc_dates(f)
        lookup = {date_key(d): i for i, d in enumerate(f_dates)}
        forcing_st = f.variables["surface_temperature"]
        forcing_oc = f.variables["ocean_fraction"]
        forcing_ice = f.variables["sea_ice_fraction"]
        control_ic.parent.mkdir(parents=True, exist_ok=True)
        if not control_ic.exists() or overwrite_control:
            control_ic.unlink(missing_ok=True)
            shutil.copy2(source_ic, control_ic)
            with nc.Dataset(control_ic, "r+") as c:
                dates = nc_dates(c)
                st = c.variables["surface_temperature"]
                for i, dt in enumerate(dates):
                    j = lookup[date_key(dt)]
                    mask = (np.asarray(forcing_oc[j]) > 0.5) & (np.asarray(forcing_ice[j]) <= 0.15)
                    values = np.asarray(st[i], dtype=np.float32)
                    clim = np.asarray(forcing_st[j], dtype=np.float32)
                    values[mask] = clim[mask]
                    st[i] = values
                c.setncattr("sst_experiment_role", "climatological-SST control IC")
            print(f"wrote {control_ic}", flush=True)

        if perturb_ic.exists() and not overwrite_perturbation:
            raise FileExistsError(f"Perturbation IC exists; use --overwrite to rebuild: {perturb_ic}")
        perturb_ic.parent.mkdir(parents=True, exist_ok=True)
        perturb_ic.unlink(missing_ok=True)
        shutil.copy2(control_ic, perturb_ic)
        with nc.Dataset(perturb_ic, "r+") as p:
            dates = nc_dates(p)
            st = p.variables["surface_temperature"]
            for i, dt in enumerate(dates):
                delta = delta_for_date(dt, mode, precursor_month, patterns, start, end)
                if delta is None:
                    continue
                j = lookup[date_key(dt)]
                mask = basin_mask & np.isfinite(delta)
                mask &= (np.asarray(forcing_oc[j]) > 0.5) & (np.asarray(forcing_ice[j]) <= 0.15)
                values = np.asarray(st[i], dtype=np.float32)
                values[mask] += delta[mask]
                st[i] = values
            p.setncattr("sst_experiment_role", f"perturbation IC: {mode}")
        print(f"wrote {perturb_ic}", flush=True)


def prepare(args: argparse.Namespace) -> None:
    if args.mode == "persistent" and args.precursor_month not in (3, 4, 5):
        raise ValueError("Persistent experiments require --precursor-month 3, 4, or 5")
    if args.mode == "evolving" and args.init_month != 5:
        raise ValueError("The May-August evolving experiment is initialized in May")
    with xr.open_dataset(args.climatology) as ds:
        climatology = np.asarray(ds["surface_temperature_climatology"], dtype=np.float32)
        clim_lat = np.asarray(ds["latitude"], dtype=float)
        clim_lon = np.asarray(ds["longitude"], dtype=float) % 360.0
    with xr.open_dataset(args.composite) as ds:
        patterns = {
            m: np.asarray(ds[f"sst_high_minus_low_{MONTH_NAMES[m]}"], dtype=np.float32)
            for m in MONTH_NAMES
        }
        lat = np.asarray(ds["lat"], dtype=float)
        lon = np.asarray(ds["lon"], dtype=float) % 360.0
    if climatology.shape[1:] != patterns[3].shape or not (np.allclose(lat, clim_lat) and np.allclose(lon, clim_lon)):
        raise ValueError("Composite and climatology grids differ")

    basin_spec = BASINS[args.basin]
    basin = np.ones(patterns[3].shape, dtype=bool) if basin_spec is None else box_mask(lat, lon, basin_spec)
    center = datetime(args.base_year, args.init_month, args.init_day)
    start = center - timedelta(hours=72)
    end = datetime(args.base_year, 8, 31, 18)
    root = Path(args.out_root)
    control_dir = root / "inputs" / "control"
    perturb_dir = root / "inputs" / args.tag
    source_forcing = Path(args.forcing_dir) / f"forcing_{args.base_year}.nc"
    control_forcing = control_dir / "forcing" / source_forcing.name
    perturb_forcing = perturb_dir / "forcing" / source_forcing.name
    source_ic = Path(args.ic_dir) / f"ic_lag_{args.base_year}{args.init_month:02d}{args.init_day:02d}_25m.nc"
    control_ic = control_dir / "initial_conditions" / source_ic.name
    perturb_ic = perturb_dir / "initial_conditions" / source_ic.name

    create_control_forcing(source_forcing, control_forcing, climatology, args.overwrite_control)
    create_perturbation_forcing(
        control_forcing, perturb_forcing, patterns, basin, args.mode,
        args.precursor_month, start, end, args.overwrite,
    )
    create_ic_pair(
        source_ic, control_ic, perturb_ic, control_forcing, patterns, basin,
        args.mode, args.precursor_month, start, end, args.overwrite, args.overwrite_control,
    )
    manifest = {
        "tag": args.tag,
        "base_year": args.base_year,
        "initialization": f"{args.init_month:02d}-{args.init_day:02d}",
        "mode": args.mode,
        "precursor_month": args.precursor_month,
        "basin": args.basin,
        "basin_bounds": basin_spec,
        "control": {"forcing": str(control_forcing), "ic": str(control_ic)},
        "perturbation": {"forcing": str(perturb_forcing), "ic": str(perturb_ic)},
        "application_window": [start.isoformat(), end.isoformat()],
        "contrast": "perturbation = climatological SST + full ACE2 high-minus-low composite; control = climatological SST",
    }
    perturb_dir.mkdir(parents=True, exist_ok=True)
    (perturb_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("composite")
    c.add_argument("--ace2-hhe-nc", required=True)
    c.add_argument("--ace2-hhe-var", default="ace2_hi_freq")
    c.add_argument("--forcing-dir", required=True)
    c.add_argument("--years", default="1980:2022")
    c.add_argument("--strict-years", action="store_true")
    c.add_argument("--index-box", nargs=4, type=float, default=[23, 38, 260, 283])
    c.add_argument("--tail-fraction", type=float, default=0.20)
    c.add_argument("--ocean-min", type=float, default=0.5)
    c.add_argument("--max-sea-ice", type=float, default=0.15)
    c.add_argument("--out", required=True)
    c.add_argument("--figure", default=None)
    c.set_defaults(func=build_composite)

    k = sub.add_parser("climatology")
    k.add_argument("--forcing-dir", required=True)
    k.add_argument("--years", default="1980:2022")
    k.add_argument("--template-year", type=int, default=2000)
    k.add_argument("--out", required=True)
    k.set_defaults(func=build_climatology)

    q = sub.add_parser("prepare")
    q.add_argument("--composite", required=True)
    q.add_argument("--climatology", required=True)
    q.add_argument("--forcing-dir", required=True)
    q.add_argument("--ic-dir", required=True)
    q.add_argument("--out-root", required=True)
    q.add_argument("--tag", required=True)
    q.add_argument("--base-year", type=int, default=2000)
    q.add_argument("--init-month", type=int, required=True)
    q.add_argument("--init-day", type=int, default=1)
    q.add_argument("--mode", choices=("persistent", "evolving"), required=True)
    q.add_argument("--precursor-month", type=int, default=None)
    q.add_argument("--basin", choices=tuple(BASINS), default="global")
    q.add_argument("--overwrite", action="store_true", help="rebuild this perturbation's forcing and IC")
    q.add_argument("--overwrite-control", action="store_true", help="also rebuild the shared control forcing/IC")
    q.set_defaults(func=prepare)
    return p


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
