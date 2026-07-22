#!/usr/bin/env python3
"""Report restart-safe progress for the multiyear SST causal experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import xarray as xr


EXPERIMENTS = (
    ("persistent_mar_global", "control_mar", 3),
    ("persistent_apr_global", "control_apr", 4),
    ("persistent_jun_global", "control_jun", 6),
    ("persistent_may_global", "control_may", 5),
    ("evolving_may_aug_global", "control_may", 5),
    ("evolving_may_aug_tropical_pacific", "control_may", 5),
    ("evolving_may_aug_north_pacific", "control_may", 5),
    ("evolving_may_aug_tropical_atlantic", "control_may", 5),
    ("evolving_may_aug_north_atlantic", "control_may", 5),
)
HHE_REQUIRED = {
    "high_minus_low_hhe_frequency",
    "regional_effect_by_case",
    "case_year",
    "case_member",
}
TEMPERATURE_REQUIRED = {
    "high_jja_6hourly_temperature_mean",
    "low_jja_6hourly_temperature_mean",
    "high_minus_low_jja_6hourly_temperature",
    "case_matched_jja_6hourly_steps",
    "regional_effect_by_case",
    "case_year",
    "case_member",
}


def parse_years(specification: str) -> list[int]:
    if ":" in specification:
        start, end = (int(value) for value in specification.split(":", 1))
        if end < start:
            raise ValueError("Year range end precedes start")
        return list(range(start, end + 1))
    values = [int(value) for value in specification.split(",") if value.strip()]
    if not values:
        raise ValueError("No years supplied")
    return values


def count_predictions(run_root: Path, year: int) -> int:
    return sum(
        path.is_file() and path.stat().st_size > 0
        for path in (run_root / str(year)).glob(
            "member_??/autoregressive_predictions.nc"
        )
    )


def count_failed_member_logs(run_root: Path, year: int) -> int:
    failed = 0
    for path in (run_root / str(year)).glob("member_??/inference.log"):
        try:
            tail = path.read_text(errors="replace")[-20_000:]
        except OSError:
            continue
        if "Traceback (most recent call last)" in tail or "FAILED" in tail:
            failed += 1
    return failed


def valid_dataset(path: Path, required: set[str]) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing"
    try:
        with xr.open_dataset(path) as dataset:
            missing = required.difference(dataset.data_vars)
            if missing:
                return False, "legacy" if path.name.endswith("_raw_tmax.nc") else "invalid"
            if dataset.sizes.get("case", 0) == 0:
                return False, "invalid"
    except Exception:
        return False, "corrupt"
    return True, "valid"


def input_ready(exp_root: Path, tag: str, control: str, year: int, month: int) -> bool:
    ic_name = f"ic_lag_{year}{month:02d}01_25m.nc"
    required = (
        exp_root / "inputs" / tag / "forcing" / f"forcing_{year}.nc",
        exp_root / "inputs" / tag / "initial_conditions" / ic_name,
        exp_root / "inputs" / "control" / "forcing" / f"forcing_{year}.nc",
        exp_root / "inputs" / "control" / "initial_conditions" / ic_name,
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def inspect_experiment(
    exp_root: Path,
    year: int,
    tag: str,
    control: str,
    month: int,
    min_members: int,
) -> dict[str, object]:
    hhe_path = exp_root / "evaluation_by_year" / tag / f"{tag}_{year}.nc"
    temperature_path = (
        exp_root
        / "evaluation_raw_tmax_by_year"
        / tag
        / f"{tag}_{year}_raw_tmax.nc"
    )
    cleanup_path = (
        exp_root / "cleanup_manifests" / tag / f"{tag}_{year}.json"
    )
    hhe_valid, hhe_detail = valid_dataset(hhe_path, HHE_REQUIRED)
    temperature_valid, temperature_detail = valid_dataset(
        temperature_path, TEMPERATURE_REQUIRED
    )
    high_root = exp_root / "runs" / tag
    low_root = exp_root / "runs" / control
    high_predictions = count_predictions(high_root, year)
    low_predictions = count_predictions(low_root, year)
    failed_logs = count_failed_member_logs(high_root, year) + count_failed_member_logs(
        low_root, year
    )
    cleaned = cleanup_path.is_file() and hhe_valid and temperature_valid

    if cleaned:
        state = "CLEANED"
    elif hhe_valid and temperature_valid:
        state = "EVALUATED"
    elif temperature_detail == "legacy":
        state = "LEGACY_TEMP"
    elif high_predictions >= min_members and low_predictions >= min_members:
        state = "READY_EVAL"
    elif high_predictions or low_predictions:
        state = "PARTIAL_RUN"
    elif input_ready(exp_root, tag, control, year, month):
        state = "PREPARED"
    else:
        state = "NOT_STARTED"

    return {
        "year": year,
        "tag": tag,
        "control": control,
        "state": state,
        "high_predictions": high_predictions,
        "control_predictions": low_predictions,
        "failed_member_logs": failed_logs,
        "hhe_artifact": hhe_detail,
        "temperature_artifact": temperature_detail,
        "cleanup_manifest": cleanup_path.is_file(),
    }


def active_slurm_jobs() -> str:
    user = os.environ.get("USER", "")
    if not user:
        return ""
    command = [
        "squeue",
        "-h",
        "-u",
        user,
        "-n",
        "sst_year_prepare,sst_causal_v2,sst_eval_clean,sst_finalize",
        "-o",
        "%12i  %-18j  %-2t  %-10M  %R",
    ]
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/home/vt55/ace2")
    )
    parser.add_argument("--years", default="1995:2005")
    parser.add_argument("--min-members", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-slurm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exp_root = args.root / "outputs" / "lag_may" / "sst_causal_v2"
    records = [
        inspect_experiment(
            exp_root, year, tag, control, month, args.min_members
        )
        for year in parse_years(args.years)
        for tag, control, month in EXPERIMENTS
    ]
    if args.json:
        print(json.dumps(records, indent=2))
        return

    print(
        f"{'YEAR':4}  {'EXPERIMENT':41}  {'STATE':11}  "
        f"{'PRED H/C':>9}  {'FAIL':>4}  {'HHE':>7}  {'TEMP':>7}"
    )
    print("-" * 102)
    for record in records:
        print(
            f"{record['year']:4d}  {record['tag']:<41}  "
            f"{record['state']:<11}  "
            f"{record['high_predictions']:>2}/{record['control_predictions']:<2}"
            f"{'':>4}  {record['failed_member_logs']:>4}  "
            f"{record['hhe_artifact']:>7}  {record['temperature_artifact']:>7}"
        )

    counts: dict[str, int] = {}
    for record in records:
        state = str(record["state"])
        counts[state] = counts.get(state, 0) + 1
    print("\nSummary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if not args.no_slurm:
        jobs = active_slurm_jobs()
        print("\nActive workflow jobs:")
        print(jobs or "(none)")


if __name__ == "__main__":
    main()
