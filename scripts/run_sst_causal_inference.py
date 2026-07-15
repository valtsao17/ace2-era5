#!/usr/bin/env python3
"""Run a 25-member ACE2 SST-causality arm from March, April, or May.

This is the date-generic counterpart of run_lag_inference_may.py. Members are
six-hourly lag initializations centered on --init-month/--init-day. If
--n-steps is omitted, all members are run far enough for the earliest member to
reach 1 September, ensuring complete JJA coverage.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from datetime import datetime, timedelta
from pathlib import Path

from run_lag_inference_may import (
    DEFAULT_WRITER_NAMES,
    ATMO_FIG7_WRITER_NAMES,
    member_complete,
    run_one,
)

N_MEMBERS = 25


def parse_years(spec: str) -> list[int]:
    if ":" in spec:
        a, b = (int(x) for x in spec.split(":", 1))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def lag_times(year: int, month: int, day: int) -> list[datetime]:
    center = datetime(year, month, day)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def auto_steps(year: int, month: int, day: int) -> int:
    earliest = lag_times(year, month, day)[0]
    target = datetime(year, 9, 1)
    return int((target - earliest).total_seconds() // (6 * 3600))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--forcing-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--years", default="2000")
    p.add_argument("--init-month", type=int, required=True)
    p.add_argument("--init-day", type=int, default=1)
    p.add_argument("--members", default="all")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--forward-steps-in-memory", type=int, default=40)
    p.add_argument("--writer-names", default=DEFAULT_WRITER_NAMES)
    p.add_argument("--writer-preset", choices=("custom", "heat", "atmo_fig7"), default="heat")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--wrapper", default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    checkpoint = Path(args.checkpoint) if args.checkpoint else root / "checkpoint/ace2_era5_ckpt.tar"
    ic_dir = Path(args.ic_dir)
    forcing_dir = Path(args.forcing_dir)
    output_dir = Path(args.output_dir)
    wrapper = Path(args.wrapper) if args.wrapper else Path(__file__).parent / "07_run_ace2s_inference_with_ensemble_fix.py"
    years = parse_years(args.years)
    members = list(range(N_MEMBERS)) if args.members == "all" else [int(x) for x in args.members.split(",")]
    writer_spec = args.writer_names
    if args.writer_preset == "heat":
        writer_spec = DEFAULT_WRITER_NAMES
    elif args.writer_preset == "atmo_fig7":
        writer_spec = ATMO_FIG7_WRITER_NAMES
    writer_names = [x.strip() for x in writer_spec.split(",") if x.strip()]

    for path, label in ((checkpoint, "checkpoint"), (ic_dir, "IC directory"),
                        (forcing_dir, "forcing directory"), (wrapper, "inference wrapper")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for year in years:
        n_steps = args.n_steps or auto_steps(year, args.init_month, args.init_day)
        ic_path = ic_dir / f"ic_lag_{year}{args.init_month:02d}{args.init_day:02d}_25m.nc"
        forcing_path = forcing_dir / f"forcing_{year}.nc"
        if not ic_path.exists() or not forcing_path.exists():
            raise FileNotFoundError(f"Missing year {year} input: IC={ic_path.exists()} forcing={forcing_path.exists()}")
        times = lag_times(year, args.init_month, args.init_day)
        print(f"{year}: init={args.init_month:02d}-{args.init_day:02d} auto/selected steps={n_steps}", flush=True)
        for member in members:
            member_dir = output_dir / str(year) / f"member_{member:02d}"
            if args.skip_existing and member_complete(member_dir, n_steps, writer_names):
                print(f"[{year} member_{member:02d}] skip", flush=True)
                continue
            jobs.append((year, member, times[member], ic_path, n_steps))

    def execute(tagged):
        (year, member, init_time, ic_path, n_steps), gpu = tagged
        member_dir = output_dir / str(year) / f"member_{member:02d}"
        return run_one(
            args.python, wrapper, member_dir, ic_path, forcing_dir, checkpoint,
            init_time, args.forward_steps_in_memory, n_steps, writer_names,
            gpu, args.dry_run,
        )

    failures = []
    tagged_jobs = [(job, i % max(args.n_gpus, 1)) for i, job in enumerate(jobs)]
    if args.n_gpus <= 1:
        results = map(execute, tagged_jobs)
        for label, ok in results:
            if not ok:
                failures.append(label)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
            for label, ok in pool.map(execute, tagged_jobs):
                if not ok:
                    failures.append(label)
    if failures:
        raise RuntimeError(f"{len(failures)} inference jobs failed: {failures}")
    print("All requested SST-causality members complete.", flush=True)


if __name__ == "__main__":
    main()
