#!/usr/bin/env python3
"""Run ACE2-ERA5 lag ensemble inference for the Nov 1 experiment.

For each year × member:
  - init: Nov 1 centre ± 12 × 6 h (25 members)
  - n_forward_steps: 500  (3000 h / 6 h)
  - output: <output_dir>/{year}/member_{i:02d}/

Multi-GPU: pass --n-gpus N to run N members in parallel, one per GPU.
Each worker gets CUDA_VISIBLE_DEVICES set to its GPU index.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

YEARS     = list(range(1980, 1990))
N_MEMBERS = 25
N_STEPS   = 500   # 3000 h / 6 h


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home()),
                   help="Root directory containing data and outputs (default: $HOME)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to ace2_era5_ckpt.tar (default: <root>/checkpoint/ace2_era5_ckpt.tar)")
    p.add_argument("--ic-dir", default=None,
                   help="Directory of IC files (default: <root>/data/lag_data/initial_conditions)")
    p.add_argument("--forcing-dir", default=None,
                   help="Directory of forcing files (default: <root>/data/lag_data/forcing_data_ace2era5)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write run outputs (default: <root>/outputs/lag_10yr/runs)")
    p.add_argument("--wrapper", default=None,
                   help="Path to 07_run_ace2s_inference_with_ensemble_fix.py (default: alongside this script)")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable to use (default: current interpreter)")
    p.add_argument("--n-gpus", type=int, default=1,
                   help="Number of GPUs — runs this many members in parallel (default: 1)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forward-steps-in-memory", type=int, default=40)
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated years, e.g. '1980,1981'")
    p.add_argument("--members", default="all",
                   help="'all' or comma-separated member indices")
    return p.parse_args()


def resolve_paths(args):
    root = Path(args.root)
    ckpt      = Path(args.checkpoint)  if args.checkpoint  else root / "checkpoint/ace2_era5_ckpt.tar"
    ic_dir    = Path(args.ic_dir)      if args.ic_dir      else root / "data/lag_data/initial_conditions"
    force_dir = Path(args.forcing_dir) if args.forcing_dir else root / "data/lag_data/forcing_data_ace2era5"
    out_dir   = Path(args.output_dir)  if args.output_dir  else root / "outputs/lag_10yr/runs"
    wrapper   = Path(args.wrapper)     if args.wrapper     else Path(__file__).parent / "07_run_ace2s_inference_with_ensemble_fix.py"
    return ckpt, ic_dir, force_dir, out_dir, wrapper


def build_config(member_dir: Path, ic_path: Path, forcing_dir: Path,
                 ckpt: Path, init_time: datetime, fsim: int) -> dict:
    return {
        "experiment_dir": str(member_dir.resolve()),
        "n_forward_steps": N_STEPS,
        "forward_steps_in_memory": fsim,
        "checkpoint_path": str(ckpt),
        "logging": {"log_to_screen": True, "log_to_wandb": False,
                    "log_to_file": True, "project": "ace"},
        "initial_condition": {
            "path": str(ic_path),
            "start_indices": {"times": [init_time.isoformat()]},
        },
        "forcing_loader": {
            "dataset": {"data_path": str(forcing_dir)},
            "num_data_workers": 0,
        },
        "data_writer": {
            "save_prediction_files": True,
            "save_monthly_files": False,
            "names": ["TMP2m"],
        },
        "n_ensemble_per_ic": 1,
        "allow_incompatible_dataset": False,
    }


def member_complete(member_dir: Path) -> bool:
    pred = member_dir / "autoregressive_predictions.nc"
    if not pred.exists() or pred.stat().st_size == 0:
        return False
    try:
        import xarray as xr
        with xr.open_dataset(pred, decode_times=False) as ds:
            return "TMP2m" in ds.data_vars and ds.sizes.get("time", 0) >= N_STEPS
    except Exception:
        return False


def run_one(python: str, wrapper: Path, member_dir: Path,
            ic_path: Path, forcing_dir: Path, ckpt: Path,
            init_time: datetime, fsim: int,
            gpu_id: int, dry_run: bool) -> tuple[str, bool]:
    label = f"[{member_dir.parent.name} {member_dir.name}]"
    member_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(member_dir, ic_path, forcing_dir, ckpt, init_time, fsim)
    config_path = member_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"{label} init={init_time.isoformat()} steps={N_STEPS} gpu={gpu_id}", flush=True)

    if dry_run:
        print(f"  DRY: CUDA_VISIBLE_DEVICES={gpu_id} {python} {wrapper} {config_path}")
        return label, True

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]    = str(gpu_id)
    env["PYTHONUNBUFFERED"]        = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_path = member_dir / "inference.log"
    with log_path.open("w") as log:
        proc = subprocess.run(
            [python, str(wrapper), str(config_path)],
            stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
        )

    success = proc.returncode == 0
    status  = "done" if success else f"FAILED rc={proc.returncode} log={log_path}"
    print(f"{label} {status}", flush=True)
    return label, success


def main():
    args = parse_args()
    ckpt, ic_dir, forcing_dir, out_dir, wrapper = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (Path(args.python), "python"),
        (wrapper,     "inference wrapper"),
        (ckpt,        "ACE2-ERA5 checkpoint"),
        (forcing_dir, "forcing directory"),
    ]:
        if not path.exists():
            print(f"ERROR: missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    years   = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]
    members = list(range(N_MEMBERS)) if args.members == "all" \
              else [int(x) for x in args.members.split(",")]

    # Build job list, skipping completed members upfront
    jobs = []
    for year in years:
        ic_path = ic_dir / f"ic_lag_{year}1101_25m.nc"
        if not ic_path.exists():
            print(f"ERROR: missing IC file for {year}: {ic_path}", file=sys.stderr)
            sys.exit(1)
        times = lag_times(year)
        for idx in members:
            member_dir = out_dir / str(year) / f"member_{idx:02d}"
            if args.skip_existing and member_complete(member_dir):
                print(f"[{year} member_{idx:02d}] skip", flush=True)
                continue
            jobs.append((year, idx, times[idx], ic_path))

    if not jobs:
        print("Nothing to run.", flush=True)
        return

    failures = []

    if args.n_gpus <= 1:
        for year, idx, init_time, ic_path in jobs:
            member_dir = out_dir / str(year) / f"member_{idx:02d}"
            label, ok = run_one(
                args.python, wrapper, member_dir,
                ic_path, forcing_dir, ckpt, init_time,
                args.forward_steps_in_memory, gpu_id=0, dry_run=args.dry_run,
            )
            if not ok:
                failures.append(label)
    else:
        # Round-robin GPU assignment, n_gpus members running concurrently
        tagged = [(job, i % args.n_gpus) for i, job in enumerate(jobs)]

        def worker(job_and_gpu):
            (year, idx, init_time, ic_path), gpu_id = job_and_gpu
            member_dir = out_dir / str(year) / f"member_{idx:02d}"
            return run_one(
                args.python, wrapper, member_dir,
                ic_path, forcing_dir, ckpt, init_time,
                args.forward_steps_in_memory, gpu_id=gpu_id, dry_run=args.dry_run,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
            for label, ok in pool.map(worker, tagged):
                if not ok:
                    failures.append(label)

    if failures:
        print(f"\n{len(failures)} job(s) failed:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    print("\nAll jobs complete.", flush=True)


if __name__ == "__main__":
    main()
