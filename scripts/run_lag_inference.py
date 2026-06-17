#!/usr/bin/env python3
"""Run 25 separate 1-member ACE2-ERA5 inference jobs for the lag ensemble experiment.

Each member is initialised at a different lag time centred on 2014-02-01T00:00 and
run for exactly the number of 6-hourly steps needed to reach the target date
2014-04-01T00:00.

  member  0: init 2014-01-29T00:00, 248 steps
  member 12: init 2014-02-01T00:00, 236 steps
  member 24: init 2014-02-03T00:00, 228 steps
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path("/home/jovyan")
ACE2_CKPT = ROOT / "hiro-ace-test/data/hiro_ace/ace2_era5_ckpt.tar"
IC_PATH = ROOT / "ace2_lag_data/initial_conditions/ic_lag_20140201_25m.nc"
FORCING_DIR = ROOT / "ace2_lag_data/forcing_data"
WORK_ROOT = ROOT / "hiro_ace_clean_v4/outputs/lag_experiment/runs"
INFERENCE_WRAPPER = ROOT / "hiro-ace-test/scripts/07_run_ace2s_inference_with_ensemble_fix.py"
DEFAULT_PYTHON = ROOT / "ace2-era5/.conda/envs/ace2/bin/python"

N_MEMBERS = 25
CENTER = datetime(2014, 2, 1, 0, 0, 0)
TARGET = datetime(2014, 4, 1, 0, 0, 0)
LAG_TIMES = [CENTER + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def n_steps_to_target(init: datetime) -> int:
    delta = (TARGET - init).total_seconds() / 3600
    return int(delta // 6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=str(DEFAULT_PYTHON))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forward-steps-in-memory", type=int, default=40)
    p.add_argument("--members", type=str, default="all",
                   help="'all' or comma-separated member indices, e.g. '0,1,2'")
    return p.parse_args()


def build_config(member_dir: Path, init_time: datetime, n_forward_steps: int,
                 forward_steps_in_memory: int) -> dict:
    return {
        "experiment_dir": str(member_dir.resolve()),
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": forward_steps_in_memory,
        "checkpoint_path": str(ACE2_CKPT),
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace",
        },
        "initial_condition": {
            "path": str(IC_PATH),
            "start_indices": {"times": [init_time.strftime("%Y-%m-%dT%H:%M:%S")]},
        },
        "forcing_loader": {
            "dataset": {"data_path": str(FORCING_DIR)},
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


def member_complete(member_dir: Path, expected_steps: int) -> bool:
    pred = member_dir / "autoregressive_predictions.nc"
    if not pred.exists() or pred.stat().st_size == 0:
        return False
    try:
        import xarray as xr
        with xr.open_dataset(pred, decode_times=False) as ds:
            return (
                "TMP2m" in ds.data_vars
                and ds.sizes.get("time", 0) >= expected_steps
            )
    except Exception:
        return False


def run_member(args, idx: int, init_time: datetime):
    n_steps = n_steps_to_target(init_time)
    member_dir = WORK_ROOT / f"member_{idx:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and member_complete(member_dir, n_steps):
        print(f"[member {idx:02d}] skip (complete)", flush=True)
        return

    cfg = build_config(member_dir, init_time, n_steps, args.forward_steps_in_memory)
    config_path = member_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(
        f"[member {idx:02d}] init={init_time.isoformat()} n_steps={n_steps}",
        flush=True,
    )

    if args.dry_run:
        print(f"  DRY RUN: {args.python} {INFERENCE_WRAPPER} {config_path}")
        return

    log_path = member_dir / "inference.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with log_path.open("w") as log:
        proc = subprocess.run(
            [args.python, str(INFERENCE_WRAPPER), str(config_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"member {idx:02d} failed rc={proc.returncode}, log={log_path}"
        )

    print(f"[member {idx:02d}] done", flush=True)


def main():
    args = parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (Path(args.python), "python"),
        (INFERENCE_WRAPPER, "inference wrapper"),
        (ACE2_CKPT, "ACE2-ERA5 checkpoint"),
        (IC_PATH, "lag IC file"),
        (FORCING_DIR / "forcing_2014.nc", "forcing_2014.nc"),
    ]:
        if not path.exists():
            print(f"ERROR: missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    if args.members == "all":
        indices = list(range(N_MEMBERS))
    else:
        indices = [int(x) for x in args.members.split(",")]

    for idx in indices:
        run_member(args, idx, LAG_TIMES[idx])

    print("\nAll members complete.", flush=True)


if __name__ == "__main__":
    main()
