#!/usr/bin/env python3
"""Run 250 ACE2-ERA5 lag ensemble inference jobs for the 10-year (1980-1989) Nov 1 experiment.

For each year × member:
  - init: Nov 1 centre ± 12 × 6 h (25 members)
  - n_forward_steps: 500  (3000 h / 6 h)
  - output: outputs/lag_10yr/runs/{year}/member_{i:02d}/
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
ACE2_CKPT    = ROOT / "hiro-ace-test/data/hiro_ace/ace2_era5_ckpt.tar"
IC_DIR       = ROOT / "ace2_lag_data/initial_conditions"
FORCING_DIR  = ROOT / "ace2_lag_data/forcing_data_ace2era5"
WORK_ROOT    = ROOT / "hiro_ace_clean_v4/outputs/lag_10yr/runs"
WRAPPER      = ROOT / "hiro-ace-test/scripts/07_run_ace2s_inference_with_ensemble_fix.py"
DEFAULT_PY   = ROOT / "ace2-era5/.conda/envs/ace2/bin/python"

YEARS      = list(range(1980, 1990))
N_MEMBERS  = 25
N_STEPS    = 500   # 3000 h / 6 h


def lag_times(year: int) -> list[datetime]:
    center = datetime(year, 11, 1, 0, 0, 0)
    return [center + timedelta(hours=6 * (i - 12)) for i in range(N_MEMBERS)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=str(DEFAULT_PY))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forward-steps-in-memory", type=int, default=40)
    p.add_argument("--years", default="all",
                   help="'all' or comma-separated years, e.g. '1980,1981'")
    p.add_argument("--members", default="all",
                   help="'all' or comma-separated member indices")
    return p.parse_args()


def build_config(member_dir: Path, ic_path: Path, init_time: datetime,
                 fsim: int) -> dict:
    return {
        "experiment_dir": str(member_dir.resolve()),
        "n_forward_steps": N_STEPS,
        "forward_steps_in_memory": fsim,
        "checkpoint_path": str(ACE2_CKPT),
        "logging": {"log_to_screen": True, "log_to_wandb": False,
                    "log_to_file": True, "project": "ace"},
        "initial_condition": {
            "path": str(ic_path),
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


def run_one(args, year: int, idx: int, init_time: datetime):
    member_dir = WORK_ROOT / str(year) / f"member_{idx:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and member_complete(member_dir):
        print(f"[{year} m{idx:02d}] skip", flush=True)
        return

    ic_path = IC_DIR / f"ic_lag_{year}1101_25m.nc"
    cfg = build_config(member_dir, ic_path, init_time,
                       args.forward_steps_in_memory)
    config_path = member_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"[{year} m{idx:02d}] init={init_time.isoformat()} "
          f"steps={N_STEPS}", flush=True)

    if args.dry_run:
        print(f"  DRY: {args.python} {WRAPPER} {config_path}")
        return

    log_path = member_dir / "inference.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with log_path.open("w") as log:
        proc = subprocess.run(
            [args.python, str(WRAPPER), str(config_path)],
            stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"[{year} m{idx:02d}] failed rc={proc.returncode} log={log_path}"
        )
    print(f"[{year} m{idx:02d}] done", flush=True)


def main():
    args = parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (Path(args.python), "python"),
        (WRAPPER, "inference wrapper"),
        (ACE2_CKPT, "ACE2-ERA5 checkpoint"),
        (FORCING_DIR, "forcing directory"),
    ]:
        if not path.exists():
            print(f"ERROR: missing {label}: {path}", file=sys.stderr)
            sys.exit(1)

    years   = YEARS if args.years == "all" else [int(y) for y in args.years.split(",")]
    members = list(range(N_MEMBERS)) if args.members == "all" \
              else [int(x) for x in args.members.split(",")]

    for year in years:
        ic_path = IC_DIR / f"ic_lag_{year}1101_25m.nc"
        if not ic_path.exists():
            print(f"ERROR: missing IC file for {year}: {ic_path}", file=sys.stderr)
            sys.exit(1)
        times = lag_times(year)
        for idx in members:
            run_one(args, year, idx, times[idx])

    print("\nAll jobs complete.", flush=True)


if __name__ == "__main__":
    main()
