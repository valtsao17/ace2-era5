#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml
from tqdm.auto import tqdm


ROOT = Path("/home/jovyan")
ACE2S_REPO = ROOT / "hiro-ace-test"
DATA_DIR = ACE2S_REPO / "data" / "hiro_ace"
DEFAULT_PYTHON = ROOT / "ace2-era5" / ".conda" / "envs" / "ace2" / "bin" / "python"
PATCHED_INFERENCE = ACE2S_REPO / "scripts" / "07_run_ace2s_inference_with_ensemble_fix.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--work-root", default="outputs/targetdate_predictions")
    p.add_argument("--python", default=str(DEFAULT_PYTHON))
    p.add_argument("--inference-wrapper", default=str(PATCHED_INFERENCE))
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--members", type=int, default=25)
    p.add_argument("--chunk-size", type=int, default=5)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forward-steps-in-memory", type=int, default=10)
    p.add_argument("--num-data-workers", type=int, default=0)
    return p.parse_args()


def member_chunks(total: int, chunk_size: int) -> list[int]:
    out = []
    rem = total
    while rem > 0:
        take = min(chunk_size, rem)
        out.append(take)
        rem -= take
    return out


def prediction_complete(path: Path, expected_members: int, expected_steps: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            return (
                "TMP2m" in ds.data_vars
                and ds.sizes.get("sample") == expected_members
                and ds.sizes.get("time") >= expected_steps
            )
    except Exception:
        return False


def raw_chunk_complete(path: Path, expected_members: int, expected_steps: int) -> bool:
    return prediction_complete(path, expected_members, expected_steps)


def build_config(args, case_dir: Path, chunk_dir: Path, init_time: str, horizon_days: int, n_members: int):
    data_dir = Path(args.data_dir)
    return {
        "experiment_dir": str(chunk_dir.resolve()),
        "n_forward_steps": int(horizon_days * 4),
        "forward_steps_in_memory": int(args.forward_steps_in_memory),
        "checkpoint_path": str((data_dir / "ACE2S.ckpt").resolve()),
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ace",
        },
        "initial_condition": {
            "path": str((data_dir / "initial_conditions" / "ic_2014.nc").resolve()),
            "start_indices": {"times": [init_time]},
        },
        "forcing_loader": {
            "dataset": {"data_path": str((data_dir / "forcing_data").resolve())},
            "num_data_workers": int(args.num_data_workers),
        },
        "data_writer": {
            "save_prediction_files": True,
            "save_monthly_files": False,
            "names": ["TMP2m"],
        },
        "n_ensemble_per_ic": int(n_members),
        "allow_incompatible_dataset": False,
    }


def write_case_configs(args, row, case_dir: Path):
    init_time = pd.Timestamp(row["init_time"]).strftime("%Y-%m-%dT%H:%M:%S")
    horizon_days = int(row["required_horizon_days"])
    expected_steps = int(row["required_6hour_steps"])

    config_dir = case_dir / "configs"
    raw_dir = case_dir / "raw_chunks"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, n_members in enumerate(member_chunks(args.members, args.chunk_size)):
        chunk = f"chunk{idx:03d}"
        chunk_dir = raw_dir / chunk
        chunk_dir.mkdir(parents=True, exist_ok=True)

        cfg = build_config(args, case_dir, chunk_dir, init_time, horizon_days, n_members)
        config_path = config_dir / f"{chunk}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        rows.append(
            {
                "chunk": chunk,
                "members": n_members,
                "config_path": str(config_path),
                "output_dir": str(chunk_dir),
                "prediction_path": str(chunk_dir / "autoregressive_predictions.nc"),
                "expected_steps": expected_steps,
            }
        )

    manifest = config_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_chunk(args, chunk_row):
    pred = Path(chunk_row["prediction_path"])
    expected_members = int(chunk_row["members"])
    expected_steps = int(chunk_row["expected_steps"])

    if args.skip_existing and raw_chunk_complete(pred, expected_members, expected_steps):
        print(f"skip complete chunk: {pred}", flush=True)
        return

    cmd = [args.python, args.inference_wrapper, str(chunk_row["config_path"])]
    log_path = Path(chunk_row["output_dir"]) / "inference_subprocess.log"

    if args.dry_run:
        print("DRY:", " ".join(cmd))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"chunk failed rc={proc.returncode}. log={log_path}")


def combine_chunks(chunk_rows, out_path: Path, expected_members: int, expected_steps: int):
    if prediction_complete(out_path, expected_members, expected_steps):
        print(f"already complete final prediction: {out_path}", flush=True)
        return

    datasets = []
    try:
        for r in chunk_rows:
            p = Path(r["prediction_path"])
            if not raw_chunk_complete(p, int(r["members"]), expected_steps):
                raise RuntimeError(f"incomplete chunk prediction: {p}")
            datasets.append(xr.open_dataset(p, decode_times=False))

        combined = xr.concat(datasets, dim="sample")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        combined.to_netcdf(tmp)
        tmp.replace(out_path)

        print(f"wrote final prediction: {out_path}", flush=True)

        if not prediction_complete(out_path, expected_members, expected_steps):
            raise RuntimeError(f"combined file failed validation: {out_path}")
    finally:
        for ds in datasets:
            ds.close()


def main():
    args = parse_args()
    plan = pd.read_csv(args.plan)
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    required = [
        Path(args.python),
        Path(args.inference_wrapper),
        Path(args.data_dir) / "ACE2S.ckpt",
        Path(args.data_dir) / "initial_conditions" / "ic_2014.nc",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    for _, row in tqdm(plan.iterrows(), total=len(plan), desc="inference cases"):
        out_path = Path(row["prediction_path"]).resolve()
        expected_steps = int(row["required_6hour_steps"])

        if args.skip_existing and prediction_complete(out_path, args.members, expected_steps):
            print(f"skip complete final prediction: {out_path}", flush=True)
            continue

        target = str(row["target_date"]).replace("-", "")
        init = pd.Timestamp(row["init_time"]).strftime("%Y%m%d")
        lead = int(row["lead_days"])
        horizon = int(row["required_horizon_days"])

        case_dir = work_root / "runs" / f"target{target}_lead{lead:03d}_init{init}_horizon{horizon:03d}d"
        chunk_rows = write_case_configs(args, row, case_dir)

        for chunk_row in tqdm(chunk_rows, desc=f"chunks lead {lead:03d}", leave=False):
            run_chunk(args, chunk_row)

        if not args.dry_run:
            combine_chunks(chunk_rows, out_path, args.members, expected_steps)


if __name__ == "__main__":
    main()
