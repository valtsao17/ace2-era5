#!/usr/bin/env python3
"""Benchmark ACE2S inference throughput on a T4 GPU.

Runs one member (init 2014-01-29T00:00, n_forward_steps=248) under four
configurations and reports wall-clock time, peak VRAM, and projected A100 times.

Usage:
    python benchmark_inference.py \
        --checkpoint /path/to/ace2_era5_ckpt.tar \
        --ic-file    /path/to/ic_2014.nc \
        --forcing    /path/to/forcing_data_ace2era5 \
        [--python    /path/to/python]  \
        [--wrapper   /path/to/07_run_ace2s_inference_with_ensemble_fix.py] \
        [--work-dir  /tmp/bench]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------
CONFIGS = [
    {"name": "fsim=40  workers=0 (baseline)", "forward_steps_in_memory": 40,  "num_data_workers": 0},
    {"name": "fsim=80  workers=0",             "forward_steps_in_memory": 80,  "num_data_workers": 0},
    {"name": "fsim=120 workers=0",             "forward_steps_in_memory": 120, "num_data_workers": 0},
    {"name": "fsim=40  workers=2",             "forward_steps_in_memory": 40,  "num_data_workers": 2},
]

INIT_TIME      = "2014-01-29T00:00:00"
N_FORWARD_STEPS = 248
T4_VRAM_MB     = 15360


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to ace2_era5_ckpt.tar (or ACE2S.ckpt)")
    p.add_argument("--ic-file", required=True,
                   help="Path to initial conditions .nc file containing 2014-01-29")
    p.add_argument("--forcing", required=True,
                   help="Path to forcing data directory")
    p.add_argument("--python",
                   default=sys.executable,
                   help="Python interpreter that has fme installed")
    p.add_argument("--wrapper",
                   default=str(Path(__file__).parent / "07_run_ace2s_inference_with_ensemble_fix.py"),
                   help="Path to 07_run_ace2s_inference_with_ensemble_fix.py")
    p.add_argument("--work-dir",
                   default=None,
                   help="Root directory for temp run dirs (default: system temp)")
    return p.parse_args()


def check_required_files(args: argparse.Namespace) -> None:
    missing = []
    for label, path in [
        ("checkpoint",    args.checkpoint),
        ("IC file",       args.ic_file),
        ("forcing dir",   args.forcing),
        ("python",        args.python),
        ("wrapper script",args.wrapper),
    ]:
        if not Path(path).exists():
            missing.append(f"  {label}: {path}")
    if missing:
        print("ERROR: required files not found:\n" + "\n".join(missing), file=sys.stderr)
        sys.exit(1)


def build_fme_config(run_dir: Path, ckpt: str, ic: str, forcing: str,
                     fsim: int, workers: int) -> dict:
    return {
        "experiment_dir":         str(run_dir.resolve()),
        "n_forward_steps":        N_FORWARD_STEPS,
        "forward_steps_in_memory": fsim,
        "checkpoint_path":        str(Path(ckpt).resolve()),
        "logging": {
            "log_to_screen": True,
            "log_to_wandb":  False,
            "log_to_file":   True,
            "project":       "ace",
        },
        "initial_condition": {
            "path":          str(Path(ic).resolve()),
            "start_indices": {"times": [INIT_TIME]},
        },
        "forcing_loader": {
            "dataset":         {"data_path": str(Path(forcing).resolve())},
            "num_data_workers": workers,
        },
        "data_writer": {
            "save_prediction_files": True,
            "save_monthly_files":    False,
            "names":                 ["TMP2m"],
        },
        "n_ensemble_per_ic":          1,
        "allow_incompatible_dataset": False,
    }


class VramMonitor(threading.Thread):
    """Poll nvidia-smi in a background thread; record peak VRAM (MB)."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval  = interval
        self.peak_mb   = 0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                for line in out.strip().splitlines():
                    try:
                        mb = int(line.strip())
                        if mb > self.peak_mb:
                            self.peak_mb = mb
                    except ValueError:
                        pass
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()


def run_config(cfg_meta: dict, args: argparse.Namespace, work_root: Path) -> dict:
    name    = cfg_meta["name"]
    fsim    = cfg_meta["forward_steps_in_memory"]
    workers = cfg_meta["num_data_workers"]

    run_dir = work_root / name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_fme_config(run_dir, args.checkpoint, args.ic_file,
                              args.forcing, fsim, workers)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    log_path = run_dir / "inference.log"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  config: {config_path}")
    print(f"  log:    {log_path}")
    print(f"{'='*60}", flush=True)

    vram_mon = VramMonitor(interval=0.5)
    vram_mon.start()
    t0 = time.perf_counter()

    status  = "OK"
    err_msg = ""
    try:
        with log_path.open("w") as log:
            proc = subprocess.run(
                [args.python, args.wrapper, str(config_path)],
                stdout=log, stderr=subprocess.STDOUT,
                text=True, env=env,
            )
        if proc.returncode != 0:
            status  = "FAILED"
            err_msg = f"returncode={proc.returncode}"
    except Exception as exc:
        status  = "FAILED"
        err_msg = str(exc)

    elapsed = time.perf_counter() - t0
    vram_mon.stop()
    vram_mon.join(timeout=2)

    peak_mb = vram_mon.peak_mb

    if status == "FAILED":
        # Check log tail for OOM clue
        try:
            tail = log_path.read_text()[-2000:]
            if "out of memory" in tail.lower() or "oom" in tail.lower():
                status = "OOM"
        except Exception:
            pass
        print(f"  => {status}: {err_msg}", flush=True)
    else:
        print(f"  => done in {elapsed/60:.1f} min, peak VRAM {peak_mb} MB", flush=True)

    return {
        "name":        name,
        "wall_min":    elapsed / 60.0,
        "peak_vram_mb": peak_mb,
        "status":      status,
        "err_msg":     err_msg,
        "log":         str(log_path),
    }


def print_summary(results: list[dict]) -> None:
    print("\n\n" + "=" * 78)
    print("BENCHMARK SUMMARY  (n_forward_steps={}, init={})".format(
        N_FORWARD_STEPS, INIT_TIME))
    print("=" * 78)

    hdr = f"{'Config':<30} {'Time(min)':>9} {'PeakVRAM(MB)':>13} {'%T4VRAM':>8} {'Status':<8}"
    print(hdr)
    print("-" * 78)

    for r in results:
        pct = (r["peak_vram_mb"] / T4_VRAM_MB * 100) if r["peak_vram_mb"] else 0.0
        wmin = f"{r['wall_min']:.1f}" if r["status"] == "OK" else "---"
        vmb  = str(r["peak_vram_mb"]) if r["peak_vram_mb"] else "---"
        pcts = f"{pct:.1f}%" if r["peak_vram_mb"] else "---"
        print(f"{r['name']:<30} {wmin:>9} {vmb:>13} {pcts:>8} {r['status']:<8}")
        if r["err_msg"]:
            print(f"  {'':30} error: {r['err_msg']}")

    print("=" * 78)

    ok = [r for r in results if r["status"] == "OK"]
    if ok:
        best = min(ok, key=lambda r: r["wall_min"])
        print(f"\nFastest OK config: {best['name']}")
        t = best["wall_min"]
        print(f"  T4  time : {t:.1f} min")
        print(f"  A100 est (÷3): {t/3:.1f} min")
        print(f"  A100 est (÷5): {t/5:.1f} min")

    print()
    for r in results:
        if r["status"] != "OK":
            print(f"  {r['name']} log: {r['log']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    check_required_files(args)

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="ace2_bench_"))
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {work_root}", flush=True)

    results = []
    for cfg_meta in CONFIGS:
        result = run_config(cfg_meta, args, work_root)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
